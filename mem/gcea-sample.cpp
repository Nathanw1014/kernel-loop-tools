// gcea-sample: device-wide per-instance GCEA (memory interface) counters, no root.
//
// Uses the rocprofiler-sdk device counting service (agent profiling): this process only opens KFD
// and samples hardware-global counters. Coverage: counts HIP / KFD work (of any process). Vulkan
// work may read zero depending on the driver (a driver that dispatches with compute perf counting
// disabled is invisible here). Verify with camp-hip (a known camping pattern) before trusting
// either a zero or a balance verdict.
//
// usage: gcea-sample <out.tsv> <pid|0> [period_ms=100] [counter[,counter...]]...
//   Each counter group (one argv word, comma separated) is one hardware pass. Groups are
//   round-robined: start context (counters reset) -> sleep period -> sample -> stop.
//   Default groups: GCEA_RDRAM_SIZE_REQ  GCEA_WDRAM_SIZE_REQ (one event per GCEA instance
//   per pass).
//   pid > 0: stop when that pid exits; pid 0: stop on SIGTERM / SIGINT.
// output: TSV rows "t0 t1 counter v0 v1 ... vN" (raw counts in the [t0,t1) window, seconds
//   since the sampler start on CLOCK_MONOTONIC), '#' header lines with metadata.
//
// build: build-gcea-sample.sh [out_dir] (ROCm root from ROCM_PATH); run via chan-run.sh / chan-start.sh.

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>
#include <hsa/hsa.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <sys/types.h>
#include <signal.h>
#include <errno.h>
#include <time.h>

#define CHECK(x, msg)                                                                         \
    do {                                                                                      \
        rocprofiler_status_t s_ = (x);                                                        \
        if (s_ != ROCPROFILER_STATUS_SUCCESS) {                                               \
            fprintf(stderr, "gcea-sample: %s failed: %s\n", msg, rocprofiler_get_status_string(s_)); \
            return -1;                                                                        \
        }                                                                                     \
    } while (0)

namespace {
rocprofiler_context_id_t        g_ctx     = {};
rocprofiler_agent_id_t          g_agent   = {};
rocprofiler_counter_config_id_t g_profile = {.handle = 0};
std::atomic<bool>               g_stop{false};
bool                            g_init_ok = false;
std::string                     g_agent_name;

double now_s() {
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int tool_init(rocprofiler_client_finalize_t, void*) {
    std::vector<rocprofiler_agent_v0_t> agents;
    CHECK(rocprofiler_query_available_agents(
              ROCPROFILER_AGENT_INFO_VERSION_0,
              [](rocprofiler_agent_version_t, const void** arr, size_t n, void* ud) {
                  auto* v = static_cast<std::vector<rocprofiler_agent_v0_t>*>(ud);
                  for (size_t i = 0; i < n; ++i) {
                      auto* a = static_cast<const rocprofiler_agent_v0_t*>(arr[i]);
                      if (a->type == ROCPROFILER_AGENT_TYPE_GPU) v->push_back(*a);
                  }
                  return ROCPROFILER_STATUS_SUCCESS;
              },
              sizeof(rocprofiler_agent_v0_t), &agents),
          "query agents");
    if (agents.empty()) { fprintf(stderr, "gcea-sample: no GPU agent\n"); return -1; }
    g_agent      = agents[0].id;
    g_agent_name = agents[0].name ? agents[0].name : "?";
    CHECK(rocprofiler_create_context(&g_ctx), "create context");
    CHECK(rocprofiler_configure_device_counting_service(
              g_ctx, rocprofiler_buffer_id_t{.handle = 0}, g_agent,
              [](rocprofiler_context_id_t ctx, rocprofiler_agent_id_t,
                 rocprofiler_device_counting_agent_cb_t set_config, void*) {
                  if (g_profile.handle != 0) set_config(ctx, g_profile);
              },
              nullptr),
          "configure device counting");
    g_init_ok = true;
    return 0;
}
void tool_fini(void*) {}

std::unordered_map<std::string, rocprofiler_counter_id_t> supported() {
    std::vector<rocprofiler_counter_id_t> ids;
    std::unordered_map<std::string, rocprofiler_counter_id_t> out;
    rocprofiler_iterate_agent_supported_counters(
        g_agent,
        [](rocprofiler_agent_id_t, rocprofiler_counter_id_t* c, size_t n, void* ud) {
            auto* v = static_cast<std::vector<rocprofiler_counter_id_t>*>(ud);
            for (size_t i = 0; i < n; ++i) v->push_back(c[i]);
            return ROCPROFILER_STATUS_SUCCESS;
        },
        &ids);
    for (auto id : ids) {
        rocprofiler_counter_info_v0_t info;
        if (rocprofiler_query_counter_info(id, ROCPROFILER_COUNTER_INFO_VERSION_0, &info) ==
            ROCPROFILER_STATUS_SUCCESS)
            out.emplace(info.name, id);
    }
    return out;
}

struct group {
    std::string                           label;     // as given on the command line
    std::vector<std::string>              names;
    std::vector<rocprofiler_counter_id_t> ids;
    rocprofiler_counter_config_id_t       cfg{};
    size_t                                nrec = 0;
};

void on_sig(int) { g_stop = true; }
}  // namespace

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t* id) {
    id->name = "gcea-sample";
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};
    return &cfg;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: gcea-sample <out.tsv> <pid|0> [period_ms=100] [ctr[,ctr..]]...\n");
        return 2;
    }
    const char* outp   = argv[1];
    pid_t       pid    = (pid_t)atol(argv[2]);
    int         period = argc > 3 ? atoi(argv[3]) : 100;
    if (period < 10) period = 10;
    std::vector<std::string> glabels;
    for (int i = 4; i < argc; ++i) glabels.push_back(argv[i]);
    if (glabels.empty()) glabels = {"GCEA_RDRAM_SIZE_REQ", "GCEA_WDRAM_SIZE_REQ"};

    signal(SIGTERM, on_sig);
    signal(SIGINT, on_sig);

    if (hsa_init() != HSA_STATUS_SUCCESS) { fprintf(stderr, "gcea-sample: hsa_init failed\n"); return 3; }
    if (!g_init_ok) { fprintf(stderr, "gcea-sample: rocprofiler tool init failed\n"); return 3; }

    auto sup = supported();
    std::vector<group> groups;
    for (auto& gl : glabels) {
        group g;
        g.label = gl;
        size_t p = 0;
        while (p <= gl.size()) {
            size_t q = gl.find(',', p);
            if (q == std::string::npos) q = gl.size();
            std::string n = gl.substr(p, q - p);
            if (!n.empty()) {
                auto it = sup.find(n);
                if (it == sup.end()) { fprintf(stderr, "gcea-sample: counter %s not supported\n", n.c_str()); return 4; }
                g.names.push_back(n);
                g.ids.push_back(it->second);
                rocprofiler_counter_info_v1_t info;
                rocprofiler_query_counter_info(it->second, ROCPROFILER_COUNTER_INFO_VERSION_1, &info);
                g.nrec += info.dimensions_instances_count;
            }
            p = q + 1;
        }
        CHECK(rocprofiler_create_counter_config(g_agent, g.ids.data(), g.ids.size(), &g.cfg),
              "create counter config");
        groups.push_back(g);
    }

    FILE* f = fopen(outp, "w");
    if (!f) { perror(outp); return 5; }
    fprintf(f, "# gcea-sample agent=%s period_ms=%d pid=%d groups=", g_agent_name.c_str(), period, (int)pid);
    for (size_t i = 0; i < groups.size(); ++i) fprintf(f, "%s%s", i ? " " : "", groups[i].label.c_str());
    fprintf(f, "\n# columns: t0 t1 counter v0..vN (raw counts in window; t = s since sampler start)\n");
    fflush(f);

    const double t_start = now_s();
    std::vector<rocprofiler_counter_record_t> recs;
    std::map<uint64_t, std::string> idname;
    for (auto& g : groups)
        for (size_t i = 0; i < g.ids.size(); ++i) idname[g.ids[i].handle] = g.names[i];

    size_t gi = 0;
    int    rc = 0;
    while (!g_stop) {
        if (pid > 0 && kill(pid, 0) != 0 && errno == ESRCH) break;
        group& g  = groups[gi];
        g_profile = g.cfg;
        recs.assign(g.nrec, rocprofiler_counter_record_t{});
        double t0 = now_s();
        if (rocprofiler_start_context(g_ctx) != ROCPROFILER_STATUS_SUCCESS) {
            fprintf(stderr, "gcea-sample: start_context failed\n"); rc = 6; break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(period));
        size_t n  = recs.size();
        auto   st = rocprofiler_sample_device_counting_service(g_ctx, {}, ROCPROFILER_COUNTER_FLAG_NONE,
                                                               recs.data(), &n);
        double t1 = now_s();
        rocprofiler_stop_context(g_ctx);
        if (st != ROCPROFILER_STATUS_SUCCESS) {
            fprintf(stderr, "gcea-sample: sample failed: %s\n", rocprofiler_get_status_string(st));
            rc = 7; break;
        }
        // records -> per counter, per instance position
        std::map<std::string, std::vector<double>> rows;
        for (size_t i = 0; i < n; ++i) {
            rocprofiler_counter_id_t cid{};
            rocprofiler_query_record_counter_id(recs[i].id, &cid);
            size_t pos = 0;
            rocprofiler_counter_info_v1_t info;
            if (rocprofiler_query_counter_info(cid, ROCPROFILER_COUNTER_INFO_VERSION_1, &info) ==
                    ROCPROFILER_STATUS_SUCCESS && info.dimensions_count > 0)
                rocprofiler_query_record_dimension_position(recs[i].id, info.dimensions[0]->id, &pos);
            auto& v = rows[idname.count(cid.handle) ? idname[cid.handle] : std::to_string(cid.handle)];
            if (v.size() <= pos) v.resize(pos + 1, 0.0);
            v[pos] += recs[i].counter_value;
        }
        for (auto& [name, v] : rows) {
            fprintf(f, "%.4f\t%.4f\t%s", t0 - t_start, t1 - t_start, name.c_str());
            for (double x : v) fprintf(f, "\t%.0f", x);
            fprintf(f, "\n");
        }
        fflush(f);
        gi = (gi + 1) % groups.size();
    }
    fclose(f);
    hsa_shut_down();
    return rc;
}
