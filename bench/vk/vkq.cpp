// vkq: do two Vulkan queues run work at the same time? Plain Vulkan, no ggml. [AMD: picks the first vendor 0x1002 device]
// Guards against assuming that a second queue (or a second VkDevice, or a higher global priority) buys overlap: the
// RESULT line says CONCURRENT / PARTIAL / SERIAL from measured GPU timestamps, not from what the API allows.
//
// usage: VKQ_SPV=bench/vk/spin.spv vkq        (all knobs are env, below; -h prints this header)
//   e.g. VKQ_A=0:c:0 VKQ_B=0:c:1 vkq          two queues of the compute family on one device
//        VKQ_A=0:c:0 VKQ_B=1:c:0 vkq          the same queue index on two VkDevices
// Control: VKQ_DELAY_US well above solo A (e.g. 5x the "solo A" us it prints) must report SERIAL with "B start - A end" > 0;
// if it does not, the timestamp calibration is off and no verdict from this build/driver is usable.
// The default B lane 0:c:2 needs a compute family with >= 3 queues; check the "compute family N xQ" line.
//
// Each "lane" = (VkDevice d, family c|g, queue index i), written D:F:I (e.g. 0:c:0, 0:c:2, 1:c:0, 0:g:0).
// A lane's submit = [reset queries, timestamp TOP, NDISP x (spin dispatch, compute->compute barrier), timestamp BOTTOM].
// Modes per sample (order rotates): soloA, soloB, conc (A and B from two threads, B released VKQ_DELAY_US after A).
// GPU timestamps are mapped to host CLOCK_MONOTONIC_RAW with VK_KHR/EXT_calibrated_timestamps, so the host submit time of B
// and its GPU start can be compared directly.
// env: VKQ_A VKQ_B lanes (default 0:c:0 / 0:c:2); VKQ_PRIO0 / VKQ_PRIO1 global priority of device 0 / 1 queues
//      (none|low|medium|high|realtime); VKQ_SHARED=1 B writes into A's buffer (same device only); VKQ_WG workgroups per dispatch
//      (1); VKQ_ITERS spin iterations x4 FMA (A), VKQ_ITERS_B (B, default = A); VKQ_NDISP dispatches per submit (1);
//      VKQ_DELAY_US (1000); VKQ_N samples (20); VKQ_SPV spin.spv path (default ./spin.spv);
//      VKQ_BDA=1 buffers with device addresses; VKQ_HOSTMEM=1 host (non device-local) memory; VKQ_BIG_MB=n extra untouched
//      device-local allocations on lane 0:c:0 (does residency of unrelated memory change the result?).
// build: glslc -O --target-env=vulkan1.3 spin.comp -o spin.spv && g++ -O2 -pthread -o vkq vkq.cpp -lvulkan (or bench/vk/build.sh)
#include <vulkan/vulkan.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <time.h>
#include <vector>

#define CK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { fprintf(stderr, "FAIL %s = %d (line %d)\n", #x, (int) r_, __LINE__); exit(3); } } while (0)

static int64_t now_ns() { timespec t; clock_gettime(CLOCK_MONOTONIC_RAW, &t); return (int64_t) t.tv_sec * 1000000000ll + t.tv_nsec; }
static const char * envs(const char * k, const char * d) { const char * v = getenv(k); return v && *v ? v : d; }
static long envi(const char * k, long d) { const char * v = getenv(k); return v && *v ? atol(v) : d; }

struct dev_t_ {
    VkDevice dev = VK_NULL_HANDLE;
    VkPipeline pipe; VkPipelineLayout pl; VkDescriptorSetLayout dsl; VkShaderModule sm;
    PFN_vkGetCalibratedTimestampsEXT calib = nullptr;
    bool used = false;
};
struct lane_t {
    int d; char f; uint32_t idx; uint32_t fam;
    VkQueue q; VkCommandPool pool; VkCommandBuffer cb; VkQueryPool qp; VkFence fence;
    VkBuffer buf; VkDeviceMemory mem; VkDescriptorPool dp; VkDescriptorSet ds;
    uint32_t iters;
};

static VkInstance inst; static VkPhysicalDevice pd; static uint32_t fam_c = ~0u, fam_g = ~0u, nq_c = 0;
static VkPhysicalDeviceProperties props; static std::vector<uint32_t> spv; static dev_t_ devs[2];
static double period;

static VkQueueGlobalPriorityKHR prio_of(const char * s, bool * set) {
    *set = true;
    if (!strcmp(s, "low")) return VK_QUEUE_GLOBAL_PRIORITY_LOW_KHR;
    if (!strcmp(s, "medium")) return VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
    if (!strcmp(s, "high")) return VK_QUEUE_GLOBAL_PRIORITY_HIGH_KHR;
    if (!strcmp(s, "realtime")) return VK_QUEUE_GLOBAL_PRIORITY_REALTIME_KHR;
    *set = false; return VK_QUEUE_GLOBAL_PRIORITY_MEDIUM_KHR;
}

static bool has_ext(const char * name) {
    uint32_t n = 0; vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> e(n); vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, e.data());
    for (auto & x : e) if (!strcmp(x.extensionName, name)) return true;
    return false;
}

static void make_device(int di, const char * prio_s) {
    dev_t_ & D = devs[di];
    bool pset; VkQueueGlobalPriorityKHR prio = prio_of(prio_s, &pset);
    VkDeviceQueueGlobalPriorityCreateInfoKHR gp{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_GLOBAL_PRIORITY_CREATE_INFO_KHR, nullptr, prio };
    std::vector<float> pr(16, 1.0f);
    VkDeviceQueueCreateInfo qci[2]{};
    qci[0] = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, pset ? &gp : nullptr, 0, fam_c, nq_c, pr.data() };
    qci[1] = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, pset ? &gp : nullptr, 0, fam_g, 1, pr.data() };
    std::vector<const char *> ext;
    const char * cal = has_ext("VK_KHR_calibrated_timestamps") ? "VK_KHR_calibrated_timestamps" : "VK_EXT_calibrated_timestamps";
    ext.push_back(cal);
    if (pset) ext.push_back(has_ext("VK_KHR_global_priority") ? "VK_KHR_global_priority" : "VK_EXT_global_priority");
    VkPhysicalDeviceVulkan12Features f12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    f12.bufferDeviceAddress = envi("VKQ_BDA", 0) ? VK_TRUE : VK_FALSE;
    VkDeviceCreateInfo dci{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.pNext = &f12;
    dci.queueCreateInfoCount = 2; dci.pQueueCreateInfos = qci;
    dci.enabledExtensionCount = (uint32_t) ext.size(); dci.ppEnabledExtensionNames = ext.data();
    VkResult r = vkCreateDevice(pd, &dci, nullptr, &D.dev);
    printf("device %d: global priority %s -> vkCreateDevice %d\n", di, prio_s, (int) r);
    if (r != VK_SUCCESS) { printf("RESULT device-create-failed dev=%d prio=%s vkresult=%d\n", di, prio_s, (int) r); exit(4); }
    D.calib = (PFN_vkGetCalibratedTimestampsEXT) vkGetDeviceProcAddr(D.dev, !strcmp(cal, "VK_KHR_calibrated_timestamps") ? "vkGetCalibratedTimestampsKHR" : "vkGetCalibratedTimestampsEXT");
    VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, nullptr, 0, spv.size() * 4, spv.data() };
    CK(vkCreateShaderModule(D.dev, &smi, nullptr, &D.sm));
    VkDescriptorSetLayoutBinding b{ 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo dli{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, nullptr, 0, 1, &b };
    CK(vkCreateDescriptorSetLayout(D.dev, &dli, nullptr, &D.dsl));
    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 8 };
    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, nullptr, 0, 1, &D.dsl, 1, &pcr };
    CK(vkCreatePipelineLayout(D.dev, &pli, nullptr, &D.pl));
    VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpi.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, D.sm, "main", nullptr };
    cpi.layout = D.pl;
    CK(vkCreateComputePipelines(D.dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &D.pipe));
    D.used = true;
}

static lane_t parse_lane(const char * s) {
    lane_t L{}; char f = 'c'; int d = 0; unsigned i = 0;
    if (sscanf(s, "%d:%c:%u", &d, &f, &i) != 3 || d < 0 || d > 1 || (f != 'c' && f != 'g')) { fprintf(stderr, "bad lane %s\n", s); exit(2); }
    L.d = d; L.f = f; L.idx = i; L.fam = f == 'c' ? fam_c : fam_g;
    return L;
}

static void setup_lane(lane_t & L, uint32_t wg, uint32_t ndisp, const lane_t * share, uint32_t off) {
    VkDevice dev = devs[L.d].dev;
    vkGetDeviceQueue(dev, L.fam, L.idx, &L.q);
    VkCommandPoolCreateInfo cpi{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, nullptr, 0, L.fam };
    CK(vkCreateCommandPool(dev, &cpi, nullptr, &L.pool));
    VkCommandBufferAllocateInfo cai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, L.pool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
    CK(vkAllocateCommandBuffers(dev, &cai, &L.cb));
    VkQueryPoolCreateInfo qpi{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, nullptr, 0, VK_QUERY_TYPE_TIMESTAMP, 2 };
    CK(vkCreateQueryPool(dev, &qpi, nullptr, &L.qp));
    VkFenceCreateInfo fci{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    CK(vkCreateFence(dev, &fci, nullptr, &L.fence));
    const VkDeviceSize bytes = (VkDeviceSize) wg * 64 * 4 * 2;
    if (share) { L.buf = share->buf; L.mem = share->mem; }
    else {
        const bool bda = envi("VKQ_BDA", 0) != 0, hostm = envi("VKQ_HOSTMEM", 0) != 0;
        VkBufferCreateInfo bci{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, bytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | (bda ? VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT : 0u), VK_SHARING_MODE_EXCLUSIVE };
        CK(vkCreateBuffer(dev, &bci, nullptr, &L.buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, L.buf, &mr);
        VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
        uint32_t mt = ~0u;
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mr.memoryTypeBits >> i & 1) && (hostm ? (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                    !(mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
                  : (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT))) { mt = i; break; }
        VkMemoryAllocateFlagsInfo mfi{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, nullptr, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, 0 };
        VkMemoryAllocateInfo mai{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, bda ? &mfi : nullptr, mr.size, mt };
        CK(vkAllocateMemory(dev, &mai, nullptr, &L.mem));
        CK(vkBindBufferMemory(dev, L.buf, L.mem, 0));
        // VKQ_BIG_MB: extra device-local allocations (BDA-resident when VKQ_BDA=1) that no job touches, in 128 MB pieces
        for (long mb = envi("VKQ_BIG_MB", 0); mb > 0 && L.d == 0 && L.idx == 0 && L.f == 'c'; mb -= 128) {
            VkMemoryAllocateInfo bi{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, bda ? &mfi : nullptr, (VkDeviceSize) 128 << 20, mt };
            VkDeviceMemory m; CK(vkAllocateMemory(dev, &bi, nullptr, &m));
        }
    }
    VkDescriptorPoolSize ps{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 };
    VkDescriptorPoolCreateInfo dpi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, nullptr, 0, 1, 1, &ps };
    CK(vkCreateDescriptorPool(dev, &dpi, nullptr, &L.dp));
    VkDescriptorSetAllocateInfo dsa{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, nullptr, L.dp, 1, &devs[L.d].dsl };
    CK(vkAllocateDescriptorSets(dev, &dsa, &L.ds));
    VkDescriptorBufferInfo dbi{ L.buf, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, L.ds, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dbi, nullptr };
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    CK(vkBeginCommandBuffer(L.cb, &bi));
    vkCmdResetQueryPool(L.cb, L.qp, 0, 2);
    vkCmdWriteTimestamp(L.cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, L.qp, 0);
    vkCmdBindPipeline(L.cb, VK_PIPELINE_BIND_POINT_COMPUTE, devs[L.d].pipe);
    vkCmdBindDescriptorSets(L.cb, VK_PIPELINE_BIND_POINT_COMPUTE, devs[L.d].pl, 0, 1, &L.ds, 0, nullptr);
    uint32_t pc[2] = { L.iters, off };
    vkCmdPushConstants(L.cb, devs[L.d].pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, pc);
    for (uint32_t k = 0; k < ndisp; ++k) {
        vkCmdDispatch(L.cb, wg, 1, 1);
        VkMemoryBarrier mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT };
        vkCmdPipelineBarrier(L.cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    }
    vkCmdWriteTimestamp(L.cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, L.qp, 1);
    CK(vkEndCommandBuffer(L.cb));
}

struct shot { int64_t sub0, sub1, done, g0, g1; };   // host ns: submit call, submit return, fence seen; GPU start/end mapped to host ns

static void to_host(int di, uint64_t * ts, int64_t * out, int n) {
    VkCalibratedTimestampInfoEXT ci[2] = { { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_DEVICE_EXT },
                                           { VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_EXT, nullptr, VK_TIME_DOMAIN_CLOCK_MONOTONIC_RAW_EXT } };
    uint64_t t[2], dev_ = 0;
    CK(devs[di].calib(devs[di].dev, 2, ci, t, &dev_));
    for (int i = 0; i < n; ++i) out[i] = (int64_t) t[1] + (int64_t) (((double) (int64_t) (ts[i] - t[0])) * period);
}

static void run_lane(lane_t & L, std::atomic<int64_t> * go, int64_t delay, shot & s) {
    int64_t t; while ((t = go->load()) == 0) {}
    while (now_ns() < t + delay) {}
    VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.commandBufferCount = 1; si.pCommandBuffers = &L.cb;
    s.sub0 = now_ns(); CK(vkQueueSubmit(L.q, 1, &si, L.fence)); s.sub1 = now_ns();
    CK(vkWaitForFences(devs[L.d].dev, 1, &L.fence, VK_TRUE, 10ull * 1000000000ull)); s.done = now_ns();
    CK(vkResetFences(devs[L.d].dev, 1, &L.fence));
}

static void finish(lane_t & L, shot & s) {
    uint64_t ts[2]; int64_t h[2];
    CK(vkGetQueryPoolResults(devs[L.d].dev, L.qp, 0, 2, sizeof(ts), ts, 8, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
    to_host(L.d, ts, h, 2); s.g0 = h[0]; s.g1 = h[1];
}

static double med(std::vector<double> v) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char ** argv) {
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        puts("vkq: two-queue concurrency probe; configure with env VKQ_A VKQ_B VKQ_SPV VKQ_PRIO0 VKQ_PRIO1 VKQ_SHARED VKQ_WG\n"
             "     VKQ_ITERS VKQ_ITERS_B VKQ_NDISP VKQ_DELAY_US VKQ_N VKQ_BDA VKQ_HOSTMEM VKQ_BIG_MB (see the source header)");
        return 0;
    }
    if (argc > 1) { fprintf(stderr, "vkq takes no arguments (env only; -h for help)\n"); return 2; }
    const char * spv_path = envs("VKQ_SPV", "spin.spv");
    FILE * f = fopen(spv_path, "rb"); if (!f) { fprintf(stderr, "no %s\n", spv_path); return 2; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); spv.resize(n / 4); if (fread(spv.data(), 1, n, f) != (size_t) n) { fprintf(stderr, "short read %s\n", spv_path); return 2; } fclose(f);
    VkApplicationInfo ai{ VK_STRUCTURE_TYPE_APPLICATION_INFO, nullptr, "vkq", 1, nullptr, 0, VK_API_VERSION_1_3 };
    VkInstanceCreateInfo ici{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, nullptr, 0, &ai };
    CK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t np = 8; VkPhysicalDevice pds[8]; CK(vkEnumeratePhysicalDevices(inst, &np, pds));
    for (uint32_t i = 0; i < np; ++i) { vkGetPhysicalDeviceProperties(pds[i], &props); if (props.vendorID == 0x1002) { pd = pds[i]; break; } }
    if (!pd) { fprintf(stderr, "no AMD device\n"); return 2; }
    VkPhysicalDeviceDriverProperties drv{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES };
    VkPhysicalDeviceProperties2 p2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv }; vkGetPhysicalDeviceProperties2(pd, &p2);
    period = props.limits.timestampPeriod;
    uint32_t nf = 0; vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, nullptr);
    std::vector<VkQueueFamilyProperties> fp(nf); vkGetPhysicalDeviceQueueFamilyProperties(pd, &nf, fp.data());
    for (uint32_t i = 0; i < nf; ++i) {
        if ((fp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && fam_g == ~0u) fam_g = i;
        if ((fp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(fp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && fam_c == ~0u) { fam_c = i; nq_c = fp[i].queueCount; }
    }
    printf("vkq: %s, driver %s %s, compute family %u x%u, gfx family %u, ts period %.2f ns\n", props.deviceName, drv.driverName,
           drv.driverInfo, fam_c, nq_c, fam_g, period);

    lane_t A = parse_lane(envs("VKQ_A", "0:c:0")), B = parse_lane(envs("VKQ_B", "0:c:2"));
    const uint32_t wg = (uint32_t) envi("VKQ_WG", 1), ndisp = (uint32_t) envi("VKQ_NDISP", 1);
    A.iters = (uint32_t) envi("VKQ_ITERS", 200000); B.iters = (uint32_t) envi("VKQ_ITERS_B", A.iters);
    const int64_t delay = envi("VKQ_DELAY_US", 1000) * 1000; const int N = (int) envi("VKQ_N", 20);
    const bool shared = envi("VKQ_SHARED", 0) != 0;
    make_device(0, envs("VKQ_PRIO0", "none"));
    if (A.d == 1 || B.d == 1) make_device(1, envs("VKQ_PRIO1", "none"));
    if (shared && A.d != B.d) { fprintf(stderr, "VKQ_SHARED needs one device\n"); return 2; }
    setup_lane(A, wg, ndisp, nullptr, 0);
    setup_lane(B, wg, ndisp, shared ? &A : nullptr, shared ? wg * 64 : 0);
    printf("config: A=%s B=%s wg=%u ndisp=%u itersA=%u itersB=%u delay_us=%ld shared=%d prio0=%s prio1=%s RADV_PERFTEST=%s RADV_DEBUG=%s\n",
           envs("VKQ_A", "0:c:0"), envs("VKQ_B", "0:c:2"), wg, ndisp, A.iters, B.iters, (long) (delay / 1000), shared ? 1 : 0,
           envs("VKQ_PRIO0", "none"), envs("VKQ_PRIO1", "none"), envs("RADV_PERFTEST", "-"), envs("RADV_DEBUG", "-"));

    // warm up both lanes
    for (int w = 0; w < 3; ++w) for (lane_t * L : { &A, &B }) { std::atomic<int64_t> go{ now_ns() }; shot s; run_lane(*L, &go, 0, s); }

    std::vector<double> dA, dB, span, ovl, gapBA, latB, latBsolo, latA, subB, cAd, cBd;
    for (int it = 0; it < N; ++it) {
        for (int m = 0; m < 3; ++m) {
            const int mode = (m + it) % 3;
            if (mode == 0 || mode == 1) {
                lane_t & L = mode == 0 ? A : B; std::atomic<int64_t> go{ now_ns() }; shot s; run_lane(L, &go, 0, s); finish(L, s);
                (mode == 0 ? dA : dB).push_back((s.g1 - s.g0) / 1e3);
                if (mode == 1) latBsolo.push_back((s.g0 - s.sub1) / 1e3);
            } else {
                std::atomic<int64_t> go{ 0 }; shot sa, sb;
                std::thread ta(run_lane, std::ref(A), &go, (int64_t) 0, std::ref(sa));
                std::thread tb(run_lane, std::ref(B), &go, delay, std::ref(sb));
                go.store(now_ns() + 2000000); ta.join(); tb.join(); finish(A, sa); finish(B, sb);
                const int64_t t0 = std::min(sa.g0, sb.g0);
                span.push_back((std::max(sa.g1, sb.g1) - t0) / 1e3);
                ovl.push_back((std::min(sa.g1, sb.g1) - std::max(sa.g0, sb.g0)) / 1e3);
                gapBA.push_back((sb.g0 - sa.g1) / 1e3);
                latB.push_back((sb.g0 - sb.sub1) / 1e3); latA.push_back((sa.g0 - sa.sub1) / 1e3);
                subB.push_back((sb.sub1 - sb.sub0) / 1e3); cAd.push_back((sa.g1 - sa.g0) / 1e3); cBd.push_back((sb.g1 - sb.g0) / 1e3);
                if (it < 4)
                    printf("conc it%d (us from A GPU start): A gpu [0, %.1f] B submit call %.1f ret %.1f B gpu [%.1f, %.1f] B fence seen %.1f\n", it,
                           (sa.g1 - sa.g0) / 1e3, (sb.sub0 - sa.g0) / 1e3, (sb.sub1 - sa.g0) / 1e3, (sb.g0 - sa.g0) / 1e3, (sb.g1 - sa.g0) / 1e3,
                           (sb.done - sa.g0) / 1e3);
            }
        }
    }
    const double sA = med(dA), sB = med(dB), sp = med(span), saved = sA + sB - sp;
    printf("solo A %.1f us, solo B %.1f us (solo B submit->GPU start %.1f us)\n", sA, sB, med(latBsolo));
    printf("conc: span %.1f, A %.1f, B %.1f, overlap %.1f us, B start - A end %.1f us, B submit->GPU start %.1f us, A submit->GPU start %.1f us, B submit call %.1f us\n",
           sp, med(cAd), med(cBd), med(ovl), med(gapBA), med(latB), med(latA), med(subB));
    printf("RESULT A=%s B=%s prio0=%s prio1=%s shared=%d wg=%u ndisp=%u: saved %.1f us = %.1f%% of min(solo), overlap %.1f us, verdict %s\n",
           envs("VKQ_A", "0:c:0"), envs("VKQ_B", "0:c:2"), envs("VKQ_PRIO0", "none"), envs("VKQ_PRIO1", "none"), shared ? 1 : 0, wg, ndisp,
           saved, 100.0 * saved / std::min(sA, sB), med(ovl), med(ovl) > 0.5 * std::min(sA, sB) ? "CONCURRENT" : (med(ovl) > 0 ? "PARTIAL" : "SERIAL"));
    return 0;
}
