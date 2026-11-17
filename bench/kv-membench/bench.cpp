// kv-membench : cross-platform (any Vulkan driver, Linux or Windows) memory-pipeline
// microbenchmark. Measures useful bandwidth vs. stride (coalescing), useful
// bandwidth vs. occupancy (memory-level parallelism / in-flight ceiling), and
// per-hop DRAM latency via pointer chase. Same SPIR-V runs on every GPU.
//
// The shader is embedded as bench_spv.h (static const uint32_t bench_spv[] plus
// static const uint32_t bench_spv_len in BYTES), generated from bench.comp:
//   glslc -O bench.comp -o bench.spv
//   python3 -c "import struct;d=open('bench.spv','rb').read();w=struct.unpack('<%dI'%(len(d)//4),d);open('bench_spv.h','w').write('#pragma once\n#include <stdint.h>\nstatic const uint32_t bench_spv[]={'+','.join('0x%08x'%x for x in w)+'};\nstatic const uint32_t bench_spv_len=%d;\n'%len(d))"
// Build (Linux):   g++ -O2 bench.cpp -o bench -lvulkan
// Build (Windows): cl /O2 /EHsc bench.cpp /I"%VULKAN_SDK%\Include" \
//                     /link /LIBPATH:"%VULKAN_SDK%\Lib" vulkan-1.lib
//
// Run:  ./bench --list                 list GPUs
//       ./bench                         run all three sweeps on the best GPU
//       ./bench --device 0 --stride-sweep
//       ./bench --occ-sweep --mb 1024
//       ./bench --latency
//       ./bench --run-sweep --interleave 4 [--pad N]   run-length sweep
//       ./bench --stream-sweep --interleave 4          concurrent-stream sweep
//       ./bench --swizzle-sweep | --tile-sweep [--head-elems N] | --gemv-sweep | --gather-sweep
//       ./bench --mlp-sweep | --vol-sweep              (other options: --reps N, --threads N)
//
// --run-sweep    holds useful bytes constant and varies the CONTIGUOUS RUN
//                length, with successive runs `interleave` runs apart. This is
//                the sweep that separates granularity waste (short runs) from
//                power-of-two period aliasing (pays for the whole period even
//                when every line fetched is fully used). --pad adds N elements
//                to the period, which breaks the power of two.
// --stream-sweep runs `interleave` concurrent, individually coalesced streams
//                whose base addresses are a power of two apart, to test whether
//                the aliasing is a base-offset effect.
// --gemv-sweep   emulates a mul_mat_vec weight read: hundreds of concurrent
//                row-reader groups, each streaming whole rows, with the row
//                stride swept across power-of-two, padded, and real-quant
//                geometries. Asks the stream-sweep question at GEMV scale.
//
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

#include "bench_spv.h"

#define VK_CHECK(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "VK error %d at %s:%d\n", (int)_r, __FILE__, __LINE__); \
    exit(1);} } while(0)

struct PushConstants {
    uint32_t nthreads;
    uint32_t stride;
    uint32_t reads;
    uint32_t mode;   // 0 = stream, 1 = chase, 2 = mlp, 3 = run
    uint32_t run;    // mode 3 only: contiguous run length in elements (power of two)
    uint32_t logrun;
    uint32_t runstride;
    uint32_t nblocks;
    uint32_t blocklog;
    uint32_t slotmask;
};

struct Ctx {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties props{};
    VkPhysicalDeviceMemoryProperties memProps{};
    uint32_t queueFamily = 0;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    double timestampPeriod = 1.0; // ns per tick

    VkBuffer dataBuf = VK_NULL_HANDLE, sinkBuf = VK_NULL_HANDLE;
    VkDeviceMemory dataMem = VK_NULL_HANDLE, sinkMem = VK_NULL_HANDLE;

    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkDescriptorSet descSet = VK_NULL_HANDLE;
    VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkShaderModule shader = VK_NULL_HANDLE;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
};

static const char* devTypeName(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "discrete";
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "integrated";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "virtual";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "cpu";
        default:                                     return "other";
    }
}

static uint32_t findMemType(Ctx& c, uint32_t typeBits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.memProps.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (c.memProps.memoryTypes[i].propertyFlags & want) == want)
            return i;
    }
    fprintf(stderr, "no memory type with flags 0x%x\n", want);
    exit(1);
}

static void makeBuffer(Ctx& c, VkDeviceSize size, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props, VkBuffer& buf, VkDeviceMemory& mem) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size; bi.usage = usage; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(c.device, &bi, nullptr, &buf));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(c.device, buf, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemType(c, req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(c.device, &ai, nullptr, &mem));
    VK_CHECK(vkBindBufferMemory(c.device, buf, mem, 0));
}

// One-time staging upload of `bytes` from `src` into a DEVICE_LOCAL buffer.
static void uploadToDevice(Ctx& c, VkBuffer dst, const void* src, VkDeviceSize bytes) {
    VkBuffer staging; VkDeviceMemory stagingMem;
    makeBuffer(c, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
               staging, stagingMem);
    void* p = nullptr;
    VK_CHECK(vkMapMemory(c.device, stagingMem, 0, bytes, 0, &p));
    memcpy(p, src, bytes);
    vkUnmapMemory(c.device, stagingMem);

    VkCommandBuffer cb;
    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = c.cmdPool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &cbi, &cb));
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &begin));
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cb, staging, dst, 1, &region);
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(c.queue));
    vkFreeCommandBuffers(c.device, c.cmdPool, 1, &cb);
    vkDestroyBuffer(c.device, staging, nullptr);
    vkFreeMemory(c.device, stagingMem, nullptr);
}

// Build a compute pipeline specialized with a given MLP (independent loads per
// thread). MLP is constant_id 0 in the shader and only affects mode 2, so the
// default pipeline is simply MLP=1.
static VkPipeline makePipeline(Ctx& c, uint32_t mlp) {
    VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};
    VkSpecializationInfo spec{};
    spec.mapEntryCount = 1; spec.pMapEntries = &entry;
    spec.dataSize = sizeof(uint32_t); spec.pData = &mlp;

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = c.shader; cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = &spec;
    cpci.layout = c.pipeLayout;
    VkPipeline p = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(c.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &p));
    return p;
}

// Run one dispatch, return GPU time in nanoseconds (min over reps).
// `pipe` overrides the default pipeline; the mlp-sweep uses it to select a
// build specialized for a particular loads-per-thread count.
static double runDispatch(Ctx& c, const PushConstants& pc, uint32_t groups, int reps,
                          VkPipeline pipe = VK_NULL_HANDLE) {
    VkPipeline usePipe = pipe ? pipe : c.pipe;
    double best = 1e300;
    for (int rep = 0; rep < reps; ++rep) {
        VK_CHECK(vkResetCommandBuffer(c.cmd, 0));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cmd, &begin));
        vkCmdResetQueryPool(c.cmd, c.queryPool, 0, 2);
        vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, usePipe);
        vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, c.pipeLayout,
                                0, 1, &c.descSet, 0, nullptr);
        vkCmdPushConstants(c.cmd, c.pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(PushConstants), &pc);
        vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, c.queryPool, 0);
        vkCmdDispatch(c.cmd, groups, 1, 1);
        vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, c.queryPool, 1);
        VK_CHECK(vkEndCommandBuffer(c.cmd));

        VK_CHECK(vkResetFences(c.device, 1, &c.fence));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &c.cmd;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, c.fence));
        VK_CHECK(vkWaitForFences(c.device, 1, &c.fence, VK_TRUE, UINT64_MAX));

        uint64_t ts[2] = {0, 0};
        VK_CHECK(vkGetQueryPoolResults(c.device, c.queryPool, 0, 2, sizeof(ts), ts,
                                       sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double ns = (double)(ts[1] - ts[0]) * c.timestampPeriod;
        if (ns > 0 && ns < best) best = ns;
    }
    return best;
}

// Sattolo's algorithm: one Hamiltonian cycle over [0,n) written straight into
// the mapped host buffer. Deterministic xorshift so runs are reproducible and
// there is no Date/rand dependency.
static void buildCycle(uint32_t* a, uint64_t n) {
    for (uint64_t i = 0; i < n; ++i) a[i] = (uint32_t)i;
    uint64_t x = 0x9E3779B97F4A7C15ull; // fixed seed
    for (uint64_t i = n - 1; i > 0; --i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;      // xorshift64
        uint64_t j = x % i;                            // 0 <= j < i  (Sattolo)
        uint32_t tmp = a[i]; a[i] = a[j]; a[j] = tmp;
    }
}

int main(int argc, char** argv) {
    // ---- args ----
    bool doList = false, doStride = false, doOcc = false, doLat = false, doVol = false, doMlp = false, doRun = false, doStreams = false, doSwizzle = false, doTile = false, doGemv = false, doGather = false;
    uint32_t interleave = 4;
    uint32_t headElems = 64;   // elements per head per token (hd128 f16 = 256B = 64 x u32)                            // mode 3: ways the runs are interleaved
    uint32_t pad = 0;                                   // mode 3: extra elements per period
    int devIndex = -1, reps = 3;
    uint64_t mb = 1024;
    uint64_t threads = 0;                                // 0 = default 1<<20
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--list") doList = true;
        else if (a == "--stride-sweep") doStride = true;
        else if (a == "--run-sweep") doRun = true;
        else if (a == "--swizzle-sweep") doSwizzle = true;
        else if (a == "--tile-sweep") doTile = true;
        else if (a == "--head-elems" && i + 1 < argc) headElems = (uint32_t)strtoul(argv[++i], nullptr, 0);
        else if (a == "--stream-sweep") doStreams = true;
        else if (a == "--gemv-sweep") doGemv = true;
        else if (a == "--gather-sweep") doGather = true;
        else if (a == "--interleave" && i + 1 < argc) interleave = (uint32_t)atoi(argv[++i]);
        else if (a == "--pad" && i + 1 < argc) pad = (uint32_t)atoi(argv[++i]);
        else if (a == "--occ-sweep") doOcc = true;
        else if (a == "--latency") doLat = true;
        else if (a == "--vol-sweep") doVol = true;
        else if (a == "--mlp-sweep") doMlp = true;
        else if (a == "--device" && i + 1 < argc) devIndex = atoi(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = atoi(argv[++i]);
        else if (a == "--mb" && i + 1 < argc) mb = strtoull(argv[++i], nullptr, 10);
        else if (a == "--threads" && i + 1 < argc) threads = strtoull(argv[++i], nullptr, 10);
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 1; }
    }
    if (!doStride && !doOcc && !doLat && !doList && !doRun && !doStreams && !doSwizzle && !doTile && !doGemv && !doGather && !doMlp && !doVol) doStride = doOcc = doLat = true;

    // ---- instance ----
    Ctx c;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "kv-membench"; app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VK_CHECK(vkCreateInstance(&ici, nullptr, &c.instance));

    uint32_t nDev = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(c.instance, &nDev, nullptr));
    std::vector<VkPhysicalDevice> devs(nDev);
    VK_CHECK(vkEnumeratePhysicalDevices(c.instance, &nDev, devs.data()));
    if (nDev == 0) { fprintf(stderr, "no Vulkan devices\n"); return 1; }

    if (doList || devIndex < 0) {
        int best = -1, bestScore = -1;
        for (uint32_t i = 0; i < nDev; ++i) {
            VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(devs[i], &p);
            VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(devs[i], &mp);
            uint64_t vram = 0;
            for (uint32_t h = 0; h < mp.memoryHeapCount; ++h)
                if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
                    vram = std::max<uint64_t>(vram, mp.memoryHeaps[h].size);
            if (doList)
                fprintf(stderr, "  [%u] %-40s %-10s %llu MB\n", i, p.deviceName,
                        devTypeName(p.deviceType), (unsigned long long)(vram >> 20));
            int score = (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   ? 3 :
                        (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 2 :
                        (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)            ? 0 : 1;
            if (score > bestScore) { bestScore = score; best = (int)i; }
        }
        if (doList) return 0;
        if (devIndex < 0) devIndex = best;
    }
    if (devIndex < 0 || devIndex >= (int)nDev) { fprintf(stderr, "bad device index\n"); return 1; }
    c.phys = devs[devIndex];
    vkGetPhysicalDeviceProperties(c.phys, &c.props);
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.memProps);
    c.timestampPeriod = c.props.limits.timestampPeriod;

    // ---- queue + device ----
    uint32_t nQF = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nQF, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nQF);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nQF, qf.data());
    c.queueFamily = UINT32_MAX;
    for (uint32_t i = 0; i < nQF; ++i)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits > 0) {
            c.queueFamily = i; break;
        }
    if (c.queueFamily == UINT32_MAX) { fprintf(stderr, "no compute queue with timestamps\n"); return 1; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = c.queueFamily; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    VK_CHECK(vkCreateDevice(c.phys, &dci, nullptr, &c.device));
    vkGetDeviceQueue(c.device, c.queueFamily, 0, &c.queue);

    fprintf(stderr, "device : %s (%s)\n", c.props.deviceName, devTypeName(c.props.deviceType));
    fprintf(stderr, "driver : ts_period=%.2f ns  maxStorageBufferRange=%.0f MB\n",
            c.timestampPeriod, c.props.limits.maxStorageBufferRange / 1048576.0);

    // ---- sizing ----
    uint64_t n = (mb * 1024 * 1024) / 4;              // uint elements in data[]
    uint64_t dataBytes = n * 4;
    if (dataBytes > c.props.limits.maxStorageBufferRange) {
        fprintf(stderr, "warning: %llu MB exceeds maxStorageBufferRange, clamping\n",
                (unsigned long long)mb);
        dataBytes = (c.props.limits.maxStorageBufferRange / 4096) * 4096;
        n = dataBytes / 4;
    }
    // sink must be index-safe up to the largest thread count we dispatch.
    uint64_t maxThreads = 1u << 23; // occ-sweep ceiling
    uint64_t sinkBytes = maxThreads * 4;

    // ---- resources ----
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.queueFamilyIndex = c.queueFamily;
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(c.device, &cpi, nullptr, &c.cmdPool));

    makeBuffer(c, dataBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, c.dataBuf, c.dataMem);
    makeBuffer(c, sinkBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
               VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, c.sinkBuf, c.sinkMem);

    // Fill data[] with a single-cycle permutation: valid for stream reads and
    // required for the pointer chase. Built directly in host staging memory.
    fprintf(stderr, "building %llu MB permutation (one-time)...\n", (unsigned long long)(dataBytes >> 20));
    {
        VkBuffer staging; VkDeviceMemory stagingMem;
        makeBuffer(c, dataBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                   staging, stagingMem);
        void* p = nullptr;
        VK_CHECK(vkMapMemory(c.device, stagingMem, 0, dataBytes, 0, &p));
        // Build in cached heap RAM (fast random access for Sattolo's swaps),
        // then stream sequentially into the mapped staging buffer. Mapped
        // host-visible memory is often write-combined, where the random swaps
        // would fall off a performance cliff.
        std::vector<uint32_t> host(n);
        buildCycle(host.data(), n);
        memcpy(p, host.data(), dataBytes);
        vkUnmapMemory(c.device, stagingMem);
        // reuse uploadToDevice's copy path via a direct copy here
        VkCommandBuffer cb;
        VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbi.commandPool = c.cmdPool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(c.device, &cbi, &cb));
        VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &begin));
        VkBufferCopy region{0, 0, dataBytes};
        vkCmdCopyBuffer(cb, staging, c.dataBuf, 1, &region);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(c.queue));
        vkFreeCommandBuffers(c.device, c.cmdPool, 1, &cb);
        vkDestroyBuffer(c.device, staging, nullptr);
        vkFreeMemory(c.device, stagingMem, nullptr);
    }

    // ---- descriptors ----
    VkDescriptorSetLayoutBinding binds[2]{};
    for (int i = 0; i < 2; ++i) {
        binds[i].binding = i; binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1; binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 2; dslci.pBindings = binds;
    VK_CHECK(vkCreateDescriptorSetLayout(c.device, &dslci, nullptr, &c.dsl));

    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.maxSets = 1; dpci.poolSizeCount = 1; dpci.pPoolSizes = &psz;
    VK_CHECK(vkCreateDescriptorPool(c.device, &dpci, nullptr, &c.descPool));

    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = c.descPool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &c.dsl;
    VK_CHECK(vkAllocateDescriptorSets(c.device, &dsai, &c.descSet));

    VkDescriptorBufferInfo dbi[2] = {{c.dataBuf, 0, VK_WHOLE_SIZE}, {c.sinkBuf, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[2]{};
    for (int i = 0; i < 2; ++i) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; w[i].dstSet = c.descSet;
        w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(c.device, 2, w, 0, nullptr);

    // ---- pipeline ----
    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = bench_spv_len; smci.pCode = bench_spv;
    VK_CHECK(vkCreateShaderModule(c.device, &smci, nullptr, &c.shader));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &c.dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(c.device, &plci, nullptr, &c.pipeLayout));

    c.pipe = makePipeline(c, 1);

    VkCommandBufferAllocateInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbi.commandPool = c.cmdPool; cbi.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbi.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(c.device, &cbi, &c.cmd));

    VkQueryPoolCreateInfo qpci{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpci.queryType = VK_QUERY_TYPE_TIMESTAMP; qpci.queryCount = 2;
    VK_CHECK(vkCreateQueryPool(c.device, &qpci, nullptr, &c.queryPool));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(c.device, &fci, nullptr, &c.fence));

    const uint32_t LOCAL = 256;
    auto groupsFor = [&](uint64_t threads){ return (uint32_t)((threads + LOCAL - 1) / LOCAL); };

    // warmup
    { PushConstants pc{1u<<16, 4, 16, 0}; runDispatch(c, pc, groupsFor(pc.nthreads), 1); }

    // ===== stride sweep: fixed threads, vary coalescing =====
    if (doStride) {
        uint64_t T = threads ? threads : (1u << 20);
        printf("# stride-sweep  mode=stream  threads=%llu  footprint<=%lluMB\n",
               (unsigned long long)T, (unsigned long long)(dataBytes >> 20));
        printf("stride_elems,stride_bytes,reads_per_thread,useful_MB,ms,GB_s\n");
        for (uint32_t stride : {1u,2u,4u,8u,16u,32u,64u,128u,256u}) {
            uint64_t R = n / (stride * T);
            if (R < 1) continue;
            PushConstants pc{(uint32_t)T, stride, (uint32_t)R, 0};
            double ns = runDispatch(c, pc, groupsFor(T), reps);
            double bytes = (double)T * R * 4.0;
            double gbs = bytes / ns;                    // bytes/ns == GB/s
            printf("%u,%u,%llu,%.1f,%.4f,%.2f\n", stride, stride*4,
                   (unsigned long long)R, bytes/1048576.0, ns*1e-6, gbs);
            fflush(stdout);
        }
        printf("\n");
    }

    // ===== run sweep: fixed interleave, vary contiguous run length =====
    // Interpolates between the stride tax (run=1) and fully coalesced reads
    // (run >= footprint). The knee is the shortest contiguous run that still
    // buys full bandwidth, which is what decides whether a block-paged KV cache
    // needs long head-contiguous runs or whether one block's worth is enough.
    if (doRun) {
        uint64_t T = threads ? threads : (1u << 20);
        printf("# run-sweep  mode=run  interleave=%u (%uB apart at run boundaries)  threads=%llu  span<=%lluMB\n",
               interleave, interleave*4, (unsigned long long)T,
               (unsigned long long)(dataBytes >> 20));
        printf("run_elems,run_bytes,reads_per_thread,useful_MB,ms,GB_s\n");
        for (uint32_t run : {1u,2u,4u,8u,16u,32u,64u,128u,256u,512u,1024u,
                             4096u,16384u,65536u,262144u,1048576u}) {
            // Keep the touched span inside the allocation: with a pad the period
            // is no longer run*interleave, so the readable prefix shrinks.
            const uint64_t period = (uint64_t)run * interleave + pad;
            uint64_t R = (uint64_t)((double)n * run / (double)period) / T;
            if (R < 1) continue;
            uint32_t logrun = 0; while ((1u << logrun) < run) ++logrun;
            PushConstants pc{(uint32_t)T, interleave, (uint32_t)R, 3u,
                             run, logrun, run * interleave + pad};
            double ns = runDispatch(c, pc, groupsFor(T), reps);
            double bytes = (double)T * R * 4.0;
            printf("%u,%u,%llu,%.1f,%.4f,%.2f\n", run, run*4,
                   (unsigned long long)R, bytes/1048576.0, ns*1e-6, bytes/ns);
            fflush(stdout);
        }
        printf("\n");
    }

    // ===== swizzle sweep: run-sweep geometry, but the sub-slot rotates =====
    // Same bytes, same runs, same allocation as --run-sweep with pad 0. The only
    // difference is WHERE inside each period the run is read from. If the stride
    // tax is power-of-two period aliasing, rotating the slot should recover the
    // bandwidth that a pad recovers -- at zero memory cost.
    if (doSwizzle) {
        uint64_t T = threads ? threads : (1u << 20);
        printf("# swizzle-sweep  mode=swizzle  interleave=%u (%uB apart)  threads=%llu  span<=%lluMB\n",
               interleave, interleave*4, (unsigned long long)T,
               (unsigned long long)(dataBytes >> 20));
        printf("run_elems,run_bytes,reads_per_thread,useful_MB,ms,GB_s\n");
        for (uint32_t run : {1u,2u,4u,8u,16u,32u,64u,128u,256u,512u,1024u,
                             4096u,16384u,65536u,262144u,1048576u}) {
            const uint64_t period = (uint64_t)run * interleave;
            uint64_t R = (uint64_t)((double)n * run / (double)period) / T;
            if (R < 1) continue;
            uint32_t logrun = 0; while ((1u << logrun) < run) ++logrun;
            PushConstants pc{(uint32_t)T, interleave, (uint32_t)R, 5u,
                             run, logrun, run * interleave};
            double ns = runDispatch(c, pc, groupsFor(T), reps);
            double bytes = (double)T * R * 4.0;
            printf("%u,%u,%llu,%.1f,%.4f,%.2f\n", run, run*4,
                   (unsigned long long)R, bytes/1048576.0, ns*1e-6, bytes/ns);
            fflush(stdout);
        }
        printf("\n");
    }

    // ===== tile sweep: tiled head-major, sweeping the tile size in TOKENS =====
    // Group G tokens into a tile and store head-major inside it. Reading one head
    // across tokens then gives a contiguous run of G*headElems, with the next tile
    // one period (run*interleave) away -- algebraically the same access pattern as
    // the run sweep, but parameterised the way a cache would actually be built, so
    // the answer comes out as "how many tokens per tile" at a real head geometry.
    // G=1 is token-major, G=kv_size is full head-major.
    if (doTile) {
        uint64_t T = threads ? threads : (1u << 20);
        printf("# tile-sweep  head_elems=%u (%uB/head/token)  heads=%u  threads=%llu  span<=%lluMB\n",
               headElems, headElems*4, interleave, (unsigned long long)T,
               (unsigned long long)(dataBytes >> 20));
        printf("tile_tokens,run_bytes,tile_bytes,reads_per_thread,useful_MB,ms,GB_s\n");
        for (uint32_t G : {1u,2u,4u,8u,16u,32u,64u,128u,256u}) {
            const uint64_t run = (uint64_t)G * headElems;
            if (run > (1u<<24)) break;
            const uint64_t period = run * interleave;
            uint64_t R = (uint64_t)((double)n * run / (double)period) / T;
            if (R < 1) continue;
            uint32_t logrun = 0; while ((1u << logrun) < run) ++logrun;
            if ((1u << logrun) != run) continue;   // shader needs a power-of-two run
            PushConstants pc{(uint32_t)T, interleave, (uint32_t)R, 3u,
                             (uint32_t)run, logrun, (uint32_t)period};
            double ns = runDispatch(c, pc, groupsFor(T), reps);
            double bytes = (double)T * R * 4.0;
            printf("%u,%llu,%llu,%llu,%.1f,%.4f,%.2f\n", G,
                   (unsigned long long)run*4, (unsigned long long)period*4,
                   (unsigned long long)R, bytes/1048576.0, ns*1e-6, bytes/ns);
            fflush(stdout);
        }
        printf("\n");
    }

    // ===== stream sweep: concurrent coalesced streams at a swept base offset =====
    // Models per-head KV reads after a contiguize/head-major layout: each head's
    // data is one long sequential run, but the heads sit a fixed stride apart.
    // Compares a power-of-two region stride against the same stride plus a pad.
    if (doStreams) {
        const uint64_t T   = 1u << 20;
        const uint32_t G   = interleave;                 // streams (KV heads)
        const uint32_t per = (uint32_t)(T / G);          // threads per stream
        uint32_t logper = 0; while ((1u << logper) < per) ++logper;
        // The base stride must stay a true power of two in the pad=0 arm -- that
        // is the whole point of the control -- so take the slack out of how much
        // each stream READS, never out of the stride itself.
        const uint64_t region = (n / G) - 8192;          // elements read per stream
        const uint64_t R      = region / per;
        printf("# stream-sweep  mode=streams  streams=%u  threads/stream=%u  region=%lluMB  reads/thread=%llu\n",
               G, per, (unsigned long long)(region*4 >> 20), (unsigned long long)R);
        printf("base_stride_elems,base_stride_MB,pad_bytes,useful_MB,ms,GB_s\n");
        for (uint32_t padb : {0u, 128u, 256u, 4096u}) {
            const uint64_t bstride = (n / G) + padb/4;   // pad=0 => exactly n/G
            if (bstride * (G-1) + region > n) { printf("# skip pad %u (overruns)\n", padb); continue; }
            PushConstants pc{(uint32_t)T, G, (uint32_t)R, 4u, per, logper, (uint32_t)bstride};
            double ns = runDispatch(c, pc, groupsFor(T), reps);
            double bytes = (double)T * R * 4.0;
            printf("%llu,%.2f,%u,%.1f,%.4f,%.2f\n", (unsigned long long)bstride,
                   bstride*4.0/1048576.0, padb, bytes/1048576.0, ns*1e-6, bytes/ns);
            fflush(stdout);
        }
        printf("\n");
    }

    // ===== gemv sweep: concurrent row readers at a swept row stride =====
    // Models a mul_mat_vec weight read: every byte of the matrix is read exactly
    // once per token, so there is no gap tax to measure -- what is measured is
    // whether hundreds of near-lockstep coalesced row streams whose bases are a
    // power of two apart serialise on the memory system. The stream sweep asks
    // this at a few long streams; a GEMV dispatch is the same question at
    // hundreds of short streams. Each stride is run at a pure power of two,
    // plus small pads (the padded-cache geometry), plus the strides real quant
    // formats produce (odd block factors: q4_0 18B, q8_0 34B, q6_K 210B blocks).
    // ===== gather sweep: selected-key KV gather (sparse attention reading whole selected blocks of keys) =====
    if (doGather) {
        struct Geo { uint32_t strideB, runB, blockRows, slots; const char* tag; };
        static const Geo geos[] = {
            {1024, 512, 4, 1, "f16_kv_1h_blk4"},  {1088, 512, 4, 1, "f16_kv+64_1h_blk4"},  {1280, 512, 4, 1, "f16_kv+256_1h_blk4"},
            {1024, 512, 4, 2, "f16_kv_2h_blk4"},  {1088, 512, 4, 2, "f16_kv+64_2h_blk4"},  {1280, 512, 4, 2, "f16_kv+256_2h_blk4"},
            {1024, 512, 1, 1, "f16_kv_1h_blk1"},  {1088, 512, 1, 1, "f16_kv+64_1h_blk1"},  {1280, 512, 1, 1, "f16_kv+256_1h_blk1"},
            {544,  272, 4, 1, "q8_0_kv_1h_blk4"}, {544,  272, 4, 2, "q8_0_kv_2h_blk4"},    {544,  272, 1, 1, "q8_0_kv_1h_blk1"},
            {1024, 1024, 4, 1, "f16_fullrow_blk4"},
            {4096, 512, 1, 1, "ctl_pos_4096_512"}, {4352, 512, 1, 1, "ctl_pos_4096+256_512"},
            {10240, 64, 1, 1, "ctl_pos_10240_64"}, {10496, 64, 1, 1, "ctl_pos_10496_64"},
        };
        uint64_t T = threads ? threads : (1u << 20);
        const double targetB = 256.0 * 1048576.0;
        printf("# gather-sweep  mode=gather  threads=%llu  span<=%lluMB  useful~256MB/dispatch\n",
               (unsigned long long)T, (unsigned long long)(dataBytes >> 20));
        printf("tag,row_stride_bytes,run_bytes,block_rows,head_slots,lanes,groups,rows_per_group,useful_MB,ms,GB_s\n");
        for (const Geo& geo : geos) {
            const uint32_t rsE = geo.strideB / 4, runE = geo.runB / 4;
            const uint32_t per = runE >= 64 ? 64u : 16u;
            uint32_t logper = 0; while ((1u << logper) < per) ++logper;
            uint32_t blog = 0; while ((1u << blog) < geo.blockRows) ++blog;
            const uint64_t nrows = n / rsE;
            const uint32_t nblocks = (uint32_t)(nrows >> blog);
            const uint64_t G = T >> logper;
            uint64_t rpg = (uint64_t)(targetB / geo.runB) / G; if (rpg < 1) rpg = 1;
            PushConstants pc{(uint32_t)T, runE, (uint32_t)rpg, 7u, per, logper, rsE, nblocks, blog, geo.slots - 1u};
            double ns = runDispatch(c, pc, groupsFor(T), reps);
            double bytes = (double)G * rpg * geo.runB;
            printf("%s,%u,%u,%u,%u,%u,%llu,%llu,%.1f,%.4f,%.2f\n", geo.tag, geo.strideB, geo.runB, geo.blockRows,
                   geo.slots, per, (unsigned long long)G, (unsigned long long)rpg, bytes/1048576.0, ns*1e-6, bytes/ns);
            fflush(stdout);
        }
        printf("\n");
    }

    if (doGemv) {
        struct Geo { uint32_t strideB, usefulB; const char* tag; };
        static const Geo geos[] = {
            {2048,  2048,  "pow2"},   {2064,  2048,  "pad16"},  {2176,  2048,  "pad128"},
            {4096,  4096,  "pow2"},   {4112,  4096,  "pad16"},  {4160,  4096,  "pad64"},  {4224, 4096, "pad128"},
            {8192,  8192,  "pow2"},   {8208,  8192,  "pad16"},  {8256,  8192,  "pad64"},  {8320, 8192, "pad128"},
            {16384, 16384, "pow2"},   {16400, 16384, "pad16"},
            {12288, 12288, "pow2x3"},
            {2304,  2304,  "q4_0"},   {4352,  4352,  "q8_0"},   {3360,  3360,  "q6_K"},
        };
        uint64_t T = threads ? threads : (1u << 20);
        printf("# gemv-sweep  mode=gemv  threads=%llu  footprint<=%lluMB\n",
               (unsigned long long)T, (unsigned long long)(dataBytes >> 20));
        printf("tag,row_stride_bytes,row_bytes,lanes_per_row,groups,rows_per_group,useful_MB,ms,GB_s\n");
        for (uint32_t per : {64u, 256u}) {
            uint32_t logper = 0; while ((1u << logper) < per) ++logper;
            for (const Geo& g : geos) {
                const uint32_t rsElems = g.strideB / 4, reElems = g.usefulB / 4;
                const uint64_t nrows = n / rsElems;
                const uint64_t G     = T >> logper;
                const uint64_t rpg   = nrows / G;
                if (rpg < 1) { printf("# skip %s %u (too few rows)\n", g.tag, g.strideB); continue; }
                PushConstants pc{(uint32_t)T, reElems, (uint32_t)rpg, 6u, per, logper, rsElems};
                double ns = runDispatch(c, pc, groupsFor(T), reps);
                double bytes = (double)G * rpg * g.usefulB;
                printf("%s,%u,%u,%u,%llu,%llu,%.1f,%.4f,%.2f\n", g.tag, g.strideB, g.usefulB,
                       per, (unsigned long long)G, (unsigned long long)rpg,
                       bytes/1048576.0, ns*1e-6, bytes/ns);
                fflush(stdout);
            }
            printf("\n");
        }
    }

    // ===== occupancy sweep: fixed large stride, vary threads in flight =====
    if (doOcc) {
        uint32_t stride = 32;                           // 128 B apart -> one line each
        printf("# occ-sweep  mode=stream  stride=%u (%uB, de-coalesced)\n", stride, stride*4);
        printf("threads,reads_per_thread,useful_MB,ms,GB_s\n");
        for (uint64_t T = 256; T <= (uint64_t)(n/stride) && T <= maxThreads; T <<= 1) {
            uint64_t R = n / (stride * T);
            if (R < 1) break;
            PushConstants pc{(uint32_t)T, stride, (uint32_t)R, 0};
            double ns = runDispatch(c, pc, groupsFor(T), reps);
            double bytes = (double)T * R * 4.0;
            double gbs = bytes / ns;
            printf("%llu,%llu,%.1f,%.4f,%.2f\n", (unsigned long long)T,
                   (unsigned long long)R, bytes/1048576.0, ns*1e-6, gbs);
            fflush(stdout);
        }
        printf("\n");
    }

    // ===== loads-per-thread sweep: fixed threads, vary loads in flight =====
    // The occupancy sweep varies thread count, which conflates "more waves" with
    // "more outstanding requests". This one holds threads fixed and gives each
    // thread MLP independent loads instead, then repeats at several thread
    // counts. Reading the result:
    //   saturates at the same MLP for every thread count -> per-wavefront limit
    //     (an outstanding-load counter such as RDNA's vmcnt)
    //   only threads*MLP matters -> the ceiling is downstream and shared:
    //     cache miss buffers, fabric credits, or the memory controller queue
    // Caveat: raising MLP costs registers, which can cut how many waves stay
    // resident. Part of any flattening at high MLP may be lost occupancy rather
    // than a request ceiling, so read the knee, not the tail.
    if (doMlp) {
        uint32_t stride = 32;                      // 128 B apart, de-coalesced
        printf("# mlp-sweep  mode=stream  stride=%u (%uB, de-coalesced)\n", stride, stride*4);
        printf("threads,loads_per_thread,requests_in_flight,reads_per_thread,useful_MB,ms,GB_s\n");
        for (uint64_t T : {1024ull, 8192ull, 65536ull}) {
            for (uint32_t K : {1u,2u,4u,8u,16u,32u}) {
                uint64_t R = n / ((uint64_t)stride * T * K);
                if (R < 1) continue;
                VkPipeline p = makePipeline(c, K);
                PushConstants pc{(uint32_t)T, stride, (uint32_t)R, 2};
                double ns = runDispatch(c, pc, groupsFor(T), reps, p);
                vkDestroyPipeline(c.device, p, nullptr);
                double bytes = (double)T * K * R * 4.0;
                double gbs = bytes / ns;
                printf("%llu,%u,%llu,%llu,%.1f,%.4f,%.2f\n",
                       (unsigned long long)T, K, (unsigned long long)(T * K),
                       (unsigned long long)R, bytes/1048576.0, ns*1e-6, gbs);
                fflush(stdout);
            }
            printf("\n");
        }
    }

    // ===== volume sweep: fixed stride + occupancy, grow the working set =====
    // Tests the "more volume forces a collapse" hypothesis: threads and stride
    // are held on the de-coalesced plateau; only the touched footprint grows.
    // A flat curve => volume-independent (concurrency-bound). A sag at the top
    // => TLB/page-reach limited, i.e. volume can drag even a big GPU down.
    if (doVol) {
        uint32_t stride = 32;                 // 128 B apart, the worst-case KV stride
        uint64_t T = 1u << 13;                // 8192 threads: firmly on the plateau
        printf("# vol-sweep  mode=stream  stride=%u (%uB)  threads=%llu  (resident lines ~= span/2)\n",
               stride, stride*4, (unsigned long long)T);
        printf("span_MB,reads_per_thread,ms,GB_s\n");
        for (uint64_t span = 1ull<<20; span <= dataBytes; span <<= 1) {
            uint64_t R = span / (stride * T * 4);
            if (R < 1) continue;
            PushConstants pc{(uint32_t)T, stride, (uint32_t)R, 0};
            double ns = runDispatch(c, pc, groupsFor(T), reps);
            double bytes = (double)T * R * 4.0;
            double gbs = bytes / ns;
            printf("%llu,%llu,%.4f,%.2f\n", (unsigned long long)(span>>20),
                   (unsigned long long)R, ns*1e-6, gbs);
            fflush(stdout);
        }
        printf("\n");
    }

    // ===== latency: single-thread dependent pointer chase =====
    if (doLat) {
        uint32_t hops = 4u << 20;                       // 4,194,304 hops
        printf("# latency  mode=chase  single dependent chain over %lluMB\n",
               (unsigned long long)(dataBytes >> 20));
        printf("threads,hops,ms,ns_per_hop\n");
        for (uint32_t T : {1u, 32u}) {                  // 1 = pure latency, 32 = one full subgroup
            PushConstants pc{T, 1, hops, 1};
            double ns = runDispatch(c, pc, groupsFor(T), reps);
            double perHop = ns / ((double)hops);        // chains run in parallel, same hop count
            printf("%u,%u,%.4f,%.2f\n", T, hops, ns*1e-6, perHop);
            fflush(stdout);
        }
        printf("\n");
    }

    // ---- teardown (process exit would suffice, but be tidy) ----
    vkDestroyFence(c.device, c.fence, nullptr);
    vkDestroyQueryPool(c.device, c.queryPool, nullptr);
    vkDestroyPipeline(c.device, c.pipe, nullptr);
    vkDestroyPipelineLayout(c.device, c.pipeLayout, nullptr);
    vkDestroyShaderModule(c.device, c.shader, nullptr);
    vkDestroyDescriptorPool(c.device, c.descPool, nullptr);
    vkDestroyDescriptorSetLayout(c.device, c.dsl, nullptr);
    vkDestroyBuffer(c.device, c.dataBuf, nullptr); vkFreeMemory(c.device, c.dataMem, nullptr);
    vkDestroyBuffer(c.device, c.sinkBuf, nullptr); vkFreeMemory(c.device, c.sinkMem, nullptr);
    vkDestroyCommandPool(c.device, c.cmdPool, nullptr);
    vkDestroyDevice(c.device, nullptr);
    vkDestroyInstance(c.instance, nullptr);
    return 0;
}
