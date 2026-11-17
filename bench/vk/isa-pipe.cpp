// isa-pipe: compile ONE ggml-vulkan flash-attention pipeline (7 storage buffers, 128-byte push range, up to 16 spec
// constants) and exit, so the driver dumps its ISA. [AMD/RADV] With RADV's null winsys this works for GPU families you
// do not have:
//   RADV_FORCE_FAMILY=gfx1201 RADV_DEBUG=asm,nocache isa-pipe flash_attn_f32_f16_cm1.spv 32 <spec...> 2> isa.txt
// (RADV_FORCE_FAMILY selects the null device; no queue, no fences, no allocation is used, the null device has none.)
//
// usage: isa-pipe <shader.spv> <subgroup_size|0> [spec0 spec1 ... spec15]
//   spec constants in the order ggml-vulkan.cpp get_fa_spec_constants() emits them for the variant you want
//   (decimal or 0x hex). Only the first nsc constants given are specialised.
// Any other shader with <= 7 storage buffers and <= 128 bytes of push constants compiles too.
// build: g++ -O2 -o isa-pipe isa-pipe.cpp -lvulkan   (or bench/vk/build.sh)
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

#define CHECK(x) do { VkResult r_ = (x); if (r_ != VK_SUCCESS) { \
    fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)r_); return 2; } } while (0)

static std::vector<uint32_t> read_spv(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint32_t> d(n / 4);
    if (fread(d.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return d;
}

int main(int argc, char **argv) {
    const char *U = "usage: isa-pipe <shader.spv> <subgroup_size|0> [spec0 ... spec15]";
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) { puts(U); return 0; }
    if (argc < 3) { fprintf(stderr, "%s\n", U); return 1; }
    std::vector<uint32_t> spv = read_spv(argv[1]);

    // spec constants in ggml-vulkan get_fa_spec_constants() order
    uint32_t sc[16] = {0}; int nsc = 0;
    for (int i = 0; i + 3 < argc && i < 16; ++i, ++nsc) sc[i] = (uint32_t)strtoul(argv[i + 3], nullptr, 0);
    uint32_t sg = (uint32_t)strtoul(argv[2], nullptr, 0);

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "isa-pipe"; app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    VkInstance inst; CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t n = 0; CHECK(vkEnumeratePhysicalDevices(inst, &n, nullptr));
    if (!n) { fprintf(stderr, "no physical devices\n"); return 2; }
    std::vector<VkPhysicalDevice> pds(n);
    CHECK(vkEnumeratePhysicalDevices(inst, &n, pds.data()));
    VkPhysicalDevice pd = pds[0];
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(pd, &props);
    fprintf(stderr, "device: %s\n", props.deviceName);

    // Feature chain matching what the shader requires.
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    coop.cooperativeMatrix = VK_TRUE;
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    v13.pNext = &coop; v13.subgroupSizeControl = VK_TRUE; v13.computeFullSubgroups = VK_TRUE;
    v13.maintenance4 = VK_TRUE;
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    v12.pNext = &v13; v12.shaderFloat16 = VK_TRUE; v12.shaderInt8 = VK_TRUE;
    v12.storageBuffer8BitAccess = VK_TRUE; v12.bufferDeviceAddress = VK_TRUE;
    v12.vulkanMemoryModel = VK_TRUE; v12.vulkanMemoryModelDeviceScope = VK_TRUE;
    v12.shaderSubgroupExtendedTypes = VK_TRUE;
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    v11.pNext = &v12; v11.storageBuffer16BitAccess = VK_TRUE;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &v11; f2.features.shaderInt16 = VK_TRUE; f2.features.shaderInt64 = VK_TRUE;
    f2.features.robustBufferAccess = VK_FALSE;

    const char *exts[] = { VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME };
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = 0; q.queueCount = 1; q.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2; dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &q;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = exts;
    VkDevice dev; CHECK(vkCreateDevice(pd, &dci, nullptr, &dev));

    VkDescriptorSetLayoutBinding b[7]{};
    for (int i = 0; i < 7; ++i) {
        b[i].binding = i; b[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        b[i].descriptorCount = 1; b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = 7; dslci.pBindings = b;
    VkDescriptorSetLayout dsl; CHECK(vkCreateDescriptorSetLayout(dev, &dslci, nullptr, &dsl));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; CHECK(vkCreatePipelineLayout(dev, &plci, nullptr, &pl));

    VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spv.size() * 4; smci.pCode = spv.data();
    VkShaderModule sm; CHECK(vkCreateShaderModule(dev, &smci, nullptr, &sm));

    VkSpecializationMapEntry me[16];
    for (uint32_t i = 0; i < 16; ++i) { me[i].constantID = i; me[i].offset = i * 4; me[i].size = 4; }
    VkSpecializationInfo si{(uint32_t)nsc, me, nsc * 4u, sc};

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    rss.requiredSubgroupSize = sg;

    VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.pNext = sg ? &rss : nullptr;
    cpci.stage.flags = sg ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = sm; cpci.stage.pName = "main";
    cpci.stage.pSpecializationInfo = &si;
    cpci.layout = pl;
    VkPipeline pipe; CHECK(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe));

    fprintf(stderr, "pipeline created OK \n");
    return 0;
}
