// vkasm: create one compute pipeline from a SPIR-V file with given spec constants, so the driver's shader dump shows the
// ISA of exactly that variant, and print VK_KHR_pipeline_executable_properties statistics (VGPRs, SGPRs, LDS, code size,
// waves per SIMD) on stderr as "STAT <name> = <value>". No dispatch, no buffers.
//
// usage: vkasm <spv> <n_storage_buffers> <push_const_bytes> <subgroup_size|0> <spec0,spec1,...>
//   RADV_DEBUG=asm,nocache vkasm ... 2> isa.txt      # [AMD/RADV] ACO ISA of the pipeline; then isa-histo.py isa.txt
// nocache matters: a pipeline-cache hit prints no ISA.
// build: g++ -O2 -o vkasm vkasm.cpp -lvulkan   (or bench/vk/build.sh)
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
static std::vector<char> rd(const char*p){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<char>b(n);if(fread(b.data(),1,n,f)!=(size_t)n){fprintf(stderr,"short read %s\n",p);exit(1);}fclose(f);return b;}
int main(int argc,char**argv){
  const char* U="usage: vkasm <spv> <n_storage_buffers> <push_const_bytes> <subgroup_size|0> <spec0,spec1,...>";
  if(argc>1&&(!strcmp(argv[1],"-h")||!strcmp(argv[1],"--help"))){puts(U);return 0;}
  if(argc<6){fprintf(stderr,"%s\n",U);return 2;}
  auto spv=rd(argv[1]); uint32_t nbuf=atoi(argv[2]); uint32_t pcs=atoi(argv[3]); uint32_t sg=atoi(argv[4]);
  std::vector<uint32_t> spec; for(char*t=strtok(argv[5],",");t;t=strtok(nullptr,",")) spec.push_back(strtoul(t,nullptr,10));
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion=VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai;
  VkInstance inst; if(vkCreateInstance(&ici,nullptr,&inst)){fprintf(stderr,"no instance\n");return 1;}
  uint32_t n=1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst,&n,&pd);
  VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(pd,&pp); fprintf(stderr,"device: %s\n",pp.deviceName);
  const char* exts[]={"VK_KHR_cooperative_matrix","VK_KHR_shader_float16_int8","VK_KHR_16bit_storage","VK_KHR_shader_subgroup_extended_types","VK_EXT_subgroup_size_control","VK_KHR_shader_integer_dot_product","VK_KHR_8bit_storage","VK_KHR_pipeline_executable_properties"};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR}; cm.cooperativeMatrix=VK_TRUE;
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}; f13.pNext=&cm; f13.subgroupSizeControl=VK_TRUE; f13.computeFullSubgroups=VK_TRUE;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext=&f13; f12.shaderFloat16=VK_TRUE; f12.shaderInt8=VK_TRUE; f12.storageBuffer8BitAccess=VK_TRUE; f12.uniformAndStorageBuffer8BitAccess=VK_TRUE; f12.shaderSubgroupExtendedTypes=VK_TRUE; f12.vulkanMemoryModel=VK_TRUE; f12.vulkanMemoryModelDeviceScope=VK_TRUE; f12.bufferDeviceAddress=VK_TRUE; f12.shaderBufferInt64Atomics=VK_TRUE;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.pNext=&f12; f11.storageBuffer16BitAccess=VK_TRUE; f11.uniformAndStorageBuffer16BitAccess=VK_TRUE;
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext=&f11; f2.features.shaderInt16=VK_TRUE; f2.features.shaderInt64=VK_TRUE; f2.features.shaderFloat64=VK_FALSE;
  float prio=1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueCount=1; qci.pQueuePriorities=&prio;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext=&f2; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=sizeof(exts)/sizeof(*exts); dci.ppEnabledExtensionNames=exts;
  VkDevice dev; if(VkResult r=vkCreateDevice(pd,&dci,nullptr,&dev)){fprintf(stderr,"no device %d\n",r);return 1;}
  std::vector<VkDescriptorSetLayoutBinding> b(nbuf); for(uint32_t i=0;i<nbuf;i++){b[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};}
  VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount=nbuf; dl.pBindings=b.data(); VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev,&dl,nullptr,&dsl);
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,pcs}; VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount=1; pl.pSetLayouts=&dsl; pl.pushConstantRangeCount=pcs?1:0; pl.pPushConstantRanges=&pcr; VkPipelineLayout lay; vkCreatePipelineLayout(dev,&pl,nullptr,&lay);
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=spv.size(); smi.pCode=(const uint32_t*)spv.data(); VkShaderModule sm; if(vkCreateShaderModule(dev,&smi,nullptr,&sm)){fprintf(stderr,"bad spv\n");return 1;}
  std::vector<VkSpecializationMapEntry> me(spec.size()); for(size_t i=0;i<spec.size();i++) me[i]={(uint32_t)i,(uint32_t)(i*4),4};
  VkSpecializationInfo si{(uint32_t)me.size(),me.data(),spec.size()*4,spec.data()};
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO}; rss.requiredSubgroupSize=sg;
  VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st.pNext=sg?&rss:nullptr; st.stage=VK_SHADER_STAGE_COMPUTE_BIT; st.module=sm; st.pName="main"; st.pSpecializationInfo=&si; st.flags=VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.stage=st; cpi.layout=lay;
  VkPipeline pipe; VkResult r=vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpi,nullptr,&pipe); fprintf(stderr,"pipeline rc=%d\n",r);
  // executable statistics (VGPRs, LDS, code size)
  auto fn=(PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(dev,"vkGetPipelineExecutableStatisticsKHR");
  if(fn&&r==VK_SUCCESS){VkPipelineExecutableInfoKHR ei{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR}; ei.pipeline=pipe; ei.executableIndex=0; uint32_t c=0; fn(dev,&ei,&c,nullptr); std::vector<VkPipelineExecutableStatisticKHR> stv(c,{VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR}); fn(dev,&ei,&c,stv.data());
    for(auto&s:stv){ if(s.format==VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR) fprintf(stderr,"STAT %s = %llu\n",s.name,(unsigned long long)s.value.u64); else if(s.format==VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_FLOAT64_KHR) fprintf(stderr,"STAT %s = %f\n",s.name,s.value.f64);} }
  return r!=VK_SUCCESS;
}
