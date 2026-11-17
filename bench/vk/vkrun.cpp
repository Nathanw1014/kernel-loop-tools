// vkrun: dispatch any compute shader (one storage buffer at binding 0, 64 MiB, host-visible device-local) with given
// specialization constants and time it: wall clock around submit + queue wait, best of <reps>. Plain Vulkan, no ggml.
//
// usage: vkrun <spv> <groups> <reps> <subgroup_size|0> <spec0,spec1,...>
//   groups         workgroups in x (y = z = 1)
//   subgroup_size  0 = driver choice; 32 / 64 = required subgroup size with full subgroups
//   spec           comma list of uint32 spec constants, constant_id 0, 1, 2, ...
// The buffer holds 0.001*(i%7) floats with o[300] = 1.0001, o[301] = 0.0001 (the peak shaders read those).
// Prints "groups=<g> best=<ms> ms". A best-of wall time includes the submit cost: fit a slope over two sizes to cancel
// it (peak.py does). Enables cooperative matrix, f16/int8, 8/16-bit storage and subgroup size control.
// build: g++ -O2 -o vkrun vkrun.cpp -lvulkan   (or bench/vk/build.sh)
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
static std::vector<char> rd(const char*p){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<char>b(n);if(fread(b.data(),1,n,f)!=(size_t)n)exit(1);fclose(f);return b;}
int main(int argc,char**argv){
  if(argc>1&&(!strcmp(argv[1],"-h")||!strcmp(argv[1],"--help"))){puts("usage: vkrun <spv> <groups> <reps> <subgroup_size|0> <spec0,spec1,...>");return 0;}
  if(argc<6){fprintf(stderr,"usage: vkrun <spv> <groups> <reps> <subgroup_size|0> <spec0,spec1,...>\n");return 2;}
  auto spv=rd(argv[1]); uint32_t groups=atoi(argv[2]); int reps=atoi(argv[3]); uint32_t sg=atoi(argv[4]);
  std::vector<uint32_t> spec; for(char*t=strtok(argv[5],",");t;t=strtok(nullptr,",")) spec.push_back(strtoul(t,nullptr,10));
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion=VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai;
  VkInstance inst; if(vkCreateInstance(&ici,nullptr,&inst)){fprintf(stderr,"no instance\n");return 1;}
  uint32_t n=1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst,&n,&pd);
  const char* exts[]={"VK_KHR_cooperative_matrix","VK_KHR_shader_float16_int8","VK_KHR_16bit_storage","VK_KHR_shader_subgroup_extended_types","VK_EXT_subgroup_size_control","VK_KHR_8bit_storage"};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR}; cm.cooperativeMatrix=VK_TRUE;
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}; f13.pNext=&cm; f13.subgroupSizeControl=VK_TRUE; f13.computeFullSubgroups=VK_TRUE;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext=&f13; f12.shaderFloat16=VK_TRUE; f12.shaderInt8=VK_TRUE; f12.storageBuffer8BitAccess=VK_TRUE; f12.shaderSubgroupExtendedTypes=VK_TRUE; f12.vulkanMemoryModel=VK_TRUE; f12.vulkanMemoryModelDeviceScope=VK_TRUE;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.pNext=&f12; f11.storageBuffer16BitAccess=VK_TRUE;
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext=&f11; f2.features.shaderInt16=VK_TRUE;
  float prio=1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueCount=1; qci.pQueuePriorities=&prio;
  uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,nullptr); std::vector<VkQueueFamilyProperties> qp(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,qp.data());
  uint32_t qf=0; for(uint32_t i=0;i<qn;i++) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qf=i;break;} qci.queueFamilyIndex=qf;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext=&f2; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=sizeof(exts)/sizeof(*exts); dci.ppEnabledExtensionNames=exts;
  VkDevice dev; if(VkResult r=vkCreateDevice(pd,&dci,nullptr,&dev)){fprintf(stderr,"no device %d\n",r);return 1;}
  VkQueue q; vkGetDeviceQueue(dev,qf,0,&q);
  // buffer 64 MB host-visible
  VkDeviceSize bsz=64u<<20; VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size=bsz; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; VkBuffer buf; vkCreateBuffer(dev,&bci,nullptr,&buf);
  VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf,&mr); VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  uint32_t mt=0; for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((mr.memoryTypeBits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){mt=i;break;}
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size; mai.memoryTypeIndex=mt; VkDeviceMemory mem; vkAllocateMemory(dev,&mai,nullptr,&mem); vkBindBufferMemory(dev,buf,mem,0);
  void* map; vkMapMemory(dev,mem,0,bsz,0,&map); float* fp=(float*)map; for(size_t i=0;i<bsz/4;i++) fp[i]=0.001f*(i%7); fp[300]=1.0001f; fp[301]=0.0001f;
  VkDescriptorSetLayoutBinding b{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
  VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount=1; dl.pBindings=&b; VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev,&dl,nullptr,&dsl);
  VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount=1; pl.pSetLayouts=&dsl; VkPipelineLayout lay; vkCreatePipelineLayout(dev,&pl,nullptr,&lay);
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=spv.size(); smi.pCode=(const uint32_t*)spv.data(); VkShaderModule sm; if(vkCreateShaderModule(dev,&smi,nullptr,&sm)){fprintf(stderr,"bad spv\n");return 1;}
  std::vector<VkSpecializationMapEntry> me(spec.size()); for(size_t i=0;i<spec.size();i++) me[i]={(uint32_t)i,(uint32_t)(i*4),4};
  VkSpecializationInfo si{(uint32_t)me.size(),me.data(),spec.size()*4,spec.data()};
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO}; rss.requiredSubgroupSize=sg;
  VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st.pNext=sg?&rss:nullptr; st.stage=VK_SHADER_STAGE_COMPUTE_BIT; st.module=sm; st.pName="main"; st.pSpecializationInfo=&si; st.flags=sg?VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT:0;
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.stage=st; cpi.layout=lay;
  VkPipeline pipe; if(VkResult r=vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpi,nullptr,&pipe)){fprintf(stderr,"pipeline rc=%d\n",r);return 1;}
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets=1; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; VkDescriptorPool dp; vkCreateDescriptorPool(dev,&dpi,nullptr,&dp);
  VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsa.descriptorPool=dp; dsa.descriptorSetCount=1; dsa.pSetLayouts=&dsl; VkDescriptorSet ds; vkAllocateDescriptorSets(dev,&dsa,&ds);
  VkDescriptorBufferInfo dbi{buf,0,bsz}; VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet=ds; w.dstBinding=0; w.descriptorCount=1; w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w.pBufferInfo=&dbi; vkUpdateDescriptorSets(dev,1,&w,0,nullptr);
  VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpc.queueFamilyIndex=qf; VkCommandPool cp; vkCreateCommandPool(dev,&cpc,nullptr,&cp);
  VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cba.commandPool=cp; cba.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount=1; VkCommandBuffer cb; vkAllocateCommandBuffers(dev,&cba,&cb);
  VkCommandBufferBeginInfo cbb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(cb,&cbb);
  vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe); vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,lay,0,1,&ds,0,nullptr); vkCmdDispatch(cb,groups,1,1); vkEndCommandBuffer(cb);
  VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO}; sub.commandBufferCount=1; sub.pCommandBuffers=&cb;
  double best=1e30;
  for(int r=0;r<reps;r++){ auto t0=std::chrono::steady_clock::now(); vkQueueSubmit(q,1,&sub,VK_NULL_HANDLE); vkQueueWaitIdle(q); double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); if(ms<best)best=ms; }
  printf("groups=%u best=%.3f ms\n",groups,best);
  return 0;
}
