// vkgemm: dispatch a ggml-vulkan mul_mm.comp pipeline (non-ID variant: 3 storage buffers A, B, D + push constants) on
// synthetic data and time it, outside ggml. Use it to time one GEMM shader variant (spec constants, tile shape) in
// isolation, and with rot > 1 to see what a cold output buffer costs compared with re-writing the same region.
//
// usage: vkgemm <spv> <M> <N> <K> <a_bytes_per_row> <groups_x> <groups_y> <reps> <spec0,spec1,...> [rot]
//   spv              a mul_mm SPIR-V from the ggml-vulkan shader build (the .spv glslc wrote, before embedding)
//   a_bytes_per_row  bytes of one A row in its storage type (e.g. K/256*144 for q4_K, K*2 for f16)
//   groups_x/y       dispatch size; ggml uses ceil(M/BM) x ceil(N/BN)
//   spec             the pipeline's spec constants in constant_id order (copy them from ggml-vulkan.cpp for that variant)
//   rot              > 1: rotate the output over rot distinct regions (cold-output test); default 1 = same region every rep
// Push constants: the first 16 words follow upstream mul_mm.comp's non-ID parameter block (M N K, strides, batch strides,
// base_work_group_z, num_batches, k_split, ne02, ne12, broadcast2, broadcast3); a 17th word (padded_N, older trees) is
// also pushed and is ignored by shaders that do not declare it. A data is random bits with every 36th dword set to small
// f16 pairs (sane q4_K / q8_0 / q6_K scales); B is f32 in [-0.5, 0.5). Required subgroup size 32.
// Prints best and warm mean in us and TFLOPS(best) = 2*M*N*K / best. Not a correctness check: d[0] is printed only so
// the result stays live.
// build: g++ -O2 -o vkgemm vkgemm.cpp -lvulkan   (or bench/vk/build.sh)
#include <vulkan/vulkan.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
static std::vector<char> rd(const char*p){FILE*f=fopen(p,"rb");if(!f){perror(p);exit(1);}fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);std::vector<char>b(n);if(fread(b.data(),1,n,f)!=(size_t)n)exit(1);fclose(f);return b;}
struct PC { uint32_t M,N,K,sa,sb,sd,bsa,bsb,bsd,bwz,nb,ks,ne02,ne12,b2,b3,pN; };
int main(int argc,char**argv){
  const char* U="usage: vkgemm <spv> <M> <N> <K> <a_bytes_per_row> <groups_x> <groups_y> <reps> <spec0,spec1,...> [rot]";
  if(argc>1&&(!strcmp(argv[1],"-h")||!strcmp(argv[1],"--help"))){puts(U);return 0;}
  if(argc<10){fprintf(stderr,"%s\n",U);return 2;}
  auto spv=rd(argv[1]); uint32_t M=atoi(argv[2]),N=atoi(argv[3]),K=atoi(argv[4]); uint32_t abpr=atoi(argv[5]); uint32_t gx=atoi(argv[6]),gy=atoi(argv[7]); int reps=atoi(argv[8]);
  std::vector<uint32_t> spec; for(char*t=strtok(argv[9],",");t;t=strtok(nullptr,",")) spec.push_back(strtoul(t,nullptr,10));
  uint32_t rot = argc>10 ? (uint32_t)atoi(argv[10]) : 1; if(rot<1) rot=1;
  VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO}; ai.apiVersion=VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo=&ai;
  VkInstance inst; if(vkCreateInstance(&ici,nullptr,&inst)){fprintf(stderr,"no instance\n");return 1;}
  uint32_t n=1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst,&n,&pd);
  const char* exts[]={"VK_KHR_cooperative_matrix","VK_KHR_shader_float16_int8","VK_KHR_16bit_storage","VK_KHR_shader_subgroup_extended_types","VK_EXT_subgroup_size_control","VK_KHR_8bit_storage"};
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cm{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR}; cm.cooperativeMatrix=VK_TRUE;
  VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES}; f13.pNext=&cm; f13.subgroupSizeControl=VK_TRUE; f13.computeFullSubgroups=VK_TRUE;
  VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES}; f12.pNext=&f13; f12.shaderFloat16=VK_TRUE; f12.shaderInt8=VK_TRUE; f12.storageBuffer8BitAccess=VK_TRUE; f12.uniformAndStorageBuffer8BitAccess=VK_TRUE; f12.shaderSubgroupExtendedTypes=VK_TRUE; f12.vulkanMemoryModel=VK_TRUE; f12.vulkanMemoryModelDeviceScope=VK_TRUE;
  VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES}; f11.pNext=&f12; f11.storageBuffer16BitAccess=VK_TRUE; f11.uniformAndStorageBuffer16BitAccess=VK_TRUE;
  VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2}; f2.pNext=&f11; f2.features.shaderInt16=VK_TRUE; f2.features.shaderInt64=VK_TRUE;
  float prio=1; VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qci.queueCount=1; qci.pQueuePriorities=&prio;
  uint32_t qn=0; vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,nullptr); std::vector<VkQueueFamilyProperties> qp(qn); vkGetPhysicalDeviceQueueFamilyProperties(pd,&qn,qp.data());
  uint32_t qf=0; for(uint32_t i=0;i<qn;i++) if(qp[i].queueFlags&VK_QUEUE_COMPUTE_BIT){qf=i;break;} qci.queueFamilyIndex=qf;
  VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; dci.pNext=&f2; dci.queueCreateInfoCount=1; dci.pQueueCreateInfos=&qci; dci.enabledExtensionCount=sizeof(exts)/sizeof(*exts); dci.ppEnabledExtensionNames=exts;
  VkDevice dev; if(VkResult r=vkCreateDevice(pd,&dci,nullptr,&dev)){fprintf(stderr,"no device %d\n",r);return 1;}
  VkQueue q; vkGetDeviceQueue(dev,qf,0,&q);
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd,&mp);
  auto mkbuf=[&](VkDeviceSize sz, VkBuffer&buf, void*&map){ VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bci.size=sz; bci.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT; vkCreateBuffer(dev,&bci,nullptr,&buf);
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev,buf,&mr); uint32_t mt=0; for(uint32_t i=0;i<mp.memoryTypeCount;i++) if((mr.memoryTypeBits&(1u<<i)) && (mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) && (mp.memoryTypes[i].propertyFlags&VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)){mt=i;break;}
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; mai.allocationSize=mr.size; mai.memoryTypeIndex=mt; VkDeviceMemory mem; vkAllocateMemory(dev,&mai,nullptr,&mem); vkBindBufferMemory(dev,buf,mem,0); vkMapMemory(dev,mem,0,sz,0,&map); };
  VkDeviceSize szA=(VkDeviceSize)M*abpr+4096, szB=(VkDeviceSize)N*K*4+4096, szD1=(VkDeviceSize)M*N*4+4096, szD=szD1*rot;
  VkBuffer bA,bB,bD; void *mA,*mB,*mD; mkbuf(szA,bA,mA); mkbuf(szB,bB,mB); mkbuf(szD,bD,mD);
  { uint32_t x=12345; uint32_t* p=(uint32_t*)mA; for(size_t i=0;i<szA/4;i++){ x=x*1664525u+1013904223u; p[i]=x; }
    // keep f16 scales sane: q4_K blocks start with dm (2 f16); force every 36th dword's halves to small values
    for(size_t i=0;i+36<=szA/4;i+=36) p[i]=0x2c002c00u; // 0.0156 f16 pairs (also fine as a q8_0/q6_K d)
    float* f=(float*)mB; for(size_t i=0;i<szB/4;i++){ x=x*1664525u+1013904223u; f[i]=((x>>9)&0xffff)/65536.0f-0.5f; } }
  std::vector<VkDescriptorSetLayoutBinding> b(3); for(uint32_t i=0;i<3;i++) b[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
  VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; dl.bindingCount=3; dl.pBindings=b.data(); VkDescriptorSetLayout dsl; vkCreateDescriptorSetLayout(dev,&dl,nullptr,&dsl);
  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(PC)}; VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}; pl.setLayoutCount=1; pl.pSetLayouts=&dsl; pl.pushConstantRangeCount=1; pl.pPushConstantRanges=&pcr; VkPipelineLayout lay; vkCreatePipelineLayout(dev,&pl,nullptr,&lay);
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=spv.size(); smi.pCode=(const uint32_t*)spv.data(); VkShaderModule sm; if(vkCreateShaderModule(dev,&smi,nullptr,&sm)){fprintf(stderr,"bad spv\n");return 1;}
  std::vector<VkSpecializationMapEntry> me(spec.size()); for(size_t i=0;i<spec.size();i++) me[i]={(uint32_t)i,(uint32_t)(i*4),4};
  VkSpecializationInfo si{(uint32_t)me.size(),me.data(),spec.size()*4,spec.data()};
  VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO}; rss.requiredSubgroupSize=32;
  VkPipelineShaderStageCreateInfo st{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}; st.pNext=&rss; st.stage=VK_SHADER_STAGE_COMPUTE_BIT; st.module=sm; st.pName="main"; st.pSpecializationInfo=&si; st.flags=VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
  VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO}; cpi.stage=st; cpi.layout=lay;
  VkPipeline pipe; if(VkResult r=vkCreateComputePipelines(dev,VK_NULL_HANDLE,1,&cpi,nullptr,&pipe)){fprintf(stderr,"pipeline rc=%d\n",r);return 1;}
  VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3*rot}; VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO}; dpi.maxSets=rot; dpi.poolSizeCount=1; dpi.pPoolSizes=&ps; VkDescriptorPool dp; vkCreateDescriptorPool(dev,&dpi,nullptr,&dp);
  std::vector<VkDescriptorSetLayout> dsls(rot,dsl); VkDescriptorSetAllocateInfo dsa{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO}; dsa.descriptorPool=dp; dsa.descriptorSetCount=rot; dsa.pSetLayouts=dsls.data(); std::vector<VkDescriptorSet> dss(rot); vkAllocateDescriptorSets(dev,&dsa,dss.data());
  for(uint32_t sl=0;sl<rot;sl++){ VkDescriptorBufferInfo dbi[3]={{bA,0,szA},{bB,0,szB},{bD,(VkDeviceSize)sl*szD1,szD1}}; VkWriteDescriptorSet w[3]; for(int i=0;i<3;i++){ w[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w[i].dstSet=dss[sl]; w[i].dstBinding=i; w[i].descriptorCount=1; w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo=&dbi[i]; } vkUpdateDescriptorSets(dev,3,w,0,nullptr); }
  VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; cpc.queueFamilyIndex=qf; VkCommandPool cp; vkCreateCommandPool(dev,&cpc,nullptr,&cp);
  VkCommandBufferAllocateInfo cba{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; cba.commandPool=cp; cba.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cba.commandBufferCount=rot; std::vector<VkCommandBuffer> cbs(rot); vkAllocateCommandBuffers(dev,&cba,cbs.data());
  PC pc{M,N,K,K,K,M,M*K,N*K,M*N,0,1,K,1,1,1,1,N};
  for(uint32_t sl=0;sl<rot;sl++){ VkCommandBuffer cb=cbs[sl]; VkCommandBufferBeginInfo cbb{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}; vkBeginCommandBuffer(cb,&cbb);
    vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipe); vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,lay,0,1,&dss[sl],0,nullptr); vkCmdPushConstants(cb,lay,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(PC),&pc); vkCmdDispatch(cb,gx,gy,1); vkEndCommandBuffer(cb); }
  double best=1e30, sum=0; int cnt=0;
  for(int r=0;r<reps;r++){ VkSubmitInfo sub{VK_STRUCTURE_TYPE_SUBMIT_INFO}; sub.commandBufferCount=1; sub.pCommandBuffers=&cbs[r%rot]; auto t0=std::chrono::steady_clock::now(); vkQueueSubmit(q,1,&sub,VK_NULL_HANDLE); vkQueueWaitIdle(q); double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); if(ms<best)best=ms; if(r>=(int)rot){sum+=ms;cnt++;} }
  double mean = cnt? sum/cnt : best;
  double tf=2.0*M*N*K/(best*1e-3)/1e12;
  printf("M=%u N=%u K=%u rot=%u best=%.1f us mean(after warm)=%.1f us  %.2f TFLOPS(best)  d[0]=%g\n",M,N,K,rot,best*1e3,mean*1e3,tf,((float*)mD)[0]);
  return 0;
}
