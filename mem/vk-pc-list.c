// vk-pc-list: list VK_KHR_performance_query counters of physical device 0, queue family 0 (no GPU work; enumeration only).
// Tells you which hardware counters the Vulkan driver exposes (and at what scope) before you plan on one.
// build: cc -O2 vk-pc-list.c -o vk-pc-list -lvulkan
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "vk-pc-list", 1, 0, 0, VK_API_VERSION_1_2 };
    VkInstanceCreateInfo ci = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &ai };
    VkInstance inst; if (vkCreateInstance(&ci, 0, &inst)) { puts("no instance"); return 1; }
    uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &n, &pd);
    PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR f = (void *) vkGetInstanceProcAddr(inst, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
    if (!f) { puts("no VK_KHR_performance_query entry point"); return 1; }
    uint32_t c = 0; f(pd, 0, &c, 0, 0);
    VkPerformanceCounterKHR * pc = calloc(c, sizeof *pc); VkPerformanceCounterDescriptionKHR * pdsc = calloc(c, sizeof *pdsc);
    for (uint32_t i = 0; i < c; ++i) { pc[i].sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR; pdsc[i].sType = VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR; }
    f(pd, 0, &c, pc, pdsc);
    static const char * U[] = { "generic", "percent", "ns", "bytes", "B/s", "K", "W", "V", "A", "Hz", "cycles" };
    printf("%u counters (queue family 0)\n", c);
    for (uint32_t i = 0; i < c; ++i) printf("  %-22s %-20s scope=%d unit=%s : %s\n", pdsc[i].category, pdsc[i].name, pc[i].scope, pc[i].unit < 11 ? U[pc[i].unit] : "?", pdsc[i].description);
    vkDestroyInstance(inst, 0); return 0;
}
