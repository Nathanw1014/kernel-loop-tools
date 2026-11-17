// camp-hip: memory-channel camping positive control.
// Reads CHUNK bytes at the start of every row of a STRIDE-byte row pitch over a BUF-byte buffer,
// back to back for SECS seconds. A stride that is a multiple of the channel interleave period
// with a small chunk concentrates every request on a few channels (camping); a padded stride or
// a full-row read spreads them.
// Use it as the positive control for gcea-sample / chan-run.sh: sweep the stride to find the interleave period
// (the camped strides all land on one instance), then check that a padded stride spreads evenly.
// usage: camp-hip <stride_B> <chunk_B> [secs=6] [buf_MiB=1024]
// prints: stride chunk rows calls us/call touched_GB/s
// build: hipcc -O3 camp-hip.cpp -o camp-hip
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#define HC(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(e_)); return 1; } } while (0)

// one block of 256 threads per 64 rows; each thread loads 16 B (uint4)
__global__ void rd(const uint4* __restrict__ buf, size_t stride16, unsigned chunk16, size_t rows,
                   unsigned* __restrict__ sink) {
    size_t   lane  = threadIdx.x % chunk16;
    size_t   rsub  = threadIdx.x / chunk16;
    size_t   rpb   = blockDim.x / chunk16;
    uint32_t acc   = 0;
    for (size_t r = blockIdx.x * rpb + rsub; r < rows; r += (size_t)gridDim.x * rpb) {
        uint4 v = buf[r * stride16 + lane];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;  // keep the loads
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: camp-hip <stride_B> <chunk_B> [secs] [buf_MiB]\n"); return 2; }
    size_t   stride = strtoull(argv[1], 0, 0);
    unsigned chunk  = (unsigned)strtoul(argv[2], 0, 0);
    double   secs   = argc > 3 ? atof(argv[3]) : 6.0;
    size_t   bufb   = (argc > 4 ? strtoull(argv[4], 0, 0) : 1024) << 20;
    if (stride % 16 || chunk % 16 || chunk > stride || chunk > 4096 || 256 % (chunk / 16)) {
        fprintf(stderr, "stride, chunk multiples of 16; chunk <= stride; 256 %% (chunk/16) == 0\n"); return 2;
    }
    size_t rows = bufb / stride;
    uint4* buf; unsigned* sink;
    HC(hipMalloc(&buf, bufb));
    HC(hipMalloc(&sink, 4));
    HC(hipMemset(buf, 1, bufb));
    HC(hipDeviceSynchronize());
    unsigned c16 = chunk / 16, threads = 256;
    if (c16 > threads) threads = c16;
    int grid = 4096;
    auto launch = [&]() { hipLaunchKernelGGL(rd, dim3(grid), dim3(threads), 0, 0, buf, stride / 16, c16, rows, sink); };
    launch(); HC(hipDeviceSynchronize());
    auto t0 = std::chrono::steady_clock::now();
    long calls = 0;
    double el = 0;
    while (el < secs) {
        for (int i = 0; i < 8; ++i) launch();
        HC(hipDeviceSynchronize());
        calls += 8;
        el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
    double us = el * 1e6 / calls;
    double gbs = (double)rows * chunk / (us * 1e-6) / 1e9;
    printf("stride %zu chunk %u rows %zu calls %ld us/call %.1f touched_GB/s %.1f\n", stride, chunk, rows, calls, us, gbs);
    HC(hipFree(buf)); HC(hipFree(sink));
    return 0;
}
