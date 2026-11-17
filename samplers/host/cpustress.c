// cpustress: pinned CPU load of a chosen kind, as a POSITIVE CONTROL for host contention: run it beside a GPU job and see
// whether (and through which resource) the job slows down. On an APU the CPU and GPU share the memory controller, the
// power budget and the clocks, so "the CPU was idle" is a claim to test, not assume.
//
// usage: cpustress <fma|l2|bw|spin> <threads> [first_cpu]     runs until SIGTERM / SIGINT, one throughput line per second
//   fma  : AVX-512 FMA on 16 zmm accumulators, no memory traffic (power / clock contention). Unit GFLOP/s.
//   l2   : AVX-512 FMA streaming over a 512 KB per-thread buffer (L2-resident). Unit GB/s of L2 reads.
//   bw   : STREAM copy over 512 MB per-thread src->dst (DRAM bandwidth contention). Unit GB/s (read + write bytes).
//   spin : _mm_pause loop (keeps cores out of C-states at minimal power). Unit Gpause/s.
// Threads pin to cpus first_cpu .. first_cpu+threads-1 (default 0). Check `lscpu -e` for which cpu ids are SMT siblings.
// fma / l2 / bw need AVX-512 (checked at start); spin runs anywhere x86.
// Output line: "<unix time> <rate> <unit>". bw allocates 1 GiB per thread.
// build: gcc -O2 -mavx512f -pthread -o cpustress cpustress.c
#define _GNU_SOURCE
#include <immintrin.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static volatile sig_atomic_t stop_flag = 0;
static void on_sig(int s) { (void)s; stop_flag = 1; }
static int mode; // 0 fma, 1 l2, 2 bw, 3 spin
static _Atomic unsigned long long work[64];
static double sink[64];

static void *worker(void *arg) {
    long id = (long)arg;
    if (mode == 0) {
        __m512 a[16], b = _mm512_set1_ps(0.999999f), c = _mm512_set1_ps(1e-7f);
        for (int i = 0; i < 16; i++) a[i] = _mm512_set1_ps((float)i);
        while (!stop_flag) {
            for (int it = 0; it < 4096; it++)
                { _Pragma("GCC unroll 16") for (int i = 0; i < 16; i++) a[i] = _mm512_fmadd_ps(a[i], b, c);
                  _Pragma("GCC unroll 16") for (int i = 0; i < 16; i++) __asm__ volatile("" : "+v"(a[i])); }
            atomic_fetch_add_explicit(&work[id], 4096ull * 16 * 16 * 2, memory_order_relaxed); // flops
        }
        float s = 0; for (int i = 0; i < 16; i++) s += _mm512_reduce_add_ps(a[i]);
        sink[id] = s;
    } else if (mode == 1) {
        size_t n = 512 * 1024 / sizeof(float);
        float *buf = aligned_alloc(64, n * sizeof(float));
        for (size_t i = 0; i < n; i++) buf[i] = 1.0f;
        __m512 acc[8]; for (int i = 0; i < 8; i++) acc[i] = _mm512_setzero_ps();
        __m512 w = _mm512_set1_ps(0.5f);
        while (!stop_flag) {
            for (size_t i = 0; i < n; i += 128)
                { _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++) acc[k] = _mm512_fmadd_ps(_mm512_load_ps(buf + i + 16 * k), w, acc[k]);
                  _Pragma("GCC unroll 8") for (int k = 0; k < 8; k++) __asm__ volatile("" : "+v"(acc[k])); }
            atomic_fetch_add_explicit(&work[id], n * sizeof(float), memory_order_relaxed); // bytes
        }
        float s = 0; for (int i = 0; i < 8; i++) s += _mm512_reduce_add_ps(acc[i]);
        sink[id] = s; free(buf);
    } else if (mode == 3) {
        while (!stop_flag) {
            for (int it = 0; it < 4096; it++) _mm_pause();
            atomic_fetch_add_explicit(&work[id], 4096, memory_order_relaxed);
        }
    } else {
        size_t n = 512ull * 1024 * 1024 / sizeof(double);
        double *src = aligned_alloc(64, n * sizeof(double)), *dst = aligned_alloc(64, n * sizeof(double));
        for (size_t i = 0; i < n; i++) { src[i] = 1.0; dst[i] = 0.0; }
        const size_t chunk = 4u << 20; // 4 M doubles = 32 MB per step, so stop is checked often
        size_t off = 0;
        while (!stop_flag) {
            size_t m = n - off < chunk ? n - off : chunk;
            for (size_t i = 0; i < m; i += 8) _mm512_store_pd(dst + off + i, _mm512_load_pd(src + off + i));
            atomic_fetch_add_explicit(&work[id], m * sizeof(double) * 2, memory_order_relaxed);
            off = (off + m) % n;
        }
        sink[id] = dst[id]; free(src); free(dst);
    }
    return NULL;
}

int main(int argc, char **argv) {
    const char *U = "usage: cpustress <fma|l2|bw|spin> <threads 1..64> [first_cpu]";
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) { puts(U); return 0; }
    if (argc < 3) { fprintf(stderr, "%s\n", U); return 2; }
    mode = !strcmp(argv[1], "fma") ? 0 : !strcmp(argv[1], "l2") ? 1 : !strcmp(argv[1], "bw") ? 2 : !strcmp(argv[1], "spin") ? 3 : -1;
    if (mode < 0) { fprintf(stderr, "%s\n", U); return 2; }
    int nt = atoi(argv[2]), first = argc > 3 ? atoi(argv[3]) : 0;
    if (nt < 1 || nt > 64 || first < 0) { fprintf(stderr, "%s\n", U); return 2; }
    if (mode != 3 && !__builtin_cpu_supports("avx512f")) { fprintf(stderr, "cpustress: %s needs AVX-512F; this CPU has none (spin works)\n", argv[1]); return 2; }
    signal(SIGTERM, on_sig); signal(SIGINT, on_sig);
    pthread_t th[64];
    for (long i = 0; i < nt; i++) {
        pthread_attr_t at; pthread_attr_init(&at);
        cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(first + i, &cs);
        pthread_attr_setaffinity_np(&at, sizeof cs, &cs);
        pthread_create(&th[i], &at, worker, (void *)i);
    }
    const char *unit = mode == 0 ? "GFLOP/s" : mode == 3 ? "Gpause/s" : "GB/s";
    unsigned long long prev = 0;
    struct timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
    double tp = 0;
    while (!stop_flag) {
        sleep(1);
        struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
        double el = (t.tv_sec - t0.tv_sec) + (t.tv_nsec - t0.tv_nsec) * 1e-9;
        unsigned long long tot = 0; for (int i = 0; i < nt; i++) tot += work[i];
        printf("%ld %.2f %s\n", (long)time(NULL), (tot - prev) / (el - tp) / 1e9, unit); fflush(stdout);
        prev = tot; tp = el;
    }
    for (int i = 0; i < nt; i++) pthread_join(th[i], NULL);
    return 0;
}
