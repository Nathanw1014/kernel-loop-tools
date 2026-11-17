// balloon: hold <GiB> of anonymous memory with every page touched, mlock as much as RLIMIT_MEMLOCK allows, then sleep
// until killed. A POSITIVE CONTROL for memory pressure: it makes a big machine behave like a smaller one, so you can test
// whether a load path, a memory gate or a "fits in RAM" claim still holds when GTT and the page cache must share less.
//
// usage: balloon <GiB> <readyfile>
//   writes "pid <pid> bytes <n> locked <n> mlock_rc <rc>" to <readyfile> once every page is touched (wait for that file
//   before starting the job under test); kill it to release the memory.
// To keep the unlocked part from being swapped out too: systemd-run --user --scope -p MemorySwapMax=0 ./balloon 16 ready
// Careful: it takes the memory for real. Leave headroom for the desktop and for the job, or the OOM killer chooses.
// build: gcc -O2 -o balloon balloon.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <unistd.h>
int main(int argc, char ** argv) {
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) { puts("usage: balloon <GiB> <readyfile>"); return 0; }
    if (argc != 3 || atof(argv[1]) <= 0) { fprintf(stderr, "usage: balloon <GiB> <readyfile>\n"); return 1; }
    size_t n = (size_t) (atof(argv[1]) * 1073741824.0) & ~4095UL;
    char * p = mmap(NULL, n, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) { perror("mmap"); return 2; }
    for (size_t i = 0; i < n; i += 4096) p[i] = 1;
    struct rlimit r; getrlimit(RLIMIT_MEMLOCK, &r);
    size_t l = r.rlim_cur == RLIM_INFINITY || r.rlim_cur > n ? n : r.rlim_cur & ~4095UL;
    int rc = l ? mlock(p, l) : 0;
    FILE * f = fopen(argv[2], "w");
    if (!f) return 3;
    fprintf(f, "pid %d bytes %zu locked %zu mlock_rc %d\n", getpid(), n, l, rc);
    fclose(f);
    for (;;) pause();
}
