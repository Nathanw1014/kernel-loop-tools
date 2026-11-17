/* qoshold.c: PM QoS CPU latency helper for job-run QOS_US=<us> cells (jobs/qos-run).
 *   qoshold read                      : print the aggregate CPU latency target (us) from /dev/cpu_dma_latency, exit.
 *   qoshold hold <us> <pid> [max_s]   : write <us> as this process's request and keep the fd open; prints "pid <own pid> us <n>".
 * The request exists only while this process lives: the kernel drops it when the fd closes, on any exit including SIGKILL.
 * Exits on SIGTERM / SIGINT, when <pid> (the cell wrapper) is gone (polled every 200 ms), or after max_s (default 43200).
 * Needs write access to /dev/cpu_dma_latency (root, or a udev rule giving your group access). It writes nothing else anywhere.
 * Build: gcc -O2 -o qoshold qoshold.c */
#define _GNU_SOURCE
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
static volatile sig_atomic_t stop = 0;
static void on_term(int s) { (void) s; stop = 1; }
int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "read") == 0) {
        int fd = open("/dev/cpu_dma_latency", O_RDONLY);
        if (fd < 0) { perror("open /dev/cpu_dma_latency"); return 1; }
        int32_t v = 0;
        if (read(fd, &v, sizeof v) != sizeof v) { perror("read"); return 1; }
        printf("%d\n", v);
        close(fd);
        return 0;
    }
    if ((argc != 4 && argc != 5) || strcmp(argv[1], "hold") != 0) { fprintf(stderr, "usage: qoshold read | qoshold hold <us> <watch pid> [max_s]\n"); return 2; }
    int32_t us = (int32_t) atoi(argv[2]); pid_t watch = (pid_t) atoi(argv[3]); long max_s = argc == 5 ? atol(argv[4]) : 43200;
    if (us < 1 || us > 100000 || watch < 2 || max_s < 1) { fprintf(stderr, "qoshold: bad arguments\n"); return 2; }
    signal(SIGTERM, on_term); signal(SIGINT, on_term);
    int fd = open("/dev/cpu_dma_latency", O_WRONLY);
    if (fd < 0) { perror("open /dev/cpu_dma_latency"); return 1; }
    if (write(fd, &us, sizeof us) != sizeof us) { perror("write"); return 1; }
    printf("pid %d us %d\n", (int) getpid(), (int) us); fflush(stdout);
    char p[64]; snprintf(p, sizeof p, "/proc/%d", (int) watch);
    struct timespec d = { 0, 200 * 1000 * 1000 };
    time_t t0 = time(NULL);
    while (!stop && access(p, F_OK) == 0 && time(NULL) - t0 < max_s) nanosleep(&d, NULL);
    close(fd);
    fprintf(stderr, "qoshold: released %d us after %ld s (%s)\n", (int) us, (long) (time(NULL) - t0), stop ? "signal" : (access(p, F_OK) == 0 ? "max lifetime" : "watched pid gone"));
    return 0;
}
