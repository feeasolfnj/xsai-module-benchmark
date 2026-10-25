#ifndef BENCH_PMU_CYCLES_H
#define BENCH_PMU_CYCLES_H
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <dirent.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* Task-bound event opened before workers. Linux sums descendant CPU cycles.
 * Off-CPU waiting, kernel and hypervisor execution are excluded. User-space
 * synchronization, memory stalls and instrumentation remain included. */
typedef struct { uint64_t count, enabled, running; } pmu_read_t;
static int pmu_fd = -1, pmu_ready, pmu_threads, pmu_required, pmu_error;
static const char *pmu_status = "unavailable", *pmu_reason = "not_initialized";
static pmu_read_t pmu_before;
static uint64_t pmu_calls, pmu_units, pmu_sum, pmu_en, pmu_run, pmu_empty;

static void pmu_unavailable(const char *reason, int error, int invalid)
{
  pmu_status = invalid ? "invalid" : "unavailable";
  pmu_reason = reason; pmu_error = error;
  if (pmu_fd >= 0) { close(pmu_fd); pmu_fd = -1; }
  if (pmu_required) {
    fprintf(stderr, "Hardware cycles required: %s (errno=%d %s)\n",
            reason, error, error ? strerror(error) : "counter validation failed");
    exit(2);
  }
}

static int pmu_read(pmu_read_t *r)
{
  ssize_t n = read(pmu_fd, r, sizeof(*r));
  if (n == sizeof(*r)) return 1;
  pmu_unavailable("counter_read_failed", n < 0 ? errno : EIO, 1);
  return 0;
}

static int pmu_on(void)
{
  if (pmu_fd < 0 || !pmu_read(&pmu_before)) return 0;
  if (ioctl(pmu_fd, PERF_EVENT_IOC_ENABLE, 0)) {
    pmu_unavailable("counter_enable_failed", errno, 1); return 0;
  }
  __asm__ volatile("" ::: "memory");
  return 1;
}

static uint64_t pmu_off(int accumulate)
{
  __asm__ volatile("" ::: "memory");
  if (pmu_fd < 0) return 0;
  if (ioctl(pmu_fd, PERF_EVENT_IOC_DISABLE, 0)) {
    pmu_unavailable("counter_disable_failed", errno, 1); return 0;
  }
  pmu_read_t x;
  if (!pmu_read(&x)) return 0;
  uint64_t cy = x.count - pmu_before.count;
  uint64_t en = x.enabled - pmu_before.enabled;
  uint64_t run = x.running - pmu_before.running;
  if (x.count < pmu_before.count || x.enabled < pmu_before.enabled ||
      x.running < pmu_before.running || en != run || !run) {
    pmu_unavailable("unscheduled_or_multiplexed", 0, 1); return 0;
  }
  if (accumulate) { pmu_sum += cy; pmu_en += en; pmu_run += run; }
  return cy;
}

static void pmu_report(void)
{
  int required = pmu_required;
  pmu_required = 0; /* Do not recursively call exit() from an atexit handler. */
  if (pmu_fd >= 0 && (!pmu_calls || !pmu_units || !pmu_sum))
    pmu_unavailable("no_valid_measurements", 0, 1);
  printf("[PMU_STATUS] status=%s reason=%s errno=%d source=perf_event_hardware_cpu_cycles\n",
         pmu_status, pmu_reason, pmu_error);
  if (pmu_fd < 0) {
    if (required) { fflush(NULL); _Exit(2); }
    return;
  }
  printf("[PMU_CYCLES] status=available windows=%llu units=%llu threads=%d cycles=%llu "
         "enabled_ns=%llu running_ns=%llu cycles_per_unit=%.6f empty_window_cycles=%.6f "
         "raw=1 exclude_kernel=1 exclude_hv=1 inherited=1 scaled=0\n",
         (unsigned long long)pmu_calls, (unsigned long long)pmu_units, pmu_threads,
         (unsigned long long)pmu_sum, (unsigned long long)pmu_en, (unsigned long long)pmu_run,
         (double)pmu_sum / pmu_units, (double)pmu_empty / 32);
  close(pmu_fd); pmu_fd = -1;
}

__attribute__((constructor(101))) static void pmu_open_before_threads(void)
{
  const char *mode = getenv("BENCH_CYCLES");
  if (!mode) mode = "auto";
  atexit(pmu_report);
  if (!strcmp(mode, "off")) { pmu_status = "disabled"; pmu_reason = "requested_off"; return; }
  pmu_required = !strcmp(mode, "required");
  if (strcmp(mode, "auto") && !pmu_required) {
    fprintf(stderr, "Invalid BENCH_CYCLES mode\n"); exit(2);
  }
  DIR *d = opendir("/proc/self/task");
  if (!d) { pmu_unavailable("cannot_enumerate_threads", errno, 0); return; }
  struct dirent *e; int n = 0;
  while ((e = readdir(d))) if (atoi(e->d_name) > 0) n++;
  closedir(d);
  if (n != 1) { pmu_unavailable("workers_already_exist", 0, 0); return; }
  struct perf_event_attr a = {0};
  a.type = PERF_TYPE_HARDWARE; a.size = sizeof(a); a.config = PERF_COUNT_HW_CPU_CYCLES;
  a.disabled = 1; a.pinned = 1; a.inherit = 1; a.exclude_kernel = 1; a.exclude_hv = 1;
  a.read_format = PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
  pmu_fd = syscall(__NR_perf_event_open, &a, 0, -1, -1, PERF_FLAG_FD_CLOEXEC);
  if (pmu_fd < 0) {
    int error = errno;
    pmu_unavailable(error == EACCES || error == EPERM ? "permission_denied" :
                    error == ENOENT || error == ENODEV || error == EOPNOTSUPP ? "hardware_event_unavailable" :
                    "perf_event_open_failed", error, 0);
    return;
  }
  pmu_status = "available"; pmu_reason = "none";
}

static void pmu_begin(void)
{
  if (pmu_fd < 0) return;
  if (!pmu_ready) {
    DIR *d = opendir("/proc/self/task");
    if (!d) { pmu_unavailable("cannot_enumerate_threads", errno, 1); return; }
    struct dirent *e;
    while ((e = readdir(d))) if (atoi(e->d_name) > 0) pmu_threads++;
    closedir(d);
    pmu_ready = 1;
    for (int i = 0; i < 32 && pmu_fd >= 0; i++) {
      if (!pmu_on()) return;
      pmu_empty += pmu_off(0);
    }
  }
  if (pmu_fd >= 0) pmu_on();
}

static void pmu_end(unsigned units)
{
  if (pmu_fd < 0) return;
  pmu_off(1);
  if (pmu_fd >= 0) { pmu_calls++; pmu_units += units; }
}
#endif
