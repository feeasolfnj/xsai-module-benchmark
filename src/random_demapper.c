/* Performance harness for the official SRK demapper_compute_llr plugin ABI.
 * This is not the upstream pytest accuracy test; see DEMAPPER.md. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "plugins/neural_demapper/src/nr_demapper_defs.h"
#include "pmu_cycles.h"
#include <dlfcn.h>
#include <limits.h>
#include <time.h>

/* Some OAI plugins resolve their fatal-error hook from the main executable. */
void exit_function(const char *file, const char *func, int line, const char *message, int code)
{
  fprintf(stderr, "%s:%d %s: %s (%d)\n", file, line, func, message, code);
  exit(2);
}
static void fail(const char *message) { fprintf(stderr, "DEMAPPER ERROR: %s\n", message); exit(2); }
static double now(clockid_t id)
{
  struct timespec t;
  if (clock_gettime(id, &t)) fail("clock_gettime");
  return t.tv_sec + t.tv_nsec * 1e-9;
}
static uint32_t rng(uint32_t *s)
{
  uint32_t x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return *s = x;
}
static uint64_t digest(uint64_t h, const void *data, size_t bytes)
{
  const unsigned char *p = data;
  while (bytes--) h = (h ^ *p++) * UINT64_C(1099511628211);
  return h;
}
static void *allocate(size_t bytes)
{
  void *p;
  if (posix_memalign(&p, 64, bytes)) fail("allocation failed");
  memset(p, 0, bytes); return p;
}
typedef struct { int32_t *rx; c16_t *mag; int16_t *storage, *out, *reference; } input_t;
static void check_guards(input_t *in, unsigned re)
{
  for (int i = 0; i < 32; i++)
    if (in->storage[i] != 12345 || in->out[4 * re + i] != 12345) fail("output buffer guard corrupted");
}
static void compute(demapper_compute_llrfunc_t *fn, input_t *in, unsigned re)
{
  if (fn(in->rx, in->mag, in->mag, in->mag, in->out, re, 0, 4) != 1)
    fail("plugin did not handle 16-QAM (fallback is not a neural benchmark)");
}
int main(int argc, char **argv)
{
  if (argc != 4) fail("expected: seed seconds RE");
  uint32_t seed = strtoul(argv[1], NULL, 0), state = seed ? seed : 1;
  double seconds = strtod(argv[2], NULL);
  unsigned re = strtoul(argv[3], NULL, 0);
  int bank = getenv("BENCH_INPUT_BANK") ? atoi(getenv("BENCH_INPUT_BANK")) : 1;
  int warmup = getenv("BENCH_WARMUP") ? atoi(getenv("BENCH_WARMUP")) : 100;
  const char *library = getenv("BENCH_DEMAPPER_LIB");
  if (!library || seconds <= 0 || re < 1 || re > 32768 || bank < 1 || bank > 16 || warmup < 0)
    fail("invalid parameters");
  void *handle = dlopen(library, RTLD_NOW | RTLD_LOCAL);
  if (!handle) fail(dlerror());
  demapper_initfunc_t *init = (demapper_initfunc_t *)dlsym(handle, "demapper_init");
  demapper_initfunc_t *thread_init = (demapper_initfunc_t *)dlsym(handle, "demapper_init_thread");
  demapper_shutdownfunc_t *shutdown = (demapper_shutdownfunc_t *)dlsym(handle, "demapper_shutdown");
  demapper_compute_llrfunc_t *fn = (demapper_compute_llrfunc_t *)dlsym(handle, "demapper_compute_llr");
  if (!init || !shutdown || !fn) fail("missing required demapper plugin symbols");
  if (init() != 0 || (thread_init && thread_init() != 0)) fail("plugin initialization failed");
  input_t in[16] = {0};
  uint64_t input_hash = UINT64_C(1469598103934665603), output_hash = input_hash;
  for (int k = 0; k < bank; k++) {
    /* Padding permits local SIMD kernels to read a rounded-up final vector. */
    in[k].rx = allocate((re + 32) * sizeof(int32_t));
    in[k].mag = allocate((re + 32) * sizeof(c16_t));
    in[k].storage = allocate((4 * re + 64) * sizeof(int16_t));
    in[k].out = in[k].storage + 32;
    in[k].reference = allocate(4 * re * sizeof(int16_t));
    for (unsigned i = 0; i < 2 * re; i++) {
      ((int16_t *)in[k].rx)[i] = (int16_t)((int)(rng(&state) & 4095) - 2048);
      ((int16_t *)in[k].mag)[i] = (int16_t)(1 + (rng(&state) & 2047));
    }
    for (int i = 0; i < 32; i++) in[k].storage[i] = in[k].out[4 * re + i] = 12345;
    input_hash = digest(input_hash, in[k].rx, re * sizeof(int32_t));
    input_hash = digest(input_hash, in[k].mag, re * sizeof(c16_t));
    compute(fn, &in[k], re); check_guards(&in[k], re);
    int nonzero = 0;
    for (unsigned i = 0; i < 4 * re; i++) nonzero |= in[k].out[i] != 0;
    if (!nonzero) fail("all-zero output; check model and plugin");
    memcpy(in[k].reference, in[k].out, 4 * re * sizeof(int16_t));
  }
  for (int i = 0; i < warmup; i++) compute(fn, &in[i % bank], re);
  printf("[RANDOM_INPUT] seed=%u source=xorshift32 iq_min=-2048 iq_max=2047 mag_min=1 mag_max=2048 re=%u qm=4 input_bank=%d input_hash=%llu file_reads=0\n",
         seed, re, bank, (unsigned long long)input_hash);
  unsigned calls = 0;
  double elapsed = 0, begin = now(CLOCK_MONOTONIC_RAW), cpu0 = now(CLOCK_PROCESS_CPUTIME_ID);
  do {
    pmu_begin(); double start = now(CLOCK_MONOTONIC_RAW);
    for (int i = 0; i < 32; i++) compute(fn, &in[(calls + i) % bank], re);
    elapsed += now(CLOCK_MONOTONIC_RAW) - start; pmu_end(32); calls += 32;
  } while (now(CLOCK_MONOTONIC_RAW) - begin < seconds);
  double window = now(CLOCK_MONOTONIC_RAW) - begin, cpu = now(CLOCK_PROCESS_CPUTIME_ID) - cpu0;
  for (int k = 0; k < bank; k++) {
    check_guards(&in[k], re);
    if (memcmp(in[k].out, in[k].reference, 4 * re * sizeof(int16_t))) fail("output changed for identical input");
    output_hash = digest(output_hash, in[k].out, 4 * re * sizeof(int16_t));
  }
  if (shutdown() != 0) fail("plugin shutdown failed");
  printf("[RESULT] calls=%u mean_us=%.9f cores=%.9f re=%u qm=4 handled=1 verification=passed verification_scope=repeatability_same_implementation input_bank=%d input_hash=%llu output_hash=%llu\n",
         calls, elapsed * 1e6 / calls, cpu / window, re, bank,
         (unsigned long long)input_hash, (unsigned long long)output_hash);
  for (int k = 0; k < bank; k++) {
    free(in[k].rx); free(in[k].mag); free(in[k].storage); free(in[k].reference);
  }
  dlclose(handle); return 0;
}
