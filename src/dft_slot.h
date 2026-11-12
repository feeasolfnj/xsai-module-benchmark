/* The original module handles one symbol. This driver distributes independent symbols. */
#include <pthread.h>
#include <sched.h>

enum { slot_i16_per_symbol = 4096, slot_symbol_bytes = 8192 };

struct slot_work {
    int16_t *input;
    int16_t *output;
    int first;
    int end;
    int stop;
    pthread_barrier_t *start;
    pthread_barrier_t *finish;
};

static void slot_compute(const struct slot_work *work)
{
    for (int symbol = work->first; symbol < work->end; ++symbol) {
        size_t offset = (size_t)symbol * slot_i16_per_symbol;
        selected_module.run(work->input + offset, work->output + offset, 1);
    }
}

static void *slot_worker(void *argument)
{
    struct slot_work *work = argument;
    for (;;) {
        pthread_barrier_wait(work->start);
        if (work->stop) return NULL;
        slot_compute(work);
        pthread_barrier_wait(work->finish);
    }
}

static void slot_check_pthread(int result, const char *operation)
{
    if (result) {
        fprintf(stderr, "%s: %s\n", operation, strerror(result));
        exit(2);
    }
}

static void slot_affinity(cpu_set_t *set, const char *name)
{
    int cpu = option(name, -1);
    if (cpu < 0 || cpu >= CPU_SETSIZE) {
        fprintf(stderr, "Invalid CPU in %s\n", name);
        exit(2);
    }
    CPU_ZERO(set);
    CPU_SET(cpu, set);
}

static int run_dft_slot(int16_t *input, int symbols, int workers, int warmup, int slots, int inverse)
{
    if ((workers != 1 && workers != 2) || (workers == 2 && symbols < 2)) return 2;
    size_t bytes = (size_t)symbols * slot_symbol_bytes;
    const char *export_path = getenv("BENCH_SLOT_IQ");
    if (export_path) {
        FILE *file = fopen(export_path, "wb");
        if (!file) {
            perror(export_path);
            return 2;
        }
        size_t written = fwrite(input, 1, bytes, file);
        if (fclose(file) || written != bytes) {
            fprintf(stderr, "Could not export all slot IQ samples\n");
            return 2;
        }
    }
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) {
        perror("/proc/self/maps");
        return 2;
    }
    char line[8192];
    const char *expected_library = getenv("BENCH_SLOT_LIBRARY");
    while (fgets(line, sizeof(line), maps)) {
        char *path = strchr(line, '/');
        if (path && (strstr(path, "/libdfts.so") || (expected_library && strstr(path, expected_library)))) {
            printf("[DFT_SLOT_LIBRARY] %s", path);
        }
    }
    fclose(maps);

    int16_t *output, *reference;
    slot_check_pthread(posix_memalign((void **)&output, 64, bytes), "allocate slot output");
    slot_check_pthread(posix_memalign((void **)&reference, 64, bytes), "allocate slot reference");
    cpu_set_t affinity;
    slot_affinity(&affinity, workers == 2 ? "BENCH_SLOT_COORD_CPU" : "BENCH_SLOT_CPU0");
    slot_check_pthread(pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity), "bind slot caller");
    for (int symbol = 0; symbol < symbols; ++symbol) {
        size_t offset = (size_t)symbol * slot_i16_per_symbol;
        reference_module.run(input + offset, reference + offset, 1);
    }

    pthread_barrier_t start, finish;
    pthread_t threads[2];
    int split = workers == 2 ? (symbols + 1) / 2 : symbols;
    struct slot_work jobs[2] = {
        {input, output, 0, split, 0, &start, &finish},
        {input, output, split, symbols, 0, &start, &finish},
    };
    if (workers == 2) {
        slot_check_pthread(pthread_barrier_init(&start, NULL, 3), "init start barrier");
        slot_check_pthread(pthread_barrier_init(&finish, NULL, 3), "init finish barrier");
        pthread_attr_t attr;
        slot_check_pthread(pthread_attr_init(&attr), "init worker attributes");
        for (int i = 0; i < 2; ++i) {
            slot_affinity(&affinity, i ? "BENCH_SLOT_CPU1" : "BENCH_SLOT_CPU0");
            slot_check_pthread(pthread_attr_setaffinity_np(&attr, sizeof(affinity), &affinity), "bind slot worker");
            slot_check_pthread(pthread_create(&threads[i], &attr, slot_worker, &jobs[i]), "start slot worker");
        }
        pthread_attr_destroy(&attr);
    }

    int status = 0;
    double *samples = calloc((size_t)warmup + slots, sizeof(*samples));
    if (!samples) exit(2);
    for (int trial = 0; trial < warmup + slots; ++trial) {
        memset(output, 0x55, bytes);
        double begin = tm(CLOCK_MONOTONIC_RAW);
        if (workers == 1) slot_compute(&jobs[0]);
        else {
            pthread_barrier_wait(&start);
            pthread_barrier_wait(&finish);
        }
        samples[trial] = (tm(CLOCK_MONOTONIC_RAW) - begin) * 1e6;
        if (memcmp(output, reference, bytes)) {
            fprintf(stderr, "Slot %d differs from the serial reference\n", trial);
            status = 2;
            break;
        }
    }
    if (workers == 2) {
        jobs[0].stop = jobs[1].stop = 1;
        pthread_barrier_wait(&start);
        for (int i = 0; i < 2; ++i) pthread_join(threads[i], NULL);
        pthread_barrier_destroy(&start);
        pthread_barrier_destroy(&finish);
    }
    if (!status) {
        for (int trial = 0; trial < warmup + slots; ++trial) {
            printf("[DFT_SLOT] trial=%d wall_us=%.9f transform=%s points=2048 symbols=%d workers=%d "
                   "verification=passed verification_scope=serial_same_implementation\n",
                   trial, samples[trial], inverse ? "idft" : "dft", symbols, workers);
        }
    }
    free(samples);
    free(output);
    free(reference);
    return status;
}
