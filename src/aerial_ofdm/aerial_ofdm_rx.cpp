#include "ofdmDemod.cuh"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef AERIAL_BENCH_SM_PLACEMENT
extern "C" void aerial_bench_reset_sm(cudaStream_t stream);
extern "C" void aerial_bench_read_sm(uint32_t* output);
#endif

namespace {
constexpr int points = 2048;
constexpr int symbols = 14;
constexpr int bins = 1272;
constexpr int slot_samples = 30720;
constexpr int output_count = symbols * bins;

struct CudaResources {
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cuComplex* input = nullptr;
    cuComplex* output = nullptr;

    ~CudaResources() {
        if (stream) cudaStreamSynchronize(stream);
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (output) cudaFree(output);
        if (input) cudaFree(input);
        if (stream) cudaStreamDestroy(stream);
    }
};

struct CudaGraphResources {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;

    ~CudaGraphResources() {
        if (executable) cudaGraphExecDestroy(executable);
        if (graph) cudaGraphDestroy(graph);
    }
};

template<typename T>
std::vector<T> read_file(const char* path, std::size_t count) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    const auto bytes = static_cast<std::streamsize>(count * sizeof(T));
    if (!file || file.tellg() != bytes) {
        throw std::runtime_error(std::string("Unexpected file size: ") + path);
    }
    std::vector<T> values(count);
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(values.data()), bytes)) {
        throw std::runtime_error(std::string("Could not read: ") + path);
    }
    return values;
}

double check_output(const std::vector<cuComplex>& actual, const std::vector<double>& reference) {
    double worst = 0;
    for (int symbol = 0; symbol < symbols; ++symbol) {
        double error = 0;
        double energy = 0;
        for (int k = 0; k < bins; ++k) {
            const int i = symbol * bins + k;
            const double re = reference[2 * i];
            const double im = reference[2 * i + 1];
            const double diff_re = actual[i].x - re;
            const double diff_im = actual[i].y - im;
            error += diff_re * diff_re + diff_im * diff_im;
            energy += re * re + im * im;
        }
        if (!std::isfinite(error) || !std::isfinite(energy) || energy <= 0) {
            throw std::runtime_error("Non-finite output/reference or zero reference energy at symbol "
                                     + std::to_string(symbol));
        }
        worst = std::max(worst, std::sqrt(error / energy));
    }
    return worst;
}

void write_output(const char* path, const std::vector<cuComplex>& output) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(cuComplex));
    out.close();
    if (!out) throw std::runtime_error("Could not write device output");
}

cuphyCarrierPrms_t carrier_parameters() {
    cuphyCarrierPrms_t carrier{};
    carrier.N_FFT = points;
    carrier.N_sc = bins;
    carrier.N_symbol_slot = symbols;
    carrier.N_bsLayer = carrier.N_ueLayer = 1;
    carrier.id_slot = carrier.id_subFrame = 0;
    carrier.cpType = 0;
    carrier.mu = 1;
    carrier.f_samp = 61440000;
    carrier.N_samp_slot = slot_samples;
    carrier.ofdmWindowLen = 0;
    return carrier;
}

using NativeRx = ofdm_demodulate::ofdmDeModulate<float, cuComplex>;

struct SlotRequest {
    CudaResources cuda;
    std::unique_ptr<NativeRx> rx;

    SlotRequest(cuphyCarrierPrms_t& carrier, const std::vector<cuComplex>& input) {
        CHECK_CUDAERROR(cudaStreamCreateWithFlags(&cuda.stream, cudaStreamNonBlocking));
        CHECK_CUDAERROR(cudaEventCreate(&cuda.stop));
        CHECK_CUDAERROR(cudaMalloc(&cuda.input, input.size() * sizeof(cuComplex)));
        CHECK_CUDAERROR(cudaMalloc(&cuda.output, output_count * sizeof(cuComplex)));
        CHECK_CUDAERROR(cudaMemcpy(cuda.input, input.data(), input.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));
        rx = std::make_unique<NativeRx>(&carrier, cuda.input, cuda.output, false, false, cuda.stream);
        CHECK_CUDAERROR(cudaStreamSynchronize(cuda.stream));
    }

    cudaGraphNode_t add_to(cudaGraph_t parent, cudaGraphNode_t start, cudaGraph_t throughput) {
        CudaGraphResources child;
        CHECK_CUDAERROR(cudaStreamBeginCapture(cuda.stream, cudaStreamCaptureModeThreadLocal));
        try {
            rx->run(1, cuda.stream);
        } catch (...) {
            cudaStreamEndCapture(cuda.stream, &child.graph);
            throw;
        }
        CHECK_CUDAERROR(cudaStreamEndCapture(cuda.stream, &child.graph));
        std::size_t nodes = 0;
        CHECK_CUDAERROR(cudaGraphGetNodes(child.graph, nullptr, &nodes));
        if (nodes != 1) throw std::runtime_error("Expected exactly one captured Aerial kernel per request");
        cudaGraphNode_t kernel = nullptr, done = nullptr, steady = nullptr;
        CHECK_CUDAERROR(cudaGraphAddChildGraphNode(&kernel, parent, &start, 1, child.graph));
        CHECK_CUDAERROR(cudaGraphAddEventRecordNode(&done, parent, &kernel, 1, cuda.stop));
        CHECK_CUDAERROR(cudaGraphAddChildGraphNode(&steady, throughput, nullptr, 0, child.graph));
        return done;
    }
};

int run_capacity(const std::vector<cuComplex>& input, const std::vector<double>& reference,
                 const char* output_path, int requests, int warmup, int waves, double limit) {
    auto carrier = carrier_parameters();
    std::vector<std::unique_ptr<SlotRequest>> slots;
    for (int i = 0; i < requests; ++i) slots.push_back(std::make_unique<SlotRequest>(carrier, input));
    CudaResources timing;
    CudaGraphResources batch;
    CudaGraphResources throughput;
    CHECK_CUDAERROR(cudaStreamCreateWithFlags(&timing.stream, cudaStreamNonBlocking));
    CHECK_CUDAERROR(cudaEventCreate(&timing.start));
    CHECK_CUDAERROR(cudaEventCreate(&timing.stop));
    CHECK_CUDAERROR(cudaGraphCreate(&batch.graph, 0));
    CHECK_CUDAERROR(cudaGraphCreate(&throughput.graph, 0));
    cudaGraphNode_t start = nullptr, stop = nullptr;
    CHECK_CUDAERROR(cudaGraphAddEventRecordNode(&start, batch.graph, nullptr, 0, timing.start));
    std::vector<cudaGraphNode_t> finishes;
    for (const auto& slot : slots) finishes.push_back(slot->add_to(batch.graph, start, throughput.graph));
    CHECK_CUDAERROR(cudaGraphAddEventRecordNode(&stop, batch.graph, finishes.data(), finishes.size(), timing.stop));
    CHECK_CUDAERROR(cudaGraphInstantiate(&batch.executable, batch.graph, 0));
    CHECK_CUDAERROR(cudaGraphInstantiate(&throughput.executable, throughput.graph, 0));

    const auto launch = [&] {
        CHECK_CUDAERROR(cudaGraphLaunch(batch.executable, timing.stream));
        CHECK_CUDAERROR(cudaStreamSynchronize(timing.stream));
    };
    std::vector<cuComplex> output(output_count);
    const auto verify = [&] {
        double worst = 0;
        for (const auto& slot : slots) {
            CHECK_CUDAERROR(cudaMemcpy(output.data(), slot->rx->getFreqDataOut(),
                                      output.size() * sizeof(cuComplex), cudaMemcpyDeviceToHost));
            worst = std::max(worst, check_output(output, reference));
        }
        if (worst > limit) throw std::runtime_error("Capacity output exceeds the relative L2 limit");
        return worst;
    };

    const auto clear_outputs = [&] {
        for (const auto& slot : slots) {
            CHECK_CUDAERROR(cudaMemsetAsync(slot->cuda.output, 0xff, output_count * sizeof(cuComplex), slot->cuda.stream));
            CHECK_CUDAERROR(cudaStreamSynchronize(slot->cuda.stream));
        }
    };
    clear_outputs();
    launch();
    const double precheck = verify();
    for (int i = 0; i < warmup; ++i) launch();

    std::vector<float> wave_ms(waves);
    std::vector<float> response_ms(static_cast<std::size_t>(waves) * requests);
    for (int wave = 0; wave < waves; ++wave) {
        launch();
        CHECK_CUDAERROR(cudaEventElapsedTime(&wave_ms[wave], timing.start, timing.stop));
        for (int request = 0; request < requests; ++request) {
            CHECK_CUDAERROR(cudaEventElapsedTime(&response_ms[wave * requests + request],
                                                timing.start, slots[request]->cuda.stop));
        }
    }
    clear_outputs();
    for (int i = 0; i < std::max(1, warmup); ++i) {
        CHECK_CUDAERROR(cudaGraphLaunch(throughput.executable, timing.stream));
    }
    CHECK_CUDAERROR(cudaStreamSynchronize(timing.stream));
    const double steady_check = verify();
    // A separate train of event-free graphs avoids charging per-request markers to throughput.
    CHECK_CUDAERROR(cudaEventRecord(timing.start, timing.stream));
    for (int wave = 0; wave < waves; ++wave) {
        CHECK_CUDAERROR(cudaGraphLaunch(throughput.executable, timing.stream));
    }
    CHECK_CUDAERROR(cudaEventRecord(timing.stop, timing.stream));
    CHECK_CUDAERROR(cudaEventSynchronize(timing.stop));
    float steady_ms = 0;
    CHECK_CUDAERROR(cudaEventElapsedTime(&steady_ms, timing.start, timing.stop));
    const double postcheck = verify();
    write_output(output_path, output);
    std::cout << std::setprecision(10) << "[AERIAL_CHECK] worst_symbol_relative_l2="
              << std::max({precheck, steady_check, postcheck}) << '\n';
    std::cout << "[AERIAL_STEADY] " << waves << ' ' << steady_ms * 1000.0 << '\n';
    for (int wave = 0; wave < waves; ++wave) {
        std::cout << "[AERIAL_WAVE] " << wave << ' ' << wave_ms[wave] * 1000.0 << '\n';
        for (int request = 0; request < requests; ++request) {
            std::cout << "[AERIAL_REQUEST] " << wave << ' ' << request << ' '
                      << response_ms[wave * requests + request] * 1000.0 << '\n';
        }
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 9 && argc != 10) {
            throw std::runtime_error("usage: aerial-ofdm-rx SLOT REFERENCE OUTPUT DEVICE SM WARMUP SAMPLES L2_LIMIT [REQUESTS]");
        }
        static_assert(std::endian::native == std::endian::little);
        static_assert(sizeof(cuComplex) == 2 * sizeof(float));
        static_assert(sizeof(float) == 4 && sizeof(double) == 8);
        static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);
        const auto input = read_file<cuComplex>(argv[1], slot_samples);
        const auto reference = read_file<double>(argv[2], 2 * output_count);
        const int device = std::stoi(argv[4]);
        const int built_sm = std::stoi(argv[5]);
        const int warmup = std::stoi(argv[6]);
        const int samples = std::stoi(argv[7]);
        const double limit = std::stod(argv[8]);
        const int requests = argc == 10 ? std::stoi(argv[9]) : 0;
        if (device < 0 || warmup < 0 || samples <= 0 || !std::isfinite(limit) || limit <= 0 ||
            (argc == 10 && (requests <= 0 || requests > 64 || samples > 10000))) {
            throw std::runtime_error("Invalid device, warmup, samples or L2 limit");
        }
#ifdef AERIAL_BENCH_SM_PLACEMENT
        if (requests) throw std::runtime_error("Use an uninstrumented driver for capacity measurements");
#endif

        CHECK_CUDAERROR(cudaSetDevice(device));
        cudaDeviceProp props{};
        CHECK_CUDAERROR(cudaGetDeviceProperties(&props, device));
        const int actual_sm = props.major * 10 + props.minor;
        if (actual_sm != built_sm || (actual_sm != 80 && actual_sm != 89 && actual_sm != 90 && actual_sm != 121)) {
            throw std::runtime_error("Device SM " + std::to_string(actual_sm)
                                     + " does not match a supported build SM " + std::to_string(built_sm));
        }
        int driver_version = 0, runtime_version = 0;
        CHECK_CUDAERROR(cudaDriverGetVersion(&driver_version));
        CHECK_CUDAERROR(cudaRuntimeGetVersion(&runtime_version));
        std::cout << "[AERIAL_DEVICE] device=" << device << " name=" << std::quoted(props.name)
                  << " sm_arch=" << actual_sm << " available_sms=" << props.multiProcessorCount
                  << " driver=" << driver_version << " runtime=" << runtime_version << '\n';

        if (requests) return run_capacity(input, reference, argv[3], requests, warmup, samples, limit);

        CudaResources cuda;
        CHECK_CUDAERROR(cudaStreamCreateWithFlags(&cuda.stream, cudaStreamNonBlocking));
        CHECK_CUDAERROR(cudaEventCreate(&cuda.start));
        CHECK_CUDAERROR(cudaEventCreate(&cuda.stop));
        CHECK_CUDAERROR(cudaMalloc(&cuda.input, input.size() * sizeof(cuComplex)));
        CHECK_CUDAERROR(cudaMalloc(&cuda.output, output_count * sizeof(cuComplex)));
        CHECK_CUDAERROR(cudaMemcpy(cuda.input, input.data(), input.size() * sizeof(cuComplex), cudaMemcpyHostToDevice));
        auto carrier = carrier_parameters();
        std::vector<cuComplex> output(output_count);
        std::vector<float> times(samples);

        // The native constructor copies the CP table and launches an initial RX.
        ofdm_demodulate::ofdmDeModulate<float, cuComplex> ofdmDeMod(
            &carrier, cuda.input, cuda.output, false, false, cuda.stream);
        // Unwritten FP32 components remain NaN and fail validation.
        CHECK_CUDAERROR(cudaMemsetAsync(cuda.output, 0xff, output_count * sizeof(cuComplex), cuda.stream));
        ofdmDeMod.run(1, cuda.stream);  // Uplink uses the BS receive layer count.
        CHECK_CUDAERROR(cudaStreamSynchronize(cuda.stream));
        const auto verify = [&] {
            CHECK_CUDAERROR(cudaMemcpy(output.data(), ofdmDeMod.getFreqDataOut(),
                                      output.size() * sizeof(cuComplex), cudaMemcpyDeviceToHost));
            const double error = check_output(output, reference);
            if (error > limit) {
                throw std::runtime_error("Worst symbol relative L2 error " + std::to_string(error)
                                         + " exceeds limit " + std::to_string(limit));
            }
            return error;
        };
        const double precheck = verify();

        for (int i = 0; i < warmup; ++i) {
            ofdmDeMod.run(1, cuda.stream);
        }
        CHECK_CUDAERROR(cudaStreamSynchronize(cuda.stream));
#ifdef AERIAL_BENCH_SM_PLACEMENT
        for (int launch = 0; launch < samples; ++launch) {
            std::array<uint32_t, 7> sm_ids{};
            aerial_bench_reset_sm(cuda.stream);
            ofdmDeMod.run(1, cuda.stream);
            CHECK_CUDAERROR(cudaStreamSynchronize(cuda.stream));
            aerial_bench_read_sm(sm_ids.data());
            if (std::find(sm_ids.begin(), sm_ids.end(), UINT32_MAX) != sm_ids.end()) {
                throw std::runtime_error("Not all seven receive blocks recorded an SM ID");
            }
            std::cout << "[AERIAL_PLACEMENT] " << launch;
            for (uint32_t sm : sm_ids) std::cout << ' ' << sm;
            std::cout << '\n';
        }
#else
        for (float& time_ms : times) {
            CHECK_CUDAERROR(cudaEventRecord(cuda.start, cuda.stream));
            ofdmDeMod.run(1, cuda.stream);
            CHECK_CUDAERROR(cudaEventRecord(cuda.stop, cuda.stream));
            CHECK_CUDAERROR(cudaEventSynchronize(cuda.stop));
            CHECK_CUDAERROR(cudaEventElapsedTime(&time_ms, cuda.start, cuda.stop));
        }
#endif
        const double postcheck = verify();
        write_output(argv[3], output);
        std::cout << std::setprecision(10) << "[AERIAL_CHECK] worst_symbol_relative_l2="
                  << std::max(precheck, postcheck) << '\n';
#ifndef AERIAL_BENCH_SM_PLACEMENT
        for (std::size_t i = 0; i < times.size(); ++i) {
            std::cout << "[AERIAL_SAMPLE] " << i << ' ' << times[i] * 1000.0 << '\n';
        }
#endif
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Aerial OFDM RX: " << error.what() << '\n';
        return 1;
    }
}
