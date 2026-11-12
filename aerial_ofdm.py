"""Fixed 14-symbol Aerial RX backend for xsai-module-benchmark."""
import cmath
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import tarfile

from bench_support import BenchError, benchmark_lock, write_csv


POINTS = 2048
SYMBOLS = 14
BINS = 106 * 12
CP = (176,) + (144,) * (SYMBOLS - 1)
SLOT_SAMPLES = SYMBOLS * POINTS + sum(CP)
FS_HZ = 61_440_000
AERIAL_COMMIT = "4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c"
NATIVE_SRC = Path("testBenches/gpu3GPPChan/src")
NATIVE_FILES = tuple(map(Path, ("ofdm_src/ofdmDemod.cu", "ofdm_src/ofdmDemod.cuh",
                               "ofdm_src/ofdmCommonFunc.h", "gpu3gppchanCommon.cpp", "gpu3gppchanCommon.h")))
ADAPTER_SRC = Path(__file__).resolve().parent / "src/aerial_ofdm"
RELATIVE_L2_LIMIT = 0.15


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def git_metadata(root):
    def git(*args):
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        if result.returncode:
            raise BenchError(f"Could not read Git metadata in {root}: {result.stderr.strip()}")
        return result.stdout.strip()

    return {"path": str(root), "commit": git("rev-parse", "HEAD"),
            "status": git("status", "--short"), "tracked_diff": git("diff", "HEAD")}


def fft(values):
    """Double-precision radix-2 reference, in natural FFT bin order."""
    result = list(values)
    j = 0
    for i in range(1, POINTS):
        bit = POINTS // 2
        while j & bit:
            j ^= bit
            bit //= 2
        j ^= bit
        if i < j:
            result[i], result[j] = result[j], result[i]
    width = 2
    while width <= POINTS:
        half = width // 2
        weights = [cmath.exp(-2j * math.pi * k / width) for k in range(half)]
        for first in range(0, POINTS, width):
            for k, weight in enumerate(weights):
                a = result[first + k]
                b = result[first + k + half] * weight
                result[first + k], result[first + k + half] = a + b, a - b
        width *= 2
    return result


def prepare(args):
    oai = json.loads(args.oai_manifest.read_text(encoding="utf-8-sig"))
    if oai.get("status", "passed") != "passed" or oai.get("metric", "dft") != "dft":
        raise BenchError("Use verified DFT slot IQ exported by the CPU benchmark")
    raw = args.iq.read_bytes()
    if len(raw) != SYMBOLS * POINTS * 4:
        raise BenchError("Expected exactly 14 × 2048 little-endian int16 IQ pairs")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != oai["input"]["batch14_sha256"]:
        raise BenchError("IQ SHA-256 differs from the archived OAI 14-symbol manifest")
    samples = [complex(i, q) for i, q in struct.iter_unpack("<hh", raw)]
    slot = bytearray()
    reference = bytearray()
    useful_offsets = []
    signed_bins = range(-BINS // 2, BINS // 2)
    for symbol, prefix in enumerate(CP):
        data = samples[symbol * POINTS:(symbol + 1) * POINTS]
        useful_offsets.append(len(slot) // 8 + prefix)
        for value in data[-prefix:] + data:
            slot.extend(struct.pack("<ff", value.real, value.imag))
        spectrum = fft(data)
        for k in signed_bins:
            value = spectrum[k % POINTS] / math.sqrt(POINTS)
            reference.extend(struct.pack("<dd", value.real, value.imag))

    args.out.mkdir(parents=True, exist_ok=False)
    slot_path = args.out / "time_slot_le_f32.bin"
    reference_path = args.out / "freq_reference_le_f64.bin"
    slot_path.write_bytes(slot)
    reference_path.write_bytes(reference)
    save_json(args.out / "fixture.json", {
        "schema": 1, "oai_iq_sha256": digest, "oai_input": oai["input"],
        "oai_manifest_sha256": sha(args.oai_manifest),
        "points": POINTS, "symbols": SYMBOLS, "active_bins": BINS,
        "signed_bin_first": -BINS // 2, "signed_bin_last": BINS // 2 - 1,
        "slot_id_within_subframe": 0, "subcarrier_spacing_hz": 30_000,
        "sample_rate_hz": FS_HZ, "cp_lengths": CP, "useful_offsets": useful_offsets,
        "slot_samples": SLOT_SAMPLES, "normalization": "1/sqrt(2048)",
        "files": {slot_path.name: sha(slot_path), reference_path.name: sha(reference_path)},
    })
    print(args.out / "fixture.json")


def run_logged(command, log):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise BenchError(f"Command exited {result.returncode}. See {log}\n{result.stdout[-2200:]}")
    return result.stdout


def read_native(args):
    if args.aerial_archive:
        archive = args.aerial_archive.resolve()
        with archive.open("rb") as stream:
            result = subprocess.run(["git", "get-tar-commit-id"], stdin=stream, capture_output=True, text=True)
        commit = result.stdout.strip()
        if result.returncode or commit != AERIAL_COMMIT:
            raise BenchError(f"Use an uncompressed git archive from Aerial commit {AERIAL_COMMIT}")
        sources = {}
        with tarfile.open(archive, "r:") as bundle:
            for name in NATIVE_FILES:
                entry = bundle.getmember((NATIVE_SRC / name).as_posix())
                if not entry.isfile():
                    raise BenchError(f"Aerial archive member is not a regular file: {entry.name}")
                with bundle.extractfile(entry) as stream:
                    sources[name] = stream.read()
        return {"archive": str(archive), "archive_sha256": sha(archive), "commit": commit}, sources

    aerial = args.aerial.resolve()
    native = git_metadata(aerial)
    if native["commit"] != AERIAL_COMMIT:
        raise BenchError(f"Use the archived Aerial commit {AERIAL_COMMIT}")
    if subprocess.check_output(["git", "-C", str(aerial), "diff", "HEAD", "--", str(NATIVE_SRC)]):
        raise BenchError("Aerial native gpu3GPPChan sources differ from the pinned commit")
    return native, {name: (aerial / NATIVE_SRC / name).read_bytes() for name in NATIVE_FILES}


def patch_sm121(source):
    # This fixed workload needs only the 2048-point, non-PRACH selector branch.
    opening = "    // current only support cudaDeviceArch = 800, 890, and 900"
    closing = "\n}\n\n" + (
        "template <typename Tscalar, typename Tcomplex> \n"
        "ofdmDeModulate<Tscalar, Tcomplex>::ofdmDeModulate(")
    if source.count(opening) != 1 or source.count(closing) != 1:
        raise BenchError("Could not identify the pinned OFDM receiver's architecture selector")
    branch = (
        "#if defined(AERIAL_BENCH_SM121)\n"
        "    if (cudaDeviceArch != 1210 || Nfft != 2048 || prach)\n"
        '        throw std::runtime_error("SM 121 benchmark requires 2048-point non-PRACH RX");\n'
        "    return ofdmDeMod_get_fft_param<Tscalar, Tcomplex, 2048, 1210>(\n"
        "        block_dim, shared_memory_size, false);\n"
        "#else\n"
    )
    return source.replace(opening, branch + opening).replace(closing, "\n#endif" + closing)


def patch_sm_placement(source):
    include = "#include <cufftdx.hpp>\n"
    kernel = ("static __global__ void ofdmDeMod_fft_kernel("
              "ofdmDeModDynDescr_t<Tscalar, Tcomplex> * ofdmDeModdynDescpr)\n{\n")
    if source.count(include) != 1 or source.count(kernel) != 1:
        raise BenchError("Could not identify the pinned native receive kernel")
    storage = '''
static __device__ uint32_t aerial_bench_sm_ids[7];

extern "C" void aerial_bench_reset_sm(cudaStream_t stream) {
    void* storage = nullptr;
    CHECK_CUDAERROR(cudaGetSymbolAddress(&storage, aerial_bench_sm_ids));
    CHECK_CUDAERROR(cudaMemsetAsync(storage, 0xff, sizeof(aerial_bench_sm_ids), stream));
}

extern "C" void aerial_bench_read_sm(uint32_t* output) {
    CHECK_CUDAERROR(cudaMemcpyFromSymbol(output, aerial_bench_sm_ids, sizeof(aerial_bench_sm_ids)));
}
'''
    probe = '''    if (blockIdx.x == 0 && blockIdx.y < 7 && threadIdx.x == 0 && threadIdx.y == 0) {
        uint32_t sm;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
        aerial_bench_sm_ids[blockIdx.y] = sm;
    }
'''
    return source.replace(include, include + storage).replace(kernel, kernel + probe)


def build(args, root):
    native, sources = read_native(args)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    native_dir = out / "native-src"
    compatibility_patch = None
    for name, contents in sources.items():
        path = native_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.name == "ofdmDemod.cu" and (args.cuda_arch == 121 or args.sm_placement):
            source = contents.decode("utf-8")
            patched = patch_sm121(source) if args.cuda_arch == 121 else source
            if args.sm_placement:
                patched = patch_sm_placement(patched)
            compatibility_patch = out / ("sm-placement.patch" if args.sm_placement else "sm121.patch")
            compatibility_patch.write_text("".join(difflib.unified_diff(
                source.splitlines(keepends=True), patched.splitlines(keepends=True),
                str(NATIVE_SRC / name), str(NATIVE_SRC / name))), encoding="utf-8")
            contents = patched.encode("utf-8")
        path.write_bytes(contents)
    configure = ["cmake", "-S", str(ADAPTER_SRC), "-B", str(out), "-DCMAKE_BUILD_TYPE=Release",
                 "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
                 f"-DAERIAL_CHAN_SRC={native_dir}", f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}-real"]
    if args.mathdx_dir:
        configure.append(f"-Dmathdx_DIR={args.mathdx_dir.resolve()}")
    if args.cuda_compiler:
        configure.append(f"-DCMAKE_CUDA_COMPILER={args.cuda_compiler.resolve()}")
    if args.sm_placement:
        configure.append("-DAERIAL_BENCH_SM_PLACEMENT=ON")
    compile_cmd = ["cmake", "--build", str(out), "--target", "aerial-ofdm-rx", "--parallel", "2", "--verbose"]
    run_logged(configure, out / "configure.log")
    run_logged(compile_cmd, out / "build.log")
    binary = out / "aerial-ofdm-rx"
    source_files = (root / "bench.py", Path(__file__).resolve(), ADAPTER_SRC / "CMakeLists.txt",
                    ADAPTER_SRC / "aerial_ofdm_rx.cpp", *(native_dir / name for name in NATIVE_FILES))
    native["original_source_sha256"] = {str(NATIVE_SRC / name): hashlib.sha256(content).hexdigest()
                                        for name, content in sources.items()}
    save_json(out / "build.json", {"benchmark": git_metadata(root), "aerial": native,
              "architecture": args.cuda_arch, "sm_placement": args.sm_placement, "configure_command": configure,
              "build_command": compile_cmd, "binary_sha256": sha(binary),
              "compatibility_patch_sha256": sha(compatibility_patch) if compatibility_patch else None,
              "source_sha256": {str(path): sha(path) for path in source_files}})
    print(out / "build.json")


def latency_summary(values):
    values = sorted(values)
    if not values or not all(math.isfinite(value) and value > 0 for value in values):
        raise BenchError("CUDA event times must be finite and positive")
    return {"count": len(values), "mean_us": statistics.mean(values), "p50_us": statistics.median(values),
            "p95_us": values[math.ceil(0.95 * len(values)) - 1],
            "p99_us": values[math.ceil(0.99 * len(values)) - 1]}


def capacity_summary(log, args):
    waves = [{"wave": int(wave), "completion_us": float(us)}
             for wave, us in re.findall(r"^\[AERIAL_WAVE\] (\d+) (\S+)$", log, re.M)]
    responses = [{"wave": int(wave), "request": int(request), "response_us": float(us)}
                 for wave, request, us in re.findall(r"^\[AERIAL_REQUEST\] (\d+) (\d+) (\S+)$", log, re.M)]
    if [row["wave"] for row in waves] != list(range(args.samples)):
        raise BenchError("Unexpected number or order of measured waves")
    expected = [(wave, request) for wave in range(args.samples) for request in range(args.requests)]
    if [(row["wave"], row["request"]) for row in responses] != expected:
        raise BenchError("Unexpected number or order of per-request responses")
    steady = re.findall(r"^\[AERIAL_STEADY\] (\d+) (\S+)$", log, re.M)
    if len(steady) != 1 or int(steady[0][0]) != args.samples:
        raise BenchError("Unexpected sustained-throughput measurement")
    duration = float(steady[0][1])
    if not math.isfinite(duration) or duration <= 0:
        raise BenchError("Invalid sustained-throughput interval")
    wave = latency_summary(row["completion_us"] for row in waves)
    response = latency_summary(row["response_us"] for row in responses)
    write_csv(args.out / "results.csv", waves)
    write_csv(args.out / "responses.csv", responses)
    return {"requests_per_wave": args.requests, "warmup": args.warmup,
            "wave": wave, "request_response": response, "sustained_waves": args.samples,
            "sustained_us": duration, "sustained_slots_per_second": args.requests * args.samples * 1e6 / duration}


def placement_summary(log, args):
    rows = []
    observed_sms = set()
    for index, sm_ids in re.findall(r"^\[AERIAL_PLACEMENT\] (\d+) ([\d ]+)$", log, re.M):
        ids = [int(value) for value in sm_ids.split()]
        if len(ids) != 7:
            raise BenchError("Expected a recorded SM ID for each of the seven blocks")
        observed_sms.update(ids)
        rows.append({"launch": int(index), "distinct_sms": len(set(ids)),
                     **{f"block_{i}_sm": value for i, value in enumerate(ids)}})
    if [row["launch"] for row in rows] != list(range(args.samples)):
        raise BenchError("Unexpected number or order of SM placement records")
    write_csv(args.out / "results.csv", rows)
    return {"launches": len(rows), "blocks_per_launch": 7, "observed_sm_ids": sorted(observed_sms),
            "distinct_sms_per_launch": {count: sum(row["distinct_sms"] == count for row in rows)
                                        for count in range(1, 8) if any(row["distinct_sms"] == count for row in rows)}}


def run(args, root):
    with benchmark_lock(wait=args.wait_lock):
        run_locked(args, root)


def run_locked(args, root):
    fixture_dir = args.fixture.resolve()
    fixture = json.loads((fixture_dir / "fixture.json").read_text(encoding="utf-8"))
    slot = fixture_dir / "time_slot_le_f32.bin"
    reference = fixture_dir / "freq_reference_le_f64.bin"
    for path in (slot, reference):
        if sha(path) != fixture["files"][path.name]:
            raise BenchError(f"Fixture hash mismatch for {path.name}")
    build_dir = args.build.resolve()
    binary = build_dir / "aerial-ofdm-rx"
    build_manifest = json.loads((build_dir / "build.json").read_text(encoding="utf-8"))
    if sha(binary) != build_manifest["binary_sha256"]:
        raise BenchError("Driver hash differs from the build manifest")
    if args.warmup < 0 or args.samples < 1 or args.device < 0:
        raise BenchError("Require warmup >= 0, samples >= 1 and device >= 0")
    if args.requests is not None and args.samples > 10000:
        raise BenchError("Capacity runs accept at most 10000 waves")
    if args.requests is not None and build_manifest.get("sm_placement"):
        raise BenchError("Use an uninstrumented driver for capacity measurements")
    args.out.mkdir(parents=True, exist_ok=False)
    output = args.out.resolve() / "freq_output_le_f32.bin"
    command = [str(binary), str(slot), str(reference), str(output), str(args.device),
               str(build_manifest["architecture"]), str(args.warmup), str(args.samples),
               str(RELATIVE_L2_LIMIT)]
    if args.requests is not None:
        command.append(str(args.requests))
    metadata_path = args.out / "metadata.json"
    metadata = {"status": "running", "argv": command,
                "benchmark": git_metadata(root), "build": build_manifest, "fixture": fixture,
                "timing_scope": ("gpu_graph_wave_and_sustained" if args.requests is not None
                                 else "sm_placement_only" if build_manifest.get("sm_placement")
                                 else "gpu_native_receive_event"),
                "environment": {key: os.environ.get(key) for key in
                                ("CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "LD_PRELOAD")},
                "relative_l2_limit": RELATIVE_L2_LIMIT}
    save_json(metadata_path, metadata)
    try:
        log = run_logged(command, args.out / "run.log")
        check = re.search(r"^\[AERIAL_CHECK\] worst_symbol_relative_l2=(\S+)$", log, re.M)
        if check is None or not 0 <= float(check[1]) <= RELATIVE_L2_LIMIT:
            raise BenchError("Aerial output did not pass the reference check")
        if output.stat().st_size != SYMBOLS * BINS * 8:
            raise BenchError("Unexpected size for the captured Aerial output")
        if build_manifest.get("sm_placement"):
            summary = placement_summary(log, args)
        elif args.requests is not None:
            summary = capacity_summary(log, args)
        else:
            rows = [{"slot_index": int(index), "slot_us": float(us), "amortized_symbol_us": float(us) / SYMBOLS}
                    for index, us in re.findall(r"^\[AERIAL_SAMPLE\] (\d+) (\S+)$", log, re.M)]
            if [row["slot_index"] for row in rows] != list(range(args.samples)):
                raise BenchError("Unexpected number or order of CUDA event samples")
            latency = latency_summary(row["slot_us"] for row in rows)
            write_csv(args.out / "results.csv", rows)
            mean = latency["mean_us"]
            summary = {"slot_samples": args.samples, "warmup": args.warmup,
                       "slot_mean_us": mean, "slot_p50_us": latency["p50_us"], "slot_p95_us": latency["p95_us"],
                       "amortized_symbol_mean_us": mean / SYMBOLS, "transforms_per_second": SYMBOLS * 1e6 / mean}
        summary.update(worst_symbol_relative_l2=float(check[1]), output_sha256=sha(output))
        save_json(args.out / "summary.json", summary)
        metadata["status"] = "passed"
        print(json.dumps(summary, indent=2))
    except BaseException as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        save_json(metadata_path, metadata)


def add_parser(subparsers):
    parser = subparsers.add_parser("aerial-ofdm", help="Fixed 14 × 2048 native Aerial OFDM receive benchmark")
    actions = parser.add_subparsers(dest="aerial_action", required=True)
    prep = actions.add_parser("prepare", help="Add normal CP and create the independent reference")
    prep.add_argument("--iq", type=Path, required=True)
    prep.add_argument("--oai-manifest", type=Path, required=True)
    builder = actions.add_parser("build", help="Compile only the native Aerial RX driver")
    source = builder.add_mutually_exclusive_group(required=True)
    source.add_argument("--aerial", type=Path)
    source.add_argument("--aerial-archive", type=Path, help="Uncompressed git archive from the pinned Aerial commit")
    builder.add_argument("--cuda-arch", type=int, choices=(80, 89, 90, 121), required=True)
    builder.add_argument("--cuda-compiler", type=Path)
    builder.add_argument("--mathdx-dir", type=Path)
    builder.add_argument("--sm-placement", action="store_true", help="Build a diagnostic driver that records the SM for each block")
    runner = actions.add_parser("run", help="Verify output, then time each RX slot on the device")
    runner.add_argument("--fixture", type=Path, required=True)
    runner.add_argument("--build", type=Path, required=True)
    runner.add_argument("--device", type=int, default=0)
    runner.add_argument("--warmup", type=int, default=20)
    runner.add_argument("--samples", type=int, default=200)
    runner.add_argument("--requests", type=int, choices=(1, 2, 4, 8, 16, 32, 64),
                        help="Use a CUDA graph with this many independent slots per wave")
    for action in (builder, runner):
        action.add_argument("--wait-lock", action="store_true")
    for action in (prep, builder, runner):
        action.add_argument("--out", type=Path, required=True, help="New output directory")


def dispatch(args, root):
    if args.aerial_action == "prepare":
        prepare(args)
    elif args.aerial_action == "build":
        with benchmark_lock(wait=args.wait_lock):
            build(args, root)
    else:
        run(args, root)
