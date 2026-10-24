# Validation record

## 2026-10-24 usability revision

Validated from a **fresh GitHub clone** of the usability revision on K3,
not by copying the previous benchmark build directory. The clone rebuilt all
eight executables / nine metrics against the existing local XSAI working tree
at Git baseline `18276ae0b271dea9b68420eecc58dd8b45e28787`. Missing MAC archive
objects were rebuilt inside the new benchmark directory. XSAI sources and
libraries were not overwritten. This checks relocation and rebuilding, not
compatibility with an unrelated third-party XSAI revision.

- All nine metrics completed both single-thread and eight-thread configurations:
  estimation, measurement, symbol, llr, UL, DL, neural, dft, idft.
- The matrix used two pregenerated inputs for PUSCH/LLR/neural/DFT. UL/DL used
  two warmup calls and three measured calls. The runner verified exactly three
  formal codec records, with no warmup records in results.json.
- Sparse CPUs `0,2,4,6` completed symbol processing. Unavailable CPU, unsupported
  duration option, and unsupported frequency each produced the expected friendly
  failure. Total: **22 on-device checks passed**.
- **16 regression tests passed on K3**, including the previous-sudo read-only
  lock case, warmup filtering, CPU validation, frequency preflight/restoration,
  synthetic DFT mismatch handling, and compile_commands.json flag extraction. The actual full build used the existing Makefiles tree; a full third-party Ninja build was not independently repeated. In the synthetic host C harness, strict
  rejects a mismatch, report preserves `verification=failed`, and skip reports
  `verification=skipped`.
- GitHub Actions passed on Ubuntu and Windows with Python 3.10 and 3.12.
  Windows intentionally skips POSIX C-harness and Linux flock tests; Linux runs
  the complete suite. These are launcher/protocol checks, not hardware tests.

The on-device matrix used the default non-PMU build and **did not lock frequency**;
these short runs validate usability rather than establish replacement latency
numbers for the earlier 2.2 GHz benchmark table. Neural tests used separately
supplied compatible X100 single/pool8 plugins and weights. A100 and optional PMU
were not retested in this revision.

`status=passed` means the measurement process completed. Verification scope is
reported separately; repeatability or serial equivalence is not an independent
mathematical or communication-quality proof. Default formal CSV/JSON now excludes
codec warmup; raw records retain it. See [USAGE.md](USAGE.md) for current semantics.

## Historical record: 2026-10-23

2026-10-23, K3 Linux/RISC-V RVV, GCC16, X100 2.2GHz.

The repository was copied to a new directory and built against an existing local XSAI checkout. Original personal result-directory dependencies were removed: the build now reconstructs missing helper objects and the small MAC archive from the supplied checkout into its own build directory. No original XSAI files were overwritten.

Validated `--pmu` disabled (the default):

| Metric | 1 thread configuration | 8 thread configuration |
|---|---|---|
| estimation | PASS | PASS |
| measurement | PASS | PASS |
| symbol | PASS | PASS |
| llr | PASS | PASS |
| UL | PASS | PASS |
| DL | PASS | PASS |
| neural | PASS | PASS |
| dft | PASS | PASS |
| idft | PASS | PASS |

Neural runs used separately supplied compatible single/pool8 libraries and model weights. They are not in this repository. The neural results do not establish compatibility with arbitrary receiver plugins. The DFT run completed initialization checks and random comparisons against the original local implementation.

PASS above means executable and its included consistency checks completed with exit code 0. It does not mean independent mathematical validation. In particular, random LDPC LLRs are expected to fail CRC.

Optional PMU code is retained from the earlier tested hardware-counter implementation. The default portable build smoke matrix above did not enable it. See the README for the explicit PMU build command and permission requirements. A100 was not included in this validation matrix. No test on an unrelated third-party XSAI revision has been performed.

Known caveats: extracted OAI internal regions/structure fields depend on the local version; DFT cooperative scratch is single-caller only; default CSV preserves all codec samples including warmup; timing and CPU usage boundaries differ by module and are documented in README.
