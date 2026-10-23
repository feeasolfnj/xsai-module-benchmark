# Validation record

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
