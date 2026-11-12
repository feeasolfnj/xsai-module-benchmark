# Aerial OFDM 接收 benchmark

在单 GPU 上测 Aerial 仿真接收的 14 个 2048 点正向 OFDM 变换。固定场景为正常 CP、30 kHz、106 PRB、单通道。CPU 侧先用 [USAGE.md](USAGE.md#独立符号的整槽计时) 导出有效 IQ 和元数据。

## 依赖与构建

准备 Linux、Python 3.10+、CMake、CUDA Toolkit、支持 C++20 的主机编译器、HDF5 C 开发库及 MathDx cuFFTDx。Aerial 源码基线固定为 `4f65f97c1d5f701ce911f7dda8f1b1f3f0c7693c`。可使用该提交的干净 checkout，或者传入由该提交导出的未压缩 Git tar。构建仅复制所需接收源码到输出目录。SM 121 下生成限于 2048 点非 PRACH 的架构分派补丁，原接收 kernel 保持原有处理步骤。DGX Spark GB10 实测构建使用 CUDA 13.1.115、MathDx 25.06.1 和 cuFFTDx 1.5.1。

```bash
python3 bench.py aerial-ofdm prepare \
  --iq results/ofdm-cpu2/iq14_le_i16.bin \
  --oai-manifest results/ofdm-cpu2/metadata.json --out results/ofdm-fixture

python3 bench.py aerial-ofdm build --aerial /your/path/to/aerial \
  --cuda-arch 121 --cuda-compiler /usr/local/cuda/bin/nvcc \
  --mathdx-dir /usr/local/nvidia/mathdx/25.06/lib/cmake/mathdx --out build/aerial-ofdm
```

`--cuda-arch` 必须与设备一致。固定 Aerial 源码提供 80、89、90 分支，本适配增加了 121。非默认的 MathDx 配置目录通过 `--mathdx-dir` 指定。每次 build/run 使用新目录。可选 `--wait-lock` 等待 benchmark 锁。

## 单槽和并发

```bash
python3 bench.py aerial-ofdm run --build build/aerial-ofdm \
  --fixture results/ofdm-fixture --warmup 20 --samples 200 --out results/aerial-slot

python3 bench.py aerial-ofdm run --build build/aerial-ofdm \
  --fixture results/ofdm-fixture --requests 8 --warmup 20 --samples 100 --out results/aerial-b8
```

- 单槽输入由归档 IQ 补正常 CP 后转换为 FP32，保持原整数值。CUDA event 包围一次接收调用，包含按 CP 偏移读取、FFT、归一化和每符号 1272 个有效频点的写回。设备输入和 CP 表预先准备，主机传输、初始化、预热和输出验收在区间外。
- CPU 整槽输出每符号 2048 个频点，采用 CPU 墙钟且包含任务分发和等待。记录双方计时口径、输出范围和数值格式后再比较。整槽时间除以 14 是摊销开销。
- `--requests 1/2/4/8/16/32/64` 启用 CUDA Graph。每个独立槽拥有各自的设备缓冲和接收描述符。每槽响应从同一批起点计到自己的完成 event。持续吞吐使用另一组不含内部标记的图。Graph 容量从 `--requests 1` 比起，普通单槽仍用独立入口。
- 使用独立双精度单位化参考验证 1272 个频点，最坏符号相对 L2 误差阈值为 15%。该门限用于验证本 benchmark 的数值通路。

## SM 分配和时间线

构建时添加 `--sm-placement` 并另设输出目录，再运行普通单槽命令，可记录每个线程块进入时的 SM ID。该二进制只用于分配诊断。并发容量和单槽延时使用无诊断打点的构建。

如有 Nsight Systems，用 `nsys profile --trace=cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node` 单独采集 CUDA 时间线并导出 SQLite，再执行 `python3 aerial_ofdm_trace.py /path/to/trace.sqlite` 汇总启动几何及 kernel 重叠。实际 warp、计算及内存利用率另需硬件计数器。

构建保存 `build.json`、配置及编译日志、源文件和二进制哈希。运行保存 `metadata.json`、`run.log`、`results.csv` 和 `summary.json`。并发另有 `responses.csv`。报告保留设备输出供校验。
