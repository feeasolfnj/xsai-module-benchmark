# K3 / x86-64 / ARM64 使用说明

本仓库测试本机的 OAI/Sionna Research Kit 源码和库，支持 64 位 Linux。
不是 TensorFlow Sionna 的 Python benchmark，也不是跨架构直接运行 K3 二进制文件。

## 构建配置

| 配置 | 选择方式 | 编译和链接 | DFT2048 / IDFT2048 |
|---|---|---|---|
| K3 XSAI | RISC-V 主机默认，或 `--profile xsai` | 保留原 RVV 构建配方；编译器从本地 CMakeCache 读取，可用 `--cc gcc-16` 覆盖 | 原实现单线程；RVV 候选八线程共同处理一份变换 |
| 官方 SRK | x86-64、ARM64 主机默认，或 `--profile official` | 读取本地 CMake 的实际 ISA/SIMD 参数和链接依赖，不携带 K3 RVV 参数 | 本地原生库的串行变换；八线程请求仍串行，日志和元数据明确标识 |

`--xsai`、`--srk`、`--source` 是同一参数的别名。可以传 OAI 源码目录，也可以传已准备好 `ext/openairinterface5g` 的 SRK 根目录。
构建输出和必要的兼容源码仅写入 benchmark 的 `build/`，不会修改用户的 OAI 源码或覆盖库。

## K3 上的现有用法

```bash
python3 bench.py doctor --xsai /home/ubuntu/XSAI_AIRAN/.work/oai-a-github --cc gcc-16
python3 bench.py build --xsai /home/ubuntu/XSAI_AIRAN/.work/oai-a-github --cc gcc-16
python3 bench.py run symbol --threads 1
python3 bench.py run symbol --threads 8
```

若厂商内核默认只向普通进程暴露 X100，须先通过本机已有的 A100 启动器启动整个 Python 命令，再指定 A100 的实际 CPU 编号。
`run --launcher` 仅包裹被测可执行程序，不会改变 Python 预检时已经看到的 CPU 允许集。
启动器需支持转发该 Python 命令；具体路径和接口以板子已有工具为准。

## 官方 SRK，x86-64 或 ARM64 Linux

先按 [官方 SRK 文档](https://nvlabs.github.io/sionna/rk/)准备、配置并编译自己的 SRK/OAI。
本工具需要完整的源码、生成头文件、静态库、动态库和 CMake 构建配置。仅下载模型或安装 Python Sionna 不够。
接口开发目标为 SRK v1.3.3（`37ef3311e9a6ccb5f1ecd4df2cba970917539a86`）搭配官方默认 OAI `2025.w34`；其他版本需通过实际构建验证。

下面的 `/opt/sionna-rk` 和构建路径请替换为本机路径：

```bash
git clone https://github.com/feeasolfnj/xsai-module-benchmark.git
cd xsai-module-benchmark

# 已有 SRK 工程，标准构建目录
python3 bench.py doctor --srk /opt/sionna-rk
python3 bench.py build --srk /opt/sionna-rk

# 构建目录在其他位置时，doctor 和 build 均指定它
python3 bench.py build --srk /opt/sionna-rk \
  --build /opt/sionna-rk/ext/openairinterface5g/build --profile official

# 可以先构建一项以检查版本兼容性
python3 bench.py build --srk /opt/sionna-rk --only llr
python3 bench.py run llr --threads 1
python3 bench.py run symbol --threads 8
python3 bench.py run UL --threads 8 --warmup 10 --trials 100
python3 bench.py run dft --threads 1
python3 bench.py run idft --threads 1
```

Makefiles 构建使用 `flags.make` 和 `link.txt`。Ninja 构建需要开启 `CMAKE_EXPORT_COMPILE_COMMANDS=ON`，并通过 `ninja -t commands` 读取链接命令。
若缺少元数据，应在原来的构建目录保留原配置重新执行 CMake，加上 `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`，再构建 `nr_ulsim`、`nr_dlsim`、`dfts` 等目标。
编译器默认与原工程一致；强制更换编译器时需保证 ABI 兼容。
检测到 `libdfts.so` 的 ELF 架构与本机不同会立即报错。

XSAI 新增的信道测量、资源提取、信道补偿统计字段在官方头文件中不存在时，仅在 benchmark 私有源码副本中跳过对应的统计开始、停止或归并调用；模块本身的外层墙钟计时仍覆盖真实计算。
这不保证任何 OAI/SRK 版本的所有结构体和内部函数都保持兼容。编译失败时需提供完整构建日志和本地源代码版本。

### 本地实现与提取代码的边界

信道估计、测量及符号处理沿用本仓库提取的 PUSCH 区域和内部处理代码，结合本地头文件、SIMD 配置及依赖库构建。
它们不是直接调用任意版本本地 `nr_ulsch_demodulation.c` 中的整个静态函数。
若用户自行修改了那些内部处理函数，需同步本仓库的提取区域，不能认为重新链接库就会包含所有修改。
LLR 核心、编译进库的估计/测量函数、LDPC 编解码和原生 DFT 依赖使用本地构建；神经网络使用指定的外部插件。
K3 的八线程 DFT 是本仓库提供的候选内核，基线和一致性参照为本地 `libdfts.so`。

## 神经网络后端

### K3 原生 CPU/RVV 插件

```bash
python3 bench.py run neural --threads 8 \
  --library /path/libreceiver_pool8.so --weights /path/model.weights
```

网络权重和线程并行能力由使用者提供的插件决定。

### 官方 CUDA / TensorRT 插件

需要 NVIDIA GPU、驱动、CUDA、TensorRT，以及针对本机 GPU/TensorRT 构建的插件和 engine。
ARM64 CPU 本身不能执行 CUDA；普通 x86-64 CPU 本身也不能执行 CUDA。
TensorRT engine 应按官方教程在目标环境中准备，不应将 K3 `.weights` 文件当作 engine 使用。

```bash
python3 bench.py run neural --threads 1 --neural-backend tensorrt \
  --library /opt/sionna-rk/ext/openairinterface5g/cmake_targets/ran_build/build/libreceiver_trt.so \
  --engine /absolute/path/to/receiver.engine
```

程序在本次结果目录内生成官方插件读取的 `plugins/neural_receiver/config/receiver_trt.config`，不更改用户工程的配置文件。
计时覆盖完整 `receiver_compute_llr()` 回调。目标官方实现内部会同步 CUDA stream，因此其耗时包含 GPU 完成等待；不是仅测异步提交。
若自行替换成异步返回的插件，必须自行确保回调返回前输出完成，否则计时和输出检查没有相同含义。
本测试保持 24 RB、13 符号、16-QAM、1Rx、1层随机接口输入；应使用支持这一接口和尺寸的模型。

`--threads 8` 仅限制 CPU 允许集，不会把一次 TensorRT 推理变成八路 CPU 推理。
`cores` 来自进程 CPU 时间 / 墙钟时间，**不包括 GPU 利用率**，不能与 K3 的 CPU 用核量直接当作同类加速比比较。
普通 GitHub 托管 runner 无 NVIDIA GPU；CPU CI 不能证明 TensorRT 推理已在 GPU 上通过。

### 官方 OpenVINO / BF16 CPU 插件（x86-64）

SRK 的 CPU 接收机另有一条 OV 候选实现：同一个 ONNX、同样的前后处理与量化，只把推理调用从 ONNX Runtime
换成 OpenVINO 2025.4，并以 BF16 精度提示执行。它导出与 CPU 插件相同的 `receiver_*` 符号，本仓库按第三种后端接入：

```bash
python3 bench.py run neural --threads 1 --cpus 0 --backend ov \
  --library /absolute/path/ov.so \
  --ov-model /absolute/path/nrx_oai.onnx \
  --ov-topology L --seconds 20
```

- 不接受 `--weights` / `--engine`：模型由 `--ov-model` 写入本次结果目录的
  `plugins/neural_receiver/config/receiver_cpu.config`，插件读该文件并回显路径，因此"实际计时的模型"
  可被逐字核对。不设置 `NRX_OV_MODEL`（它会在读配置之后覆盖模型，使回显失去证明力）。
  `--ov-topology L|C` 选拓扑，L 为进程级共享 CompiledModel，C 为每线程私有；默认跟随插件（L）。
- 运行后解析插件自报的 `NRX_OV_PROPS` 与每线程 `NRX_WORKER_PROPS`，并要求：回显的模型路径等于 `--ov-model`、
  拓扑回显一致、`inferencePrecision=bf16`、每个 worker 的 `CPolicyMatches=1`。任一不满足即判失败。
  `NRX worker property readback failed or C policy mismatch` 直接终止。
- `OV_BF16_HINT` 只是执行提示，插件自己记录 `amxProven=false`。这些字段只能证明"确实走 OV BF16 路径"，
  **不是逐算子 AMX 证据**，也不能与 demapper 的 AMX 数字并列。
- 输入形状固定为 288 RE（24 PRB）×13 符号 ×1 天线 ×1 层；桥接层会把模型 reshape 到该形状，不匹配即抛错。
  这与本仓库 neural 夹具（24 RB、13 符号、1Rx、1 层）一致，不需要额外改规模。
- 单次回调量级是毫秒级（历史 P0-L 约 51 ms/次），`--seconds` 窗口下记录条数远少于 PUSCH 类指标，属正常。

## 输入和结果的解释

随机范围、计时边界、预热剔除、验证范围沿用 [USAGE.md](USAGE.md) 和 [README.md](README.md)。
所有架构默认尝试采集硬件 CPU 周期；未暴露 PMU 或权限不足时周期为 null，耗时继续测试。可使用 `--cycles off/required`，详见 [CYCLES.md](CYCLES.md)。
UL 当前默认从随机payload编码、BPSK映射、AWGN及量化构造合法码字对应的软输入，另报CRC和payload检查结果。旧版本直接随机LLR的全部失败结果属于不同输入协议，不能与新版混用；详见 [UL_DECODING.md](UL_DECODING.md)。
DFT 的原生后端是调用本地库再检查重复一致性，不是独立数学参考正确性证明。
1Rx 信道估计、测量和 LLR 仍为串行测试；八线程配置不保证八核同时计算。
不同架构的频率、SIMD、编译器、库版本和缓存条件应随结果记录，不能将 hosted runner 的时间作为 K3 的硬件性能对照。

## 自动验证范围

`Official SRK CPU modules` 工作流在 x86-64 和 ARM64 Linux runner 上从固定官方源码构建本地 OAI 库，再构建本仓库并执行八个 CPU 指标的单/八线程配置。
为避免托管 runner 依赖 NVIDIA GPU，CI 单独关闭插件初始化钩子，保留所测 DSP/LDPC 实现；这份测试夹具仅写入 CI 临时目录。
用户正常运行不执行 `tests/prepare_official_ci.py`。它不是生产工程安装器。
神经插件会参与编译，但官方 TensorRT 的 GPU 推理需要另行在有 GPU 的机器验证。
具体已完成和未完成的验证见 [VALIDATION.md](VALIDATION.md)。
