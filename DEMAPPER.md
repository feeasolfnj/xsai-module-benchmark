# 神经网络解调器 benchmark

`demapper` 独立测量外部插件的完整 `demapper_compute_llr()` 回调。支持 K3 CPU/RVV 插件和官方 CUDA/TensorRT 插件。默认 600 RE、16-QAM，每次产生 2400 个 int16 LLR。

## 与官方 testbench 的关系

官方 SRK 的 [test_demappers.py](https://github.com/NVlabs/sionna-rk/blob/main/plugins/neural_demapper/tests/unit/test_demappers.py) 是正确性 pytest：构建 Python 扩展，在 GPU 上运行解调器，用 Sionna 产生参考结果。它不提供本仓库的完整插件回调计时、CPU 用核量与硬件周期汇总。

本仓库新增 C 性能夹具 `src/random_demapper.c`，直接使用本地官方 `nr_demapper_defs.h` 定义的函数类型，不复制推理算法。性能输入沿用本仓库的随机接口数组协议，不是官方测试中“随机比特 → QAM → AWGN”的准确率输入。另提供入口直接运行使用者本地的官方测试文件，保留原测试和断言。

## 构建

先按 README 准备自己的 SRK/XSAI 及 CMake 构建目录。所有构建默认包含 demapper，也可以单独构建：

```bash
python3 bench.py build --srk /path/sionna-rk \
  --build /path/oai/build --only demapper
```

需要本地 `plugins/neural_demapper/src/nr_demapper_defs.h`；使用本地 PHY_NR 的编译参数。构建测试程序不需要加载 GPU，也不构建或打包模型、CUDA 插件。运行时指定与本机架构和模型兼容的插件。

## 官方 TensorRT 性能测试

```bash
python3 bench.py run demapper --threads 1 --cpus 5 \
  --backend tensorrt --library /absolute/path/libdemapper_trt.so \
  --engine /absolute/path/neural_demapper.2xfloat16.plan \
  --re 600 --seconds 2 --warmup 100 --cycles auto

# 换成你机器实际允许使用的 CPU 集合
python3 bench.py run demapper --threads 8 --cpus 0-7 \
  --backend tensorrt --library /absolute/path/libdemapper_trt.so \
  --engine /absolute/path/neural_demapper.2xfloat16.plan
```

在本次结果目录内生成 `plugins/neural_demapper/config/demapper_trt.config`：第一行是绝对 engine 路径，第二行为 `1`，启用归一化输入；不修改工程原配置。需要对应目标 GPU、TensorRT 版本和归一化两输入模型的 engine。

如果设置过 `PLAN_POSTFIX`，请先 `unset PLAN_POSTFIX`，直接用 `--engine` 指定真实文件。这样记录的 engine 哈希与指定加载文件一致。

## K3 原生 CPU/RVV 性能测试

```bash
python3 bench.py run demapper --threads 1 --cpus 0 \
  --backend native --library /absolute/path/libdemapper_cpu.so \
  --seconds 2 --re 600
```

已有 K3 X100 解调器将参数编译进插件，因此不要求 `--weights`。插件 SHA256 同时标识这份二进制内的参数。外部加载权重的插件（例如 XSAI A100 版）可使用 `--weights /absolute/path/model.weights`，测试器通过 `OAI_DEMAPPER_WEIGHTS` 和 `XSAI_DEMAPPER_WEIGHTS` 传递；**插件必须自行支持相应环境变量**，测试器不会将内嵌模型改成文件加载模型。外部权重哈希只是所提供文件的记录，不证明插件实际使用了它。

插件原有的 `OAI_DEMAPPER_*`、`XSAI_DEMAPPER_*` 环境配置会保留并写入 metadata，包括输入/输出缩放、专用核选择等。CPU 编号、频率及执行权限以本机为准。

K3 的 A100 使用独立调度核集：如果当前 shell 只能使用 X100，必须先用专用 launcher 启动**整个 Python benchmark 进程**，再选择 A100 CPU。只加 `run --launcher` 无法改变启动器预检时可见的 CPU 集合。例如：

```bash
XSAI_DEMAPPER_LLR_SCALE=32 /absolute/path/run_on_a100 \
  python3 bench.py run demapper --threads 1 --cpus 8 \
  --library /absolute/path/libdemapper_a100_rvv.so \
  --weights /absolute/path/demapper_a100.weights
```

## 输入、计时与校验

- `xorshift32` 生成器，默认 seed=20261022；I/Q 为 [-2048,2047]，幅度两个分量各为 [1,2048]，避免零除。I/Q 和幅度分别随机生成，不构成物理一致的链路。
- `--re` 范围 1..32768，但插件和模型可能有更小的上限；当前只测 16-QAM。默认 `--input-bank 1` 重复同一输入；可设为 2..16 轮换预生成数组。随机生成与分配均在计时外。
- 排除动态库加载、模型初始化、线程初始化、首轮参考输出、预热及校验、释放。正式每批 32 次调用，记录回调完整耗时，包含插件内部的预处理、推理、量化、写回和同步等待，以及少量调用/索引开销。
- 官方实现内部完成 CUDA stream 同步后返回，GPU 计时包含等待推理完成；异步返回的自定义插件必须自行实现同等语义。
- `cores` 为进程 CPU 时间 / 经过时间，不包括 GPU 利用率。`--threads 8` 限制允许的 CPU 集合，不会自动创建八份推理，也不会把串行插件变成八线程插件；测试器不创建解调工作线程。
- 所有正式调用必须返回 `1`（已处理）；返回 `0` 的传统路径回退立即失败。另检查非全零输出、输出边界保护和相同输入的输出一致性。
- `verification_scope=repeatability_same_implementation` 只证明重复一致性，不证明数学正确性、BER/BLER 或与独立 Sionna 参考一致。正确性需另跑下一节官方测试。
- 沿用 `--cycles auto/off/required`；无权限或 PMU 时明确为 null。统计的是 CPU 用户态硬件周期，**不是 GPU SM 周期**；不以耗时乘频率代替周期。

输出沿用 `metadata.json`、`run.log`、`results*.json/csv`、`summary.json`、`cycles.json`，另记录插件、模型哈希和 `timing_scope=full_demapper_compute_llr`。

## 直接运行官方正确性 testbench

先按本地 SRK 官方要求准备 Python、Sionna、PyTorch、pytest、nanobind、CMake、CUDA、TensorRT 和对应 GPU engine。此命令不自动安装依赖，不更改官方测试文件。

```bash
# Spark 示例；其他平台按官方 conftest.py 设置 SRK_PLATFORM
export SRK_PLATFORM='DGX Spark'
export PATH=/usr/local/cuda/bin:$PATH
python3 bench.py official-demapper-test --srk /absolute/path/sionna-rk \
  --python /absolute/path/sionna-rk/env/bin/python
```

官方 fixture 会在临时目录构建扩展，从 SRK 的 `plugins/neural_demapper/models` 加载模型；该命令不接受性能入口的 `--engine`。记录 `run.log`、`junit.xml`、`metadata.json`（含上游测试文件哈希）。存在失败、跳过或未收集到测试时返回失败，避免把缺 GPU、缺模型当作已通过。官方测试依赖 CUDA，不能直接验证 K3 RVV 插件。

## 2026-10-27 Spark 验证

ARM64 Spark，GB10 GPU，原有官方 TensorRT 解调器，600 RE、16-QAM、seed=20261022；每配置约 1 秒，预热 100 次，固定输入。

| 主机 CPU 配置 | 平均耗时 μs/回调 | 平均 CPU 用核量 |
|---|---:|---:|
| A725 单线程 | 26.395 | 1.00 |
| A725 八线程允许集 | 26.394 | 1.00 |
| X925 单线程 | 25.718 | 1.00 |
| X925 八线程允许集 | 25.884 | 1.00 |

四项配置通过回调处理状态、重复一致性及缓冲区边界检查，另以双输入库检查 1、2、24、32、512、600、1024 RE 七种尺寸，均通过。官方 pytest **11/11 通过、无跳过**；新增回归后仓库 **44 项测试通过**；原八个 CPU 模块的单/八线程冒烟测试及 PMU 模式检查通过。Spark 的硬件周期因系统权限限制为不可用。

## 2026-10-27 K3 验证

连接恢复后，将本仓库独立部署到 `/home/ubuntu/benchmark-demapper-20261027/repo`，以 XSAI 本地 CMake 编译参数构建 `test_demapper`，直接加载现有 X100/A100 插件，没有重新编译推理模型。

两组均为单核、600 RE、16-QAM、seed=20261022、输入缩放 1、输出缩放 32；预热 100 次，每轮约 2 秒，共 3 轮。下表为三轮均值，周期是 `perf_event_open` 实测的每回调用户态 CPU 周期。

| 插件 / CPU | 当前频率 | μs/回调 | 硬件周期/回调 | 平均 CPU 用核量 |
|---|---:|---:|---:|---:|
| sionna-rk X100 内嵌模型 | 2.2 GHz | 118.695 | 259568 | 0.997 |
| XSAI A100 外部权重模型 | 1.8 GHz | 163.242 | 291929 | 0.997 |

两组默认输入哈希及最终量化输出哈希完全一致，正式调用全部返回已处理，重复一致性和边界检查通过。双输入库的 1、2、24、32、512、600、1024、2048 RE 均通过。另确认 X100 在 2049 RE 返回回退时，benchmark 明确失败而不报告神经推理耗时；A100 的 2049 RE 通过。K3 上 44 项回归测试全部通过。

这验证了新增入口的实际运行、计时与硬件周期采集，不等于重新进行了独立参考正确性或 BER/BLER 测试。两种核频率不同，不应仅凭此表判断相同频率下的核性能。
