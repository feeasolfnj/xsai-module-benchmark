# XSAI module benchmark

**跨架构使用先看 [K3 / 官方 SRK x86-64、ARM64 说明](MULTIARCH.md)**。

**完整参数说明见 [新版完整使用说明](USAGE.md)**：环境检查、非连续CPU集、输入库、UL/DL扫参、预热剔除和DFT校验模式。

在 **K3 RISC-V、x86-64、ARM64 Linux** 上，用直接生成的随机输入测试使用者自己的 XSAI 或官方 SRK/OAI PHY 模块。不是 RFsim，不包含完整 XSAI 工程、预编译库、神经网络权重或测试机器的私人日志。

## 测试项目

| metric | 可执行程序 | 范围 |
|---|---|---|
| estimation | test_pusch_estimation | 50 RB，13符号，3 DMRS 的信道估计区域，包含内部测量 |
| measurement | test_pusch_measurement | 首次DMRS对应测量区域 |
| symbol | test_pusch_symbol | 一个50 RB PUSCH的符号任务构造、执行、等待与统计合并 |
| llr | test_llr_region | 16-QAM，600 RE的完整LLR计时区域 |
| UL | nr_ulschsim | 50 RB、两个码块的整个TB译码 |
| DL | nr_dlschsim | 50 RB、两个码块的整个编码调用 |
| neural | test_neural | 24 RB，完整receiver_compute_llr插件回调 |
| dft / idft | bench-coop | 单份2048点变换；K3 RVV 可八线程协同，官方配置使用本地串行原实现 |

PUSCH估计、测量、符号处理是提取的原处理区域；不是上游正式发布的独立模块API。构建时链接本地XSAI库。源码/结构体不兼容时必须修改适配，不保证任意XSAI版本都能编译。

## 环境和本地工程

- 64位 Linux；K3 RVV 配置要求 VLEN=256bit 和兼容编译器。x86-64 / ARM64 官方配置读取本地 CMake 参数，详见 [MULTIARCH.md](MULTIARCH.md)。需要 Python3.10+、make/CMake 或 Ninja、ar、C++链接器。
- 提前编译自己的XSAI，保留源码、CMake构建目录、静态库、动态库与flags.make。建议先完成本地工程的 `nr_ulsim`、`nr_ulschsim`、`nr_dlschsim` 构建；具体配置跟随该工程文档。
- 这不是只提供安装后的几个动态库就能编译的项目。
- 已知开发环境XSAI Git基线：`18276ae0b271dea9b68420eecc58dd8b45e28787`，包含本地修改（例如RVV内核与统计字段）；commit相同也不能保证工作树相同。仓库运行时会校验并行候选相对本地原实现的一致性。
- 不会修改你的XSAI源文件或覆盖其库。缺少的小型依赖会在本仓库build目录构建。缺少大范围工程依赖时请先完成XSAI构建。

## 快速使用

```bash
git clone https://github.com/feeasolfnj/xsai-module-benchmark.git
cd xsai-module-benchmark

# 指向你自己的源码目录；构建目录默认位于它的cmake_targets/ran_build/build
python3 bench.py doctor --xsai /your/path/to/XSAI
python3 bench.py build --xsai /your/path/to/XSAI

# 构建目录在其他位置
python3 bench.py build --xsai /your/path/to/XSAI --build /your/path/to/build

# 先只构建一项也可以
python3 bench.py build --xsai /your/path/to/XSAI --only llr

# 单核 / 八线程配置；普通时间测量通常不需要sudo
python3 bench.py run symbol --threads 1 --cpus 7
python3 bench.py run symbol --threads 8 --cpus 0-7

# 想固定频率时显式指定，需要权限；不指定就保留当前频率
sudo -n python3 bench.py run symbol --threads 8 --cpus 0-7 --freq 2200000

python3 bench.py run estimation --threads 1
python3 bench.py run measurement --threads 8
python3 bench.py run llr --threads 1
python3 bench.py run UL --threads 8
python3 bench.py run DL --threads 8
python3 bench.py run dft --threads 8 --seed 1
python3 bench.py run idft --threads 1 --seed 1
```

`--threads 8`是配置，不保证模块能够用满8核。LLR当前仍是单线程；1Rx信道估计/测量也可能只用1核。K3 RVV 配置中 DFT/IDFT的8线程是共同处理**一份**变换，不是8进程吞吐测试；候选未自动接入生产链路。该候选使用共享工作数组，基准程序串行提交变换，不支持多个调用者同时重入。

DL使用你本地的LDPC编码库；开发记录中曾使用过的私有自适应编码库不在此仓库内，也不自动加载。不同后端的性能结果必须分别标识。

### 神经网络

由使用者提供与其XSAI接口兼容的插件和权重，单/八线程应分别提供对应插件。不会替你将单线程插件自动并行化。

```bash
python3 bench.py run neural --threads 1 --cpus 7 \
  --library /your/plugins/libreceiver_single.so --weights /your/models/model.weights
python3 bench.py run neural --threads 8 --cpus 0-7 \
  --library /your/plugins/libreceiver_pool8.so --weights /your/models/model.weights
```

A100应按本地设备实际核编号使用 `--cpus 8` 或 `--cpus 8-15`。某些后端需要硬件专用launcher，可加 `--launcher /your/path/run_on_a100`。当前K3的A100频率表最高1.8GHz，不能假定支持2.2GHz。核编号不是跨设备通用标准，需先查CPU拓扑及允许CPU范围。

### 硬件周期（可选）

默认仅测时间，不依赖PMU权限。要启用真实硬件周期，请以 `--pmu` 重新构建，再用有perf权限的账号运行：

```bash
python3 bench.py build --xsai /your/path/to/XSAI --pmu
sudo -n python3 bench.py run llr --threads 1 --cpus 7
```

DFT/IDFT目前不提供PMU统计，即使启用了此构建选项。PMU计入窗口内用户态计算、同步自旋及访存停顿，排除内核/虚拟机执行及被调度离CPU的时间；不能消除缓存等调度间接影响。线程累计周期不是墙钟周期。无法开PMU或发生复用时明确失败，不用频率乘时间代替。

## 随机输入与验证

| 输入 | 范围（含端点） |
|---|---|
| PUSCH/神经网络接收I/Q、接口信道分量 | -2048至2047 |
| 测量初始每RB噪声功率 | 1024至2047 |
| 符号处理解扰符号 | -1或+1 |
| LLR计算的幅度阈值分量 | 1至2048 |
| UL译码输入LLR | -127至127，int16数组 |
| DL编码payload字节 | 0至255 |
| DFT/IDFT I/Q分量 | -364至364 |

输入生成、配置、分配、状态恢复、校验在计时外。PUSCH和神经网络等默认种子20261022，可设置 `--seed`。UL每trial由该种子派生新输入，其他测试默认重复固定数组；PUSCH/LLR/neural/DFT可用 `--input-bank` 轮换2至16组预生成输入。批次中的输入指针/索引切换开销仍计入批次时间。

I/Q与信道数组独立生成，不是相互一致的物理信道样本。随机LLR不对应合法码字，CRC失败为预期，不用于BER/BLER结论。UL含实际的早停、共享中止和数据相关迭代，不能默认单/八线程工作量完全相同。

一致性检查以原串行实现为参考，并不证明原算法本身正确。DFT候选初始化还检查不同输入范围、双方向、双缩放是否与本地原库逐位一致。默认strict失败会终止；report保留失败标记，skip明确标记未验证。串行基线不初始化并行候选。运行器对选定输入库复核，不复制外部项目整个测试框架；设计参考了 [dft-idft-benchmark](https://github.com/fuxuandg/dft-idft-benchmark)。

## 输出和重复测试

每次新建 `results/<日期-模块-随机后缀>/`，包含run.log、results.json、results.csv、metadata.json。指定目录用 `--out /new/path`，拒绝覆盖已有目录。元数据包含命令、环境、频率、配置和本地libdfts哈希。

`mean_us`为平均耗时；DFT/IDFT `median_batch_mean_us`是批次均值的中位数，不能说成逐次调用耗时的中位数。编译码默认100次正式调用、10次预热，均可配置。正式CSV/JSON已剔除预热，raw文件保留全部记录。平均用核量为CPU时间/对应墙钟窗口，包含同步自旋；PUSCH重复窗口包含计时外状态恢复/校验，和插件回调窗口的边界不同。

```bash
for seed in 20261022 20261023 20261024; do
  python3 bench.py run symbol --threads 1 --seed "$seed"
  python3 bench.py run symbol --threads 8 --seed "$seed"
done
# 仅打印实际命令及额外环境，不执行
python3 bench.py run dft --threads 8 --print-command
```

原始log和results.raw.csv/json保留预热及正式记录，results.csv/json只保留正式记录；summary.json提供正式统计。频率在finally恢复，进程被SIGKILL或机器断电时无法保证恢复。脚本使用全局锁避免多个benchmark竞争。

## 已验证范围与限制

目标是能在不同目录的兼容XSAI工作树上重建，而不是承诺任意版本或平台通用。构建配方源自上述开发版本，依赖其CMake静态库布局、结构体字段及RVV接口。遇到构建错误应保留build/build-commands.json和输出，检查本地源码/构建是否匹配；不要继续使用旧二进制。

不公布某组延迟为所有K3/XSAI的标准值。比较时保持频率、输入、库版本、核范围、统计方式一致。用户可以通过环境变量影响设备库，请同时检查metadata与实际加载的库。

## 目录与许可

- `src/`：模块驱动、提取的处理区域、随机上下文、可选PMU及协同FFT候选。
- `templates/commands.json`：仅含路径占位符的构建配方。
- `bench.py`：构建与运行入口；只用Python标准库。
- `LICENSE`、`NOTICE`：保留OpenAirInterface派生代码的OAI Public License v1.1及来源说明。
- `VALIDATION.md`：本次可移植整理版本的实际验证记录。

外部XSAI、插件和权重各自的许可仍适用。忽略build、results及机器配置文件，不上传凭据、库或模型。
