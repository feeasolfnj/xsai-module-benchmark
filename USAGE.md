# 新版使用与结果口径

跨架构配置、官方 SRK 和 TensorRT engine 用法见 [MULTIARCH.md](MULTIARCH.md)。

## 克隆后的操作

```bash
git clone https://github.com/feeasolfnj/xsai-module-benchmark.git
cd xsai-module-benchmark
python3 bench.py doctor --xsai /your/path/to/XSAI
python3 bench.py build --xsai /your/path/to/XSAI
python3 bench.py run symbol --threads 1
python3 bench.py run symbol --threads 8
```

K3 的 xsai 配置前提是 Linux、RVV、Python 3.10+、GCC16、C++链接器、ar、taskset，以及兼容的本地 XSAI 源码和 CMake 构建目录。x86-64 / ARM64 的 official 配置使用本地 OAI 的编译器与架构参数，具体环境见 MULTIARCH.md。保留生成的头文件、静态库、动态库。补建缺失helper时需要 flags.make，或开启 CMAKE_EXPORT_COMPILE_COMMANDS 后的 compile_commands.json；完整Ninja构建不会仅因缺少flags.make而被拒绝。Ninja缺MAC库时按doctor提示先构建MAC_UE_NR目标。仅安装几个动态库不够。构建目录不同则为 doctor/build 加 `--build /path/to/build`。

doctor 列出缺失库和补建命令，不安装软件、不修改 XSAI 源码。MAC 依赖可在 benchmark 私有 build 目录补建。上游 asn1c/LTE SSE 等问题仍需按 XSAI 文档处理。

## CPU、参数、输入

```bash
# 默认从当前允许的CPU集中选择，不再写死CPU7
python3 bench.py run symbol --threads 8 --cpus 0,2,4,6

# 轮换4组预生成随机输入；默认1组是固定输入热缓存稳态测试
python3 bench.py run symbol --threads 8 --input-bank 4 --seconds 2
python3 bench.py run llr --threads 1 --input-bank 4

# UL/DL 可调整无线参数；trials不含预热
python3 bench.py run UL --threads 8 --rb 50 --carrier-rb 106 --mcs 14 --symbols 13 --snr 20 --warmup 10 --trials 100
python3 bench.py sweep examples/sweep.json
```

- CPU 集可不连续、可少于工作线程数；超额订阅会提示风险。taskset 限制允许集，没有逐线程一核绑定。
- 1Rx estimation、measurement、llr 是串行后端，八线程配置会明确提示，不自动获得并行加速。
- 神经插件自行决定线程数量，`--threads` 不会把单线程插件自动并行化。
- `--input-bank` 支持 PUSCH/LLR/neural/demapper/DFT，1–16组。UL将一份随机payload编码后，每trial加入新AWGN；DL仍复用随机payload，不支持输入库。
- 输入生成在计时外；PUSCH恢复也在模块计时外。LLR批内切换指针、DFT选择索引的开销计入批次时间。输入库不保证冷缓存。
- RB/MCS/符号数/SNR/trials 仅用于 UL/DL；层数固定1，其余PUSCH规模仍固定。不能对不支持的模块静默应用选项。
- `--seconds` 仅用于PUSCH/LLR/neural。UL/DL用 `--trials`；DFT用 `--iterations` 和 `--batches`。
- DL 保留原程序3次外层链路循环，内部编码次数由 trials+warmup 控制。以实际日志和正式记录数为准。

配置列表是 JSON 数组，每项填写 metric、threads 及 run 参数，键名可用下划线或连字符。各配置顺序运行、分别出结果。

## DFT/IDFT

```bash
# 串行基线不初始化八线程候选
python3 bench.py run dft --threads 1
# 默认strict，不一致即失败
python3 bench.py run dft --threads 8 --input-bank 4
# 允许诊断计时，仍明确保留失败标记
python3 bench.py run dft --threads 8 --verify report
# 不比较，标记为skipped，不能宣称正确
python3 bench.py run idft --threads 8 --skip-verify
python3 bench.py run dft --threads 8 --iterations 1000 --batches 7 --warmup 100
```

`median_batch_mean_us` 是各批平均值的中位数，不是单次延迟中位数。`mean_us` 为全部批次摊销耗时。初始化比较16种模式、双方向、两种缩放，正式输入另行比较。串行基线的比较只是原实现重复性，不能冒充独立参考。

这里计时一次2048点变换，**不含CP添加、资源提取或完整symbol_dft/symbol_idft外层**。八线程协同处理同一变换，共享工作区，不支持多调用者重入。

## 神经插件

```bash
python3 bench.py run neural --threads 8 --cpus 0-7 --input-bank 4 \
  --library /your/plugin_pool8.so --weights /your/model.weights
```

插件必须与本地 XSAI ABI、CPU指令集和权重格式兼容。导出 receiver_init、receiver_init_thread、receiver_compute_llr；回调回退会失败。A100专用插件不能在只有X100允许集的会话运行；X100插件不需要A100。脚本不能根据库名自动识别其指令集。需要硬件launcher时显式传 `--launcher`。

## 频率、锁、结果

```bash
# 不改频率，记录实际频率和governor
python3 bench.py run symbol --threads 8
# 事先配置userspace、可写scaling_setspeed及受支持频率
sudo -n python3 bench.py run symbol --threads 8 --freq 2200000
python3 bench.py run symbol --threads 8 --wait-lock
```

锁频前检查全部涉及的策略，结束或异常恢复原设置，不自动改governor。只检查运行前后频率，不能证明全过程没有瞬时变化。全局锁冲突友好报错，或显式等待；获取锁之前不创建结果目录。跨用户测试应统一使用可访问锁文件的账号。

每个结果目录：

- metadata.json：命令、CPU、频率、种子、输入模式、预热、运行状态。
- run.log、results.raw.csv/json：完整记录，保留预热。
- results.csv/json：正式记录，剔除UL/DL预热。
- summary.json：正式调用均值、平均用核量和DFT批均值指标。

`status=passed` 表示运行流程完成，算法检查另看 `verification/verification_scope`；report模式可能完成运行但比较失败。PUSCH只检查串行等价，neural/LLR/DL主要检查重复性。UL使用合法码字加噪输入，`decoder_quality` 记录剔除预热后的成功数、失败数、BLER及译码前BER；低SNR下译码失败仍可正常报告，不把计时完成当作译码成功。

`cores` 是CPU时间/对应墙钟窗口。PUSCH用核量窗口包含恢复/周期性检查，模块耗时窗口不包含它们；neural逐回调、LLR/DFT按批次统计。它不是“曾参与的物理核个数”，也不能跨口径直接当并行效率。

改源码或构建失败后拒绝运行旧二进制，需重建。默认自动尝试PMU，包含DFT/IDFT；不支持或无权限时记录null并继续耗时测试。`--cycles off`关闭采集，`--cycles required`要求有效硬件计数。详见 [CYCLES.md](CYCLES.md)。PMU包含同步和访存停顿，不能称为无任何系统影响的纯指令周期。

```bash
python3 -m unittest discover -s tests -v
```

CI检查启动器逻辑，不能代替K3实测。支持范围取决于XSAI源码/API版本，不承诺任意工作树都无需适配。

## 神经网络解调器新增入口

`python3 bench.py run demapper --threads 1 --library /absolute/path/libdemapper_cpu.so` 测量完整解调插件回调；TensorRT 使用 `--backend tensorrt --engine /absolute/path/model.plan`。
构建方法、随机输入、CPU/GPU 计时边界和直接运行官方正确性 pytest 的方法见 [DEMAPPER.md](DEMAPPER.md)。
