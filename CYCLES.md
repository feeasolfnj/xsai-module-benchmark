# 自动采集硬件周期数

正常 `build` 会为全部模块构建硬件周期入口，`run` 默认使用 `--cycles auto`。
无需手动选择 RISC-V、x86 或 ARM 的计数寄存器，也不要求安装 perf 命令行工具。
仍需要 Linux 内核提供 PMU 事件，并允许当前用户访问。程序不修改系统权限、内核或虚拟机设置。

```bash
# 首次或更新代码后重新构建；路径使用本机实际位置
python3 bench.py build --source /path/to/oai --build /path/to/oai/build
# 默认自动采集：失败时仍完成耗时测试
python3 bench.py run symbol --threads 8
# 仅测耗时，便于与此前耗时记录比较
python3 bench.py run symbol --threads 8 --cycles off
# 必须取得有效硬件周期数，否则本次测试失败
python3 bench.py run symbol --threads 8 --cycles required
```

保留旧 `build --pmu` 选项：将该构建的默认运行模式设为 `required`。
运行时可用 `--cycles auto/off/required` 覆盖。旧版本构建需要重新编译。

## 输出

每次运行保留原来的耗时和用核量，在终端、`summary.json`、`metadata.json` 中增加 `hardware_cycles`，并单独生成 `cycles.json`。

| status | 含义 | cycles / cycles_per_unit |
|---|---|---|
| available | 取得通过基本完整性检查的硬件计数 | 实测数值 |
| unavailable | 没有硬件事件、访问被拒绝或未编译入口 | null |
| disabled | 指定了 off | null |
| invalid | 读数失败、未运行、发生复用或计数不完整 | null |

不可用原因包含 `hardware_event_unavailable`、`permission_denied`；保留 errno。
程序不会把不可用写成 0，也不会用耗时乘主频、TSC 或软件 task-clock 冒充 CPU 周期数。
有效硬件记录仍写入 `results.json/csv` 的 `PMU_CYCLES` 行，原始日志包含 `PMU_STATUS`。

## 范围和解释

- PUSCH、LLR、CPU 神经网络回调按一次模块调用统计；编解码按一个 TB；DFT/IDFT 按一次 2048 点变换（批量采集后除以变换次数）。预热不计入周期结果。
- 使用任务绑定的 `PERF_COUNT_HW_CPU_CYCLES`。事件在工作线程创建前打开并继承至子线程，Linux 汇总计数；不统计被调度出去期间、内核态和 hypervisor 执行。所选 CPU 必须支持这一事件。
- 多线程结果是各线程用户态周期的总和，不能当成最长线程的周期数，也不能直接除以单个 CPU 主频还原并行墙钟耗时。
- 内存等待、用户态同步、插件代码及测量包装在窗口内发生时仍会计入。这不是仅统计算法算术指令的理想周期数。
- PMU 窗口包住原墙钟计时范围，包含少量时钟读取和计数开关边界开销；记录空窗口开销用于诊断，不自动扣减。对小模块应增加批量/测量量。启用 PMU 会扰动时间结果，正式耗时比较请统一 cycles 模式，必要时另跑 off。
- 不使用复用比例缩放，也不自动扣除测量开销。计数未正常运行或复用时丢弃整轮周期结果，auto 模式保留正常的耗时结果，required 模式失败。
- 神经网络 GPU 后端（包括接收机和新增 `demapper` 解调器）这里只计主机 CPU 周期，不能表示 GPU 周期。`demapper` 按完整回调调用统计，默认每批 32 次；详见 [DEMAPPER.md](DEMAPPER.md)。
- 不同架构、CPU 型号、核心类型的周期数不能直接当成相同工作量或 IPC；应一并保留模型、输入、源码版本、频率和线程配置。

WSL、容器、虚拟机或受限用户可能只能完成耗时测试。自动检测会明确记录这个事实；增加软件计时器不能创造没有暴露出来的 PMU。

## 验证

`python3 -m unittest discover -s tests -v` 包含计数状态和结果验证：无硬件/无权限继续运行、required 明确失败、off 跳过采集、复用/缺失/错误数据不会被报告成有效周期数。
其中 syscall fixture 使用合成计数，仅验证逻辑，不是硬件性能数据；实际硬件能力必须在目标机器运行确认。
