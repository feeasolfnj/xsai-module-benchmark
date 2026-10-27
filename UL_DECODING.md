# UL 译码：合法码字加噪输入

当前 `run UL` 保持整个 `nr_ulsch_decoding()` 调用的计时范围，将旧版“直接随机 LLR”改成 OAI `ulschsim.c` / `ldpctest` 所采用的编译码测试思路：

```text
固定种子生成随机信息字节
  → NR ULSCH 编码（包括对应 CRC、分段、LDPC、速率匹配等）
  → 按编码比特 0→+1、1→−1 映射
  → 每 trial 加入新的高斯白噪声
  → 定标并量化成整数软信息
  → 整个 TB 译码
  → 检查 CRC 且恢复出的 payload 与原始字节一致
```

这是对现有 benchmark 的输入构造改造，不是将性能指标替换成独立 `ldpctest`。NR 配置仍决定 TBS、码块和速率匹配长度；其中逐编码比特的信道是 BPSK/AWGN，不是完整的 16-QAM PUSCH 波形或 RFsim 链路，即便 MCS14 的配置参数包含16-QAM。

## 使用

```bash
python3 bench.py build --srk /path/sionna-rk --only UL
python3 bench.py run UL --threads 1 --rb 50 --carrier-rb 106 \
  --mcs 14 --symbols 13 --snr 20 --trials 100 --warmup 10
python3 bench.py run UL --threads 8 --cpus 0-7 --snr 20

# 减小SNR会增加译码难度；不要假定所有SNR都能译码成功
python3 bench.py run UL --threads 1 --snr 0 --trials 100
```

默认 seed=20261022、SNR=20 dB、最大 LDPC 迭代次数沿用程序默认5。信息字节通过 `rand()` 生成，使用该seed；信道通过本地OAI的 `randominit` / `gaussdouble` 生成噪声。相同机器、实现、seed和参数下单/八线程使用相同输入序列；不保证不同C库或OAI RNG版本逐比特一致。

与上游编译码测试一致，每进程生成一份随机payload并编码一次，trial之间复用码字、使用新的噪声。不是每trial生成新的payload。默认8 bit软信息，存储类型为int16；上游定标为 `sigma/16`，噪声标准差 `sigma=1/sqrt(2*10^(SNR/10))`。此处SNR属于该编码比特信道定义，不应直接称为任意调制或码率下的 Eb/N0。

## 计时与结果

编码、噪声生成、量化、校验均在译码计时及PMU窗口之外。窗口仍包括整个TB译码调用及其任务分发、等待等处理。默认清除上轮HARQ状态，不进行跨trial重传合并。

`results.json/csv` 的每个 `CODING_UL` 记录增加：

- `input_mode=encoded_bpsk_awgn`、`snr_db`；
- `decode_success`、`crc_valid`、`payload_match`；
- `uncoded_bit_errors` 和 `coded_bits`，用于译码前BER；
- 既有输入/输出哈希、耗时、线程及码块计时统计。

`summary.json` 和 `metadata.json` 的 `decoder_quality` 仅统计正式trial，排除预热：`measured_trials`、`succeeded`、`failed`、`bler`、`false_positives`、`uncoded_ber`。译码成功必须同时满足CRC和原始payload匹配。CRC通过却payload不匹配时保留记录并将运行标为失败。

正常的低SNR译码失败会被报告为BLER，而不是启动器运行失败；因此 `status=passed` 仍不代表所有TB均成功，应看 `decoder_quality`。没有运行完整PHY链路，不能把这里的BLER当作实际16-QAM RFsim系统的BLER。

旧版 `random_llr_not_decode_correctness` 数据仍保留为历史测试结果，不应与本协议下耗时直接拼成同一性能表。合法码字可能更早完成译码，因此新版耗时可能明显下降。

## 验证记录（2026-10-27）

- K3 X100、Spark A725/X925，分别使用单/八线程配置。20 dB每配置100个正式TB，六个配置共600个TB，全部CRC通过且payload与原始信息一致。
- 0 dB、−10 dB各配置20个正式TB：本次均译码失败，译码前BER分别约7.88%、32.85%，没有把低SNR失败当成程序异常或隐藏结果。
- 同一设备、参数和SNR的单/八线程输入哈希序列完全一致；正式统计排除了预热。
- 额外检查6 RB/MCS4/seed0和106 RB/MCS28，在Spark单/八线程各5个正式TB，全部成功，覆盖较小与较大的码块规模。
- K3、Spark上的48项回归测试全部通过。本轮功能验证使用 `--cycles off`，未重新采集硬件周期，不能与此前开启PMU的数据直接拼成同一轮性能表。
