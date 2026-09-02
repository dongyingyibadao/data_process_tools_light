# LeRobot 并行转换基准实验交接

更新日期：2026-09-02（UTC）

本文用于把当前 LeRobot 转换进展和下一台实例需要执行的并行性能实验一次性交接清楚。新实例上的 Agent 应先读完本文，再检查实际硬件和代码状态。不要直接在正式数据输出目录中做实验。

## 1. 实验目标和已确认约束

本轮只优化 `whole_body_joint` 的视频转换吞吐，核心变量是并行分片进程数。实验约束已经确认如下：

- 取消 LeRobot 转换路径中的 70% CPU 性能上限，允许使用实例实际可用的全部 CPU 配额。
- `encoder_threads` 始终固定为 `4`，不得把编码线程数和分片数同时作为变量。
- 编码参数固定为 H.264、CRF 18、preset `fast`、encoding mode `parallel`。
- 测试规模按帧数定义，不按 Episode 数定义。所有对比配置必须转换同一份约 1000 帧样本。
- 样本准备和一次性的预检不计入转换计时；转换过程中不可避免的逐分片检查应保留并单独说明。
- 不做全量视频解码或全量训练读取验证，只做报告、状态、帧数和零字节文件等轻量检查。
- 每轮必须记录 wall time、frames/s、实际平均 CPU 核心占用、CPU 配额利用率和分片数。
- 单轮目标耗时不超过约 3 分钟。若 1000 帧明显不足以进入稳定编码阶段，应先报告，再把样本适度扩大，不能悄悄改变实验口径。

这里的“相同帧数”指所有并行度配置使用同一个固定样本，总帧数相同，而不是每增加一个分片就增加 1000 帧。LeRobot 当前以完整 Episode 为提交单位，因此样本帧数允许接近而不必严格等于 1000，但一旦选定，所有配置必须完全一致。

## 2. 当前代码状态

仓库：

```text
/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/2026_cz/data_autopro_tools_light
```

交接时分支和提交：

```text
branch: main
commit: bb2ee73 lerobot
```

当前正式多分片入口：

```text
scripts/run_merged_shards.py
```

它当前具备以下行为：

- 从成功的 preflight `conversion_report.json` 读取 accepted Episodes。
- 按总帧数把 Episode 连续均衡到 N 个 shard。
- 使用 `taskset` 给每个 shard 分配互不重叠的 CPU ID。
- 同时启动所有 shard，分别执行 `convert-merged --action-mode whole_body_joint`。
- 固定 H.264/CRF 18/fast/parallel，`encoder_threads` 和 `preflight_workers` 可传参。
- 每个 shard 完成后检查 report、conversion state、pending state 和连续 Episode index。

当前尚未进行任何性能优化或正式基准测试。特别是，70% 限制仍然存在：

```python
# lightworkbench/config.py
available = min(len(affinity), _quota_cpu_count() or len(affinity))
budget = max(1, math.floor(available * 0.70))
```

`resource_budget()` 还会立即调用 `os.sched_setaffinity()`，把当前进程缩到预算 CPU 集合。这个副作用会被 ffmpeg/ffprobe 子进程继承。

需要注意：`lightworkbench/config.py` 同时服务 Web 剪切工作台。若只想让 LeRobot 跑满，优先实现可测试的 LeRobot 专用开关或环境变量，例如将 CPU fraction 配成 `1.0`，并让分片启动脚本显式设置；不要在未评估影响的情况下无意改变 Web 服务默认行为。最终生产 LeRobot 命令应默认或显式使用 100% 可用配额。

另一个待修正点是 `scripts/run_merged_shards.py::_cpu_groups()` 当前假设可用 CPU ID 从 0 连续开始。新实例必须依据 `os.sched_getaffinity(0)` 返回的真实 CPU ID 分组，不能只依据 `nproc` 构造 `0..N-1`。

交接时工作区还有一个未跟踪文件：

```text
?? scripts/merge_completed_shards.py
```

它用于正式数据的 shard merge，不是基准实验的依赖。不要删除、覆盖或顺手提交，除非用户另行要求。

## 3. 已完成的正式转换

原始数据：

```text
/inspire/qb-ilm/project/robot-decision/public/demo2/raw_08_29
```

正式 shard 工作区：

```text
/inspire/qb-ilm/project/robot-decision/public/demo2/lerobot_data_08_29_shards_20260901
```

正式合并输出：

```text
/inspire/qb-ilm/project/robot-decision/public/demo2/lerobot_data_08_29
```

已有结果：

- 原始发现 1142 个 Episodes。
- accepted 914 个，共 256353 帧。
- skipped 228 个，共 88682 帧；这些是已知坏数据，按要求跳过。
- failed 0。
- 六个转换 shard 全部 return code 0，分别约 42730、42451、42910、42699、42836、42727 帧。
- `whole_body_joint` 已合并成功，约 37 GB。
- 合并结果为 914 Episodes、256353 frames、914 meta rows、2742 training videos、2743 auxiliary rows/videos、pending 为 null、零字节文件为 0。
- 多出的 1 个 auxiliary video 是合法的 `head_rear` 流，不是重复或错误文件。

正式 merge 日志：

```text
/inspire/qb-ilm/project/robot-decision/public/demo2/lerobot_data_08_29/merge_whole_body_joint.log
```

`body_joint_eef` 在原实例的 tmux `lerobot_shards:merge` 中派生。它不重新编码视频，只重写约 172 MB 的动作 Parquet 和统计信息，并把 videos/auxiliary 链接到 owner。可查看：

```bash
tmux capture-pane -p -t lerobot_shards:merge -S -40
tail -n 40 /inspire/qb-ilm/project/robot-decision/public/demo2/lerobot_data_08_29/derive_body_joint_eef.log
```

新实例的实验不得写入上述三个正式路径，也不得删除 shard 工作区。

## 4. 为什么旧实例 CPU 看起来很低

旧实例硬件可见值和实际调度配额不同：

```text
lscpu / affinity visible: 128 logical CPUs
cgroup cpu.max:           5500000 100000
cgroup quota:             55 CPUs
current 70% budget:       floor(55 * 0.70) = 38 CPUs
```

正式任务使用 6 shards、`--cpus 54`，每个 shard 初始分到 9 个 CPU ID。但每个 Python shard 导入 `lightworkbench.config` 后，又对自己的 9 核 affinity 应用 70% 限制，实际缩为 `floor(9 * 0.70) = 6` 核。因此六个 shard 最多只保留约 36 个 CPU ID，这与之前观察到的低 CPU 使用率一致。

新实例的数字可能完全不同。必须重新测量，不得复制旧实例的 `54`、`55` 或 `38`。

## 5. 新实例首先执行的环境检查

在修改代码或运行实验前保存以下输出：

```bash
cd /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/2026_cz/data_autopro_tools_light

git status --short
git branch --show-current
git rev-parse --short HEAD
nproc
nproc --all
lscpu
taskset -pc $$
cat /sys/fs/cgroup/cpu.max 2>/dev/null || true
cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null || true
```

使用项目 Python 查看代码当前计算出的预算。注意导入配置会改变这个短生命周期 Python 进程自身的 affinity，但不会改变父 shell：

```bash
PYTHONPATH=. /inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/envs/data_process_pre/bin/python \
  -c 'from lightworkbench.config import RESOURCES; print(RESOURCES)'
```

记录四个不同概念：

- 主机逻辑 CPU 数量。
- 当前进程 cpuset/affinity 中允许的 CPU ID。
- cgroup quota 折算出的 CPU 数。
- 代码施加 70% 后的 budget。

实验可用 CPU 数应取 `floor(min(affinity_count, quota_count))`；若 `cpu.max` 为 `max`，只取 affinity count。

同时确认 Python 和 FFmpeg 环境：

```bash
/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/envs/data_process_pre/bin/python --version
ffmpeg -version
ffprobe -version
```

## 6. 固定 1000 帧样本

不要重新扫描和预检全部 1142 个 Episodes。优先从已经合并的 owner state 或六个 shard report 中读取 914 个成功 Episode 的相对路径和帧数：

```text
/inspire/qb-ilm/project/robot-decision/public/demo2/lerobot_data_08_29/whole_body_joint/conversion_state.json
/inspire/qb-ilm/project/robot-decision/public/demo2/lerobot_data_08_29_shards_20260901/shard-*/conversion_report.json
```

样本选择要求：

- 总帧数尽量接近 1000，建议误差不超过 5%。
- 至少包含计划测试的最大 shard 数量那么多个 Episode，确保每个 shard 非空。
- 建议第一轮最大测试 6 shards，因此样本至少含 6 个 Episode。
- 不要只选择一个超长 Episode，否则无法测试 shard 并行。
- 固定并保存 Episode 相对路径、单 Episode 帧数、总帧数和选择算法。
- 所有配置和重复轮次必须使用同一个列表。

当前 accepted Episode 的帧数范围是 91–1193，中位数 223，均值约 280.47。约 1000 帧且至少 6 个 Episode 是可构造的。

在独立 benchmark 根下，只对这个列表执行一次 `convert-merged --preflight-only`，生成 `scripts/run_merged_shards.py` 所需的成功 preflight report。使用 Bash 数组逐项追加 `--include-episode`，避免超长命令的缩进和引号错误。该准备步骤不计入任何测试轮次。

建议 benchmark 根：

```text
/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/lerobot_parallel_benchmark_20260902
```

每个配置、每次重复必须使用新的输出目录，例如：

```text
run-cap70-shards01-rep01/
run-cap100-shards01-rep01/
run-cap100-shards02-rep01/
run-cap100-shards04-rep01/
run-cap100-shards06-rep01/
```

不要复用已经完成的输出，否则增量 no-op 会让计时失真。

## 7. 实验矩阵

先跑一轮不计入结果的 warm-up，使 Python import、共享库和少量文件页缓存进入稳定状态。不得尝试 `drop_caches`。

正式矩阵建议为：

| 配置 | CPU fraction | shards | encoder threads | 目的 |
| --- | ---: | ---: | ---: | --- |
| baseline | 0.70 | 1 | 4 | 保存当前代码基线 |
| uncapped-1 | 1.00 | 1 | 4 | 测量单 shard 去上限收益 |
| uncapped-2 | 1.00 | 2 | 4 | 并行扩展 |
| uncapped-4 | 1.00 | 4 | 4 | 并行扩展 |
| uncapped-6 | 1.00 | 6 | 4 | 对比正式任务配置 |

每个配置至少重复 2 次，推荐 3 次。轮次顺序应交错，例如先依次跑 1/2/4/6，再反向跑 6/4/2/1，以降低共享存储负载和页缓存造成的顺序偏差。

固定参数：

```text
action_mode=whole_body_joint
video_codec=h264
video_crf=18
encoder_preset=fast
encoder_threads=4
video_encoding_mode=parallel
preflight_workers=6
```

第一轮不要测 `body_joint_eef`，因为它不转码视频，无法回答视频转存并行度问题。merge 也不包含在转换计时中；它是另一个以元数据和文件搬运为主的阶段。

如果 6 shards 时 CPU 仍未接近配额且存储没有饱和，可以在同一固定样本包含足够 Episode 的前提下增加 8 shards。不要因为想测试更高 shard 数而在中途换样本；需要换样本时，应建立第二组独立实验并重新跑全部对照组。

## 8. 计时和 CPU 记录

必须使用 `/usr/bin/time -v` 包裹每轮最外层 shard launcher，并保存独立日志。GNU time 会给出 wall time、user time、system time、CPU percentage 和最大 RSS。平均实际占用核心数按下式计算：

```text
average_used_cores = (user_seconds + system_seconds) / wall_seconds
quota_utilization = average_used_cores / available_quota_cores * 100%
assigned_utilization = average_used_cores / assigned_cpu_count * 100%
throughput = exact_sample_frames / wall_seconds
```

这里只看主 Python 进程的 `ps %CPU` 是错误的，因为大量 CPU 时间发生在短生命周期的 ffmpeg/ffprobe 子进程中。若新实例安装了 `pidstat`/`mpstat`，可以每秒采样作为补充；没有这些工具时，GNU time 的进程树累计 CPU 时间仍是必填口径。

建议每轮同时保存：

- 完整命令和 git commit。
- 运行前的 affinity、quota 和 CPU fraction。
- shard 到 CPU ID 的分配。
- 精确 Episode 数与帧数。
- 开始/结束 UTC 时间和 wall seconds。
- user seconds、system seconds、average used cores。
- quota utilization、assigned utilization、frames/s。
- 每个 shard 的 return code、accepted/skipped/failed/committed。
- 输出体积和零字节文件数量。
- 运行期间是否有其他任务共享 CPU 或源/目标存储。

将汇总结果写成 CSV 或 JSON，至少包含：

```text
commit,cpu_fraction,shards,repeat,episodes,frames,encoder_threads,
wall_seconds,user_seconds,system_seconds,avg_used_cores,
quota_cpus,quota_utilization_pct,frames_per_second,
failed,zero_byte_files,output_bytes
```

## 9. 轻量验收

每轮只做以下检查，不做全量帧读取：

- launcher 最终判定所有 shard complete。
- 每个 `conversion_report.json` 的 failed 数为 0。
- 每个 `whole_body_joint/conversion_state.json` 的 `pending_episode` 为 null。
- committed Episode 集合等于固定样本集合。
- committed 总帧数等于固定样本精确总帧数。
- 输出目录中不存在零字节文件。
- 编码配置确实为 `encoder_threads=4` 和其余固定参数。

若某轮失败，该轮不能纳入性能均值；先保存日志并解释失败原因，不要直接删除证据后重跑。

## 10. 需要修改和验证的代码

开始实验时预计至少会涉及：

```text
lightworkbench/config.py
scripts/run_merged_shards.py
tests/test_workbench.py
```

可能需要新增 benchmark runner/结果采集脚本，但不要把一次性绝对路径硬编码进库代码。修改要求：

- 为 LeRobot 提供明确的 100% CPU budget 行为。
- 基于真实 affinity CPU ID 分组。
- 保持 `encoder_threads=4`。
- 记录每个 shard 的开始、结束和耗时，避免只记录最终 return code。
- 保留现有断点复用和完整性判断能力。
- 更新受影响测试，并先跑定向测试：

```bash
/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/envs/data_process_pre/bin/python \
  -m pytest tests/test_workbench.py tests/test_convert_cli.py tests/test_shard_merge.py -q
```

不要为了基准实验修改转换数据语义、action schema、视频 codec/CRF/preset，或把 `encoder_threads` 提高到 4 以上。

## 11. 最终交付给用户的结果

实验结束后应提供：

1. 新实例的真实 CPU affinity、quota、实验可用核数。
2. 固定样本清单、精确帧数和选取方法。
3. 每个并行度每轮原始数据及中位数。
4. shard 数对 wall time、frames/s、平均 CPU 核心占用和 quota 利用率的对比。
5. 最优并行度，以及继续增加 shard 后收益下降的证据。
6. 瓶颈判断：CPU、源盘读取、目标盘写入，还是单 Episode/单视频内部串行。
7. 已修改文件、测试结果和推荐的正式转换命令。

不要只报告“CPU 百分比更高”。最终选择应以同样帧数下的 wall time 和 frames/s 为主，CPU 利用率用于解释瓶颈。
