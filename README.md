# 极简高性能剪切工作台

独立的 Capture Episode 浏览与逐帧同步剪切服务。生产运行只依赖 Python、FastAPI 和 FFmpeg，不需要 Node、数据库、后台索引或定时任务。

## 启动

```bash
source /home/ubuntu/miniconda3/bin/activate
conda activate data_process
/home/ubuntu/workspace/junxi/data_process_tools_light/start.sh
```

默认地址为 `http://127.0.0.1:8332`。可通过 `HOST` 和 `PORT` 修改监听地址。页面中的 cleaned 输出根目录必须与源根目录完全分离；原始数据永不修改。

## 多人并行使用

本工作台支持多人对同一批源数据中的不同 Episode 并行操作。推荐只启动一个服务，让所有人的任务进入同一个共享队列；默认并发数为 2，可同时处理两个不同的输出 Episode。

不同使用者不强制使用不同浏览器。同一浏览器的多个标签页也能分别操作，但会共享“操作人员”、cleaned 输出根目录和并发数等浏览器本地配置。为避免姓名或配置相互影响，建议使用不同浏览器、不同浏览器 Profile，或者普通窗口与无痕窗口。每位使用者应填写不同的“操作人员”姓名，并提前按 Episode 编号或目录划分工作范围。

两人在同一台电脑上操作时，启动一次服务后，都访问：

```text
http://127.0.0.1:8332
```

如果使用者实际位于不同电脑，第二台电脑不能访问 `127.0.0.1`，因为该地址始终指向当前电脑自身。应在运行服务的电脑上监听局域网地址：

```bash
cd 2026_cz/data_autopro_tools_light
HOST=0.0.0.0 PORT=8332 ./start.sh
```

在服务端查询局域网 IP：

```bash
hostname -I
```

假设服务端 IP 为 `192.168.1.100`，其他电脑应访问 `http://192.168.1.100:8332`。如果无法连接，请确认两台电脑网络互通，并检查防火墙是否允许 TCP 端口 `8332`。如果服务运行在远程服务器或容器中，还需要配置相应的端口转发。

使用时请注意：

- 队列和并发数是所有页面共享的全局状态；建议保持默认并发数 2。
- 系统允许多人同时查看同一个 Episode，但不会在打开时自动“领取”或锁定它，因此需要人工划分不同 Episode。
- 相同输出 Episode 已在排队或运行时，重复提交会被拒绝；已完成的 Episode 再次提交会询问是否覆盖，除非明确返工，否则不要确认覆盖。
- 不要启动两个后端进程并让它们共用同一个 cleaned 输出根目录。进程之间不共享任务锁和 CSV 写入锁，可能发生输出覆盖竞争或 `CUT_HISTORY.csv` 写入冲突。

## 数据约定

- Episode 目录名为 `episode_数字`，包含 `manifest.jsonl`、`task_meta.json` 和 `videos/`。
- 目录浏览保持原始层级；只有当前目录的直接子目录为 Episode 时才切换到 Episode 列表。
- 视频引用来自 manifest 每帧的 `videos.<stream>.path`。
- 删除区间使用半开范围 `[start, end)`；后端会裁边、排序并合并。
- 输出位于 cleaned 根下与源根一致的相对路径，含 `CUT_INFO.json`。

## 任务队列

- 任务仅缓存在当前服务进程内，最多排队 256 个；服务重启不会恢复或重放任务。
- 默认和推荐并发为 2，可在页面队列面板调整为 1–4。
- 相同输出 Episode 在排队或运行时不能重复提交；不同目标可并行发布。
- 所有运行任务共享 70% CPU 硬预算，每个任务启动时获得固定 FFmpeg slot 配额。

- cleaned 根的 `CUT_HISTORY.csv` 只追加历史记录；覆盖不会改写旧记录。

## 测试

```bash
python -m pip install -r requirements-dev.txt
python -m playwright install chromium
pytest
```

## LeRobot v3 增量转换

转换器是独立命令行工具，不进入 Web 服务或任务队列。Web 服务仍可使用 Python 3.10 和原有依赖；转换环境使用 Python 3.12。建议在单独虚拟环境中安装完整 FFmpeg（必须包含 `ffmpeg`、`ffprobe`、H.264、H.265/x265、FFV1 和 SVT-AV1 支持）及 CPU Torch：

```bash
python3.12 -m venv .venv-lerobot
. .venv-lerobot/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
  'torch>=2.7,<2.12.0' 'torchvision>=0.22,<0.27'
python -m pip install -e '.[lerobot]'
```

可选依赖将 LeRobot 固定在 commit `adccdea1cfbec83ed98263feb7e59f7d047c5692`。也可在先安装 CPU Torch 后使用 `requirements-lerobot.txt` 安装相同环境。先运行只读预检：

```bash
data-autopro-light convert \
  --input-root /path/to/cleaned \
  --output-root /path/to/lerobot \
  --preflight-only
```

确认报告后去掉 `--preflight-only` 执行转换。默认 `--action-mode both` 从同一批 cleaned Episode 生成两套数据：

`--include-episode <input-root-relative-path>` 可重复使用，只转换精确列出的 Episode；重复、越界或未发现的路径会在写数据前拒绝。指定路径时转换器直接解析这些 Episode，不再扫描输入根下的其他 Episode。预检查默认在 CPU 预算内按 Episode 并行，也可通过 `--preflight-workers` 调低并发。视频参数可通过 `--video-codec`、`--video-crf`、`--encoder-preset`、`--encoder-threads`、`--video-workers` 和 `--video-encoding-mode` 配置，并写入 summary；影响输出内容的参数也写入 conversion state。`--video-workers` 只控制调度，不改变输出，因此可在增量续跑时调整。值不超过 3 时控制每组三路视频的读取和校验；streaming 模式设为 4–6 时，训练视频和辅助视频两组会同时编码，并在两组完成后统一提交。并行 H.264 示例：

```bash
data-autopro-light convert \
  --input-root /path/to/cleaned \
  --output-root /path/to/lerobot \
  --include-episode date/site/task/episode_000001 \
  --video-codec h264 --video-crf 18 --encoder-preset fast \
  --encoder-threads 6 --video-workers 3 --video-encoding-mode parallel
```

离线转换也可使用 `--video-encoding-mode streaming`，直接把解码帧送入每路视频编码器，省去 PNG/TIFF 中间文件。为避免 LeRobot 在编码队列满时静默丢帧，转换器会把每路队列扩展到能够容纳完整 Episode；因此该模式速度更高，但峰值内存随 Episode 帧数、分辨率和并行 Episode 数线性增长。超长 Episode 或内存受限机器应继续使用 `parallel`。

```text
OUT/whole_body_joint/<relative-task>  # 三路训练视频的唯一 owner
OUT/body_joint_eef/<relative-task>    # 独立 data/meta，videos/auxiliary 为相对软链接
OUT/.bundle/<relative-task>           # 每 Episode、每 mode 的提交 ledger
```

### 跨日期统一合并

持续增长的 cleaned 根目录应使用 `convert-merged`。它不会再按日期、场景或任务创建子 dataset，而是把所有兼容 Episode 追加到两个统一的 LeRobot dataset：

```bash
data-autopro-light convert-merged \
  --input-root /inspire/qb-ilm/project/robot-decision/public/demo2/raw_08_29 \
  --video-codec h264 --video-crf 18 --encoder-preset fast \
  --encoder-threads 6 --video-workers 3 --video-encoding-mode parallel
```

批量转换可先运行只读预检，再用 `scripts/run_merged_shards.py` 将连续 Episode 分片绑定到实际可用 CPU。脚本默认使用 `min(CPU affinity, cgroup quota)` 的完整预算，不再施加 70% 上限，并在 summary 中记录平均使用核心数和 quota 利用率。`--cpu-binding` 支持 `exclusive`、`numa-shared` 和 `global-shared`；独占绑定通常最稳定，共享模式适合存在明显长尾、需要进程动态借核的负载。当前 55 核环境的 1500 帧基准推荐：

```bash
python scripts/run_merged_shards.py \
  --report /path/to/preflight/conversion_report.json \
  --input-root /path/to/cleaned \
  --work-root /path/to/sharded-output \
  --shards 13 --cpu-binding exclusive \
  --video-encoding-mode streaming \
  --encoder-threads 2 --video-workers 4 --preflight-workers 1
```

### 并行全量转换与派生

下面的完整流程先生成 `whole_body_joint` 分片，按 Episode 顺序合并 owner dataset，最后派生不重复编码视频的 `body_joint_eef` dataset。`RUN_ROOT` 必须是新的输出目录；需要续跑时保持路径、分片数和所有编码参数不变。

```bash
cd '/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/2026_cz/data_autopro_tools_light copy'

export PYTHON='/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/miniconda3/envs/data_process/bin/python'
export INPUT_ROOT='/inspire/qb-ilm/project/robot-decision/public/demo2/raw_08_29'
export RUN_ROOT='/inspire/ssd/project/robot-decision/laijunxi-CZXS25230141/2026_cz/lerobot_data_08_29_optimized'
export PREFLIGHT_ROOT="$RUN_ROOT/preflight"
export SHARD_ROOT="$RUN_ROOT/shards"
export FINAL_ROOT="$RUN_ROOT/final"
export SHARD_COUNT=13

mkdir -p "$RUN_ROOT"

"$PYTHON" -m lightworkbench.cli convert-merged \
  --input-root "$INPUT_ROOT" \
  --output-root "$PREFLIGHT_ROOT" \
  --preflight-only \
  --action-mode whole_body_joint \
  --video-codec h264 \
  --video-crf 18 \
  --encoder-preset fast \
  --encoder-threads 2 \
  --video-workers 4 \
  --video-encoding-mode streaming \
  --preflight-workers 13 \
  2>&1 | tee "$RUN_ROOT/preflight.log"

"$PYTHON" scripts/run_merged_shards.py \
  --report "$PREFLIGHT_ROOT/conversion_report.json" \
  --input-root "$INPUT_ROOT" \
  --work-root "$SHARD_ROOT" \
  --shards "$SHARD_COUNT" \
  --cpu-binding exclusive \
  --video-encoding-mode streaming \
  --encoder-threads 2 \
  --video-workers 4 \
  --preflight-workers 1 \
  --python "$PYTHON" \
  2>&1 | tee "$RUN_ROOT/convert_shards.log"

"$PYTHON" - <<'PY' 2>&1 | tee "$RUN_ROOT/merge.log"
import os
from pathlib import Path

from lightworkbench.shard_merge import merge_whole_body_joint_shards

shard_root = Path(os.environ["SHARD_ROOT"])
final_root = Path(os.environ["FINAL_ROOT"])
shard_count = int(os.environ["SHARD_COUNT"])
result = merge_whole_body_joint_shards(
    [shard_root / f"shard-{index:02d}" for index in range(shard_count)],
    final_root / "whole_body_joint",
)
print(result)
PY

"$PYTHON" -m lightworkbench.cli convert-merged \
  --input-root "$INPUT_ROOT" \
  --output-root "$FINAL_ROOT" \
  --action-mode body_joint_eef \
  --video-codec h264 \
  --video-crf 18 \
  --encoder-preset fast \
  --encoder-threads 2 \
  --video-workers 4 \
  --video-encoding-mode streaming \
  --preflight-workers 13 \
  2>&1 | tee "$RUN_ROOT/derive_body_joint_eef.log"
```

最终训练入口为 `$FINAL_ROOT/whole_body_joint` 和 `$FINAL_ROOT/body_joint_eef`。分片转换可原路径重跑并复用已完成 shard；merge 的目标目录必须不存在，避免覆盖已经发布的数据。

省略 `--output-root` 时默认写入 `<input-root>/lerobot_data`：

```text
raw_08_29/lerobot_data/
├── whole_body_joint/  # 统一 23-D dataset，同时持有唯一视频实体和辅助视频索引
├── body_joint_eef/    # 统一 21-D dataset，videos/auxiliary 相对链接到 owner
└── .bundle/           # 以源相对路径为身份的双模式增量 ledger
```

这里的“统一 dataset”是一个训练入口和一套连续全局 Episode index；其 Parquet 和 MP4 仍由 LeRobot 按 chunk/file 标准分片，不会制造一个无法增量恢复的超大单文件。每个 Episode 保留自己的规范化英文 task。不同日期或任务中重复出现的 `episode_000024` 不会冲突，因为增量身份使用 `日期/场景/任务/episode` 的完整源相对路径；原始数字 ID 仍保存在审计字段中。

每天源目录增加数据后重复运行同一条命令即可。已提交 Episode 是 no-op，只有新的相对路径会编码并追加；源数据被删除、改名或原地修改时命令会拒绝继续。编码参数是 conversion state 的一部分，因此以后每次运行必须保持 `codec`、CRF、preset、线程数和 encoding mode 完全一致。首次转换或日常全量发现不要使用 `--include-episode`；该选项主要用于预检和 smoke，增量重跑时必须同时选入所有历史已提交 Episode。

如需放到其他位置，可显式传入 `--output-root /path/to/lerobot_data`。输出根可以位于 cleaned 根内部，发现器会排除该输出目录；它不能等于输入根，也不能包含输入根。

训练端直接读取统一根目录：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

body_joint_eef = LeRobotDataset(
    repo_id="autolife/body_joint_eef",
    root="/inspire/qb-ilm/project/robot-decision/public/demo2/raw_08_29/lerobot_data/body_joint_eef",
    video_backend="pyav",
    return_uint8=True,
)
```

可选值为 `both`、`whole_body_joint` 和 `body_joint_eef`。单独执行 `body_joint_eef` 只用于补写或修复派生数据，要求对应 owner 已经完整提交；首次转换应使用 `both`。训练任务文本固定为英文 `task_title` 去下划线并压缩空白，例如 `pick_the_egg_tart_from_tray` 写成 `pick the egg tart from tray`。中文 `task_description` 只出现在转换报告中。`--require-source` 可要求原始 `sourceRoot` 仍可访问且 `sourceToken` 复验通过。

两种 action 合同如下：

```text
body_joint_eef (21-D):
  lower body [Ankle, Knee, Waist_Pitch, Waist_Yaw]
  left  [local delta_xyz(3), local delta_rotvec(3), gripper]
  right [local delta_xyz(3), local delta_rotvec(3), gripper]
  neck  [Yaw, Pitch, Roll]

whole_body_joint (23-D):
  lower body(4), left arm joints(7), right arm joints(7),
  left/right gripper(2), neck [Yaw, Pitch, Roll](3)
```

普通帧使用下一状态监督：whole-body 为 `qpos[t+1]`；hybrid 的 lower body、neck 和 gripper 同样取下一帧，EEF 使用 `current_eef_pose[t] -> current_eef_pose[t+1]` 的局部增量。夹爪严格读取 `joints.position[18]` 和 `[19]`，不读取 speed/force，也不裁剪、环绕或归一化。每个 Episode 的全部 N 帧都会保留；末帧 joint/gripper/neck hold，EEF delta 为零。两套数据共同保留 38-D `observation.state`、velocity、torque、current/target EEF、height 和 source frame/timestamp 字段。neck 数值列仍是源 `[20,21,22]`，只把轴标签修正为 `Yaw, Pitch, Roll`。

标准训练图像 feature 严格只有：

```text
observation.images.rgbd_head_color
observation.images.hand_left
observation.images.hand_right
```

深度、鱼眼及其他有效流由 owner 编码到 `auxiliary/videos/`，并在 `auxiliary/index.parquet` 记录 Episode、路径、codec/pixel format、尺寸、FPS、帧数、depth 编码参数和统计量。它们不进入初始 `meta/info.json`，标准 LeRobot reader 不会自动解码。需要正式训练某个辅助流时可执行：

```bash
data-autopro-light promote-aux-videos \
  --owner-root OUT/whole_body_joint[/<relative-task>] \
  --stream rgbd_head_depth head_left head_right
```

提升只创建到现有辅助视频的相对链接并同步 owner/hybrid 的 `info.json`、Episode locator 和统计 metadata，不重新编码。提升被视为该 bundle 的终态；如还要追加 Episode，应先完成追加，再执行提升。

转换始终增量执行。owner 已提交但 hybrid 派生失败时，下一次运行只重建低维 hybrid 数据，不重新编码 owner 视频。重复输入是 no-op；源 Episode 缺失、改名、内容变化，或三路训练视频 schema 不一致时会停止对应任务。旧 v1/v2（包括 14-D 和 20-D）输出不能原位追加，必须使用新的输出根。

cleaned Episode 必须有可复验的 `CUT_INFO.json`。验证覆盖审计路径、裁剪区间、输出指纹、manifest、连续帧、有限 joint/EEF 数值和所有有效视频流的逐帧解码。旧的多保留区间输出若没有 `timestampRewriteVersion` 会以 `unverified_nested_timestamp_stitching` 跳过。

相对软链接要求整个 `OUT` bundle 一起移动或使用保留链接的复制方式（例如 `rsync -a`）。只复制 `body_joint_eef` 子目录会断链；发布到 Hugging Face Hub 或单目录归档前应先物化链接。标准读取示例：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset(
    repo_id="autolife/whole_body_joint__task",
    root="OUT/whole_body_joint/<relative-task>",
    video_backend="pyav",
    return_uint8=True,
)
print(dataset[0]["action"], dataset[0]["observation.images.rgbd_head_color"].shape)
```

对已生成 bundle 运行真实 LeRobot schema、软链接、视频、移动和 auxiliary promotion 集成测试：

```bash
LEROBOT_DUAL_BUNDLE=/path/to/OUT pytest -q tests/test_lerobot_real_integration.py
```
