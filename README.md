# 极简高性能剪切工作台

独立的 Capture Episode 浏览与逐帧同步剪切服务。生产运行只依赖 Python、FastAPI 和 FFmpeg，不需要 Node、数据库、后台索引或定时任务。

## 启动

```bash
source /home/ubuntu/miniconda3/bin/activate
conda activate data_process
/home/ubuntu/workspace/junxi/data_process_tools_light/start.sh
```

默认地址为 `http://127.0.0.1:8332`。可通过 `HOST` 和 `PORT` 修改监听地址。页面中的 cleaned 输出根目录必须与源根目录完全分离；原始数据永不修改。

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

确认报告后去掉 `--preflight-only` 执行转换。命令递归发现 `episode_数字`，按 cleaned 根下的任务相对路径镜像输出；每个任务是独立 LeRobot dataset。`--namespace autolife` 是默认仓库命名空间，`--task-language english|source` 选择训练任务文本（默认将英文 `task_title` 的下划线改为空格），`--require-source` 要求原始 `sourceRoot` 仍可访问且 `sourceToken` 复验通过。

转换始终增量追加，不覆盖已有 dataset。每次运行原子更新根目录 `conversion_summary.json` 和任务内 `conversion_report.json`。任一 Episode 被拒绝或转换失败时命令返回非零，但其他独立 Episode/任务仍会继续。已有源 Episode 缺失、改名、内容变化，或配置/视频 schema 不一致时会停止该任务，避免混合不同修订。旧 14 维 action dataset 不兼容，必须选择新的输出根。

cleaned Episode 必须有可复验的 `CUT_INFO.json`。验证包括审计路径、模式、删除区间、帧数、完成时间、排除 `CUT_INFO.json` 后的输出指纹、manifest JSONL、连续帧号、有限数值、EEF/关节/control 结构，以及所有有效视频流的逐帧解码数、FPS、尺寸和引用 containment。`rgbd_head_color`、`hand_left`、`hand_right` 是必需彩色流；深度和其他有效相机流也全部写入 LeRobot。旧的多保留区间输出若没有 `timestampRewriteVersion` 会以 `unverified_nested_timestamp_stitching` 跳过，需从工作台重新剪切生成，转换器不会改写 cleaned 数据。

新 action 固定为 20 维，每只手 10 维，顺序如下：

```text
left:  delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz,
       gripper_speed, gripper_force,
       gripper_speed_valid, gripper_force_valid
right: 同上
```

EEF delta 使用帧 `t -> t+1`；夹爪 speed/force 读取帧 `t` 的 `control.commands.SET_*` 原值，不猜测单位或归一化。缺失命令写数值 `0` 且 valid 为 `0`；真实零命令写数值 `0` 且 valid 为 `1`。最后一帧不产生 action，夹爪反馈位置仍保留在 observation。

一次离线只读审计已扫描 `raw/289` 与 `raw/290` 的 10,088 个 Episode，其中 9,060 个含显式夹爪命令，确认控制量真实存在。该扫描约涉及 72.2 GiB，常规 CI 不重复执行。
