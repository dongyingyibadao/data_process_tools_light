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
