# 运行与复现

转译队友直接读取 [交接数据](../handoff/qingtian/song-package.json)，不需要安装模型。以下供重新分析歌曲时使用。

## 快速运行：基础版

需要 Python 3.11、FFmpeg/ffprobe；当前实测 macOS Apple Silicon，基础版也面向 Linux。

```sh
# macOS：brew install ffmpeg uv
uv venv --python 3.11
uv pip install --python .venv/bin/python -e .

# 用自生成声音先试跑，不需要歌曲文件
.venv/bin/python examples/create_demo.py
.venv/bin/musicprep analyze work/demo.wav --output work/demo --profile baseline

# 查看本地报告（macOS）
open work/demo/report.html
```

基础版生成真实算法估计，不提供神经模型的曲风/情绪标签，也不会把窗口名称伪装成主歌/副歌。

## 神经模型版

按 [安装指南](setup.md) 安装模型依赖与准备权重后：

```sh
.venv/bin/musicprep analyze data/raw/qingtian.flac \
  --output work/qingtian-final --track-id qingtian --config config/neural.json

.venv/bin/musicprep validate work/qingtian-final/song-package.json
open work/qingtian-final/report.html
```

可换成任意本地音频；`--lyrics song.lrc --background background.json` 添加可选文本。配置文件中的相对路径以**配置文件目录**为基准。默认配置采用 CPU，适合先验证；首次运行可能下载模型。

- 原命令再次执行会复用有效缓存；改输入、参数、代码、显式模型文件或 worker 环境会自动失效。
- `--force` 强制重新分析。`--require-complete` 让部分完成返回退出码 2。
- `musicprep doctor` 检查环境；`musicprep report package.json --output report.html --audio song.flac` 重建报告。
- 模型缺失或失败保留实际原因；任务配置 `fallback:false` 可禁用降级。

## 完整配置：Qwen2-Audio + SOFA

在上述神经模型环境基础上，Apple Silicon 可运行：

```sh
.venv/bin/python scripts/setup_qwen.py
.venv/bin/python scripts/setup_sofa.py
.venv/bin/musicprep analyze data/raw/qingtian.flac \
  --output work/qingtian-full --track-id qingtian \
  --config config/full.json --lyrics data/raw/qingtian-lyrics.txt
```

Qwen 使用 6.56 GB 的真实 4-bit 权重，输出按来源保留的曲风、乐器和质感候选；SOFA 用中文演唱模型将无时间歌词对齐到人声。两者各有独立环境，按顺序运行。**SOFA 需要歌词文本，不负责自动听写。** [Qwen 实现](qwen-implementation.md) · [SOFA 实现](sofa-implementation.md)


复跑自动流程生成原始模型估计，不会自动重现《晴天》后续谱面补充与声部复核；其记录见 [复核说明](score-supplement.md)。
