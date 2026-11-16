# 安装与模型准备

实测环境：macOS、Apple M5、16 GB、Python 3.11；核心分析使用 CPU，Qwen 使用 MLX / Metal。基础版可在 macOS/Linux 运行；Windows 原生的进程锁/终止组尚未适配，可使用 WSL。模型首次下载完成后，本地音频分析不会上传歌曲。

## 1. 基础环境

```sh
brew install ffmpeg uv cmake
uv venv --python 3.11
uv pip install --python .venv/bin/python -e .
.venv/bin/musicprep doctor
```

Linux 用系统包管理器安装 FFmpeg、CMake 和编译器。基础版仅需项目基础依赖，使用 `--profile baseline`。

## 2. 神经分析依赖

本机已验证的 Python 依赖解析保存在 `requirements-macos.lock`（Python 3.11 / ARM64 macOS 15+）。先安装它，再安装本项目：

```sh
uv pip install --python .venv/bin/python -r requirements-macos.lock
uv pip install --python .venv/bin/python -e .
```

其他平台可从 extras 解析，但需自行验证对应预编译包：

```sh
uv pip install --python .venv/bin/python -e '.[neural,emotion,tags]'
```

核心固定 PyTorch/torchaudio 2.5.1；Basic Pitch ONNX 使用 `setuptools<81` 兼容 resampy。Essentia 使用带 TensorFlow 推理的 `essentia-tensorflow`，普通 essentia 包不等价。

## 3. All-In-One 段落环境

它依赖旧 NATTEN API，不能直接升级成最新版。先确保前一步 PyTorch、NumPy 已安装：

```sh
uv pip install --python .venv/bin/python cython wheel
uv pip install --python .venv/bin/python --no-build-isolation \
  'madmom @ git+https://github.com/CPJKU/madmom.git@27f032e8947204902c675e5e341a3faf5dc86dae'

# 此编译参数用于本机新版 Apple Clang 与旧 PyTorch 头文件兼容
NATTEN_N_WORKERS=2 CXXFLAGS='-Wno-invalid-specialization' \
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps 'natten==0.17.3'

uv pip install --python .venv/bin/python --no-deps 'allin1==1.1.0'
.venv/bin/python -c 'import allin1; print("All-In-One import OK")'
```

macOS 需要 Command Line Tools。Linux 根据其编译器选择参数；不要照抄 Apple 专用 flag。模型配置可用 `harmonix-fold0` 单折，默认 `harmonix-all` 为八折集成。无法安装时把 sections.backend 改成 baseline，输出会明确标注未知段落标签。

## 4. 模型文件

```sh
.venv/bin/python scripts/setup_models.py --all
```

该命令准备 BTC、Music2Emotion 源码与权重、Beat This!、Essentia；源码锁定 commit，下载文件按 SHA-256 校验。已存在且匹配的文件直接复用；不会覆盖被修改的源码仓库。

- `--core`：BTC + Beat This!。
- `--emotion`：Music2Emotion；MERT 首次推理由上游加载，实测 snapshot 见描述模块文档；严格离线复现需保留该缓存并禁用下载。
- `--tags`：Essentia EffNet 与曲风/乐器分类头。
- Demucs、All-In-One 使用上游模型缓存，首次推理会下载。All-In-One 固定 HF revision 并记录权重哈希；Demucs 当前记录模型名/软件版本/参数，尚未对隐式缓存权重单独锁定哈希，复现时应保留该缓存。
- Beat This! 官方地址本轮返回 500，采用已核对 SHA-256 的权重镜像，不运行镜像仓库代码。

`config/model-sources.json` 是下载来源清单。模型、分轨和输出目录均已加入 Git 忽略规则。第三方模型有各自许可，特别是 Essentia 及部分预训练模型的商业使用条件需按上游许可处理；本仓库不会重新授权它们。

## 5. 一条命令处理

```sh
.venv/bin/musicprep analyze data/raw/qingtian.flac \
  --output work/qingtian-final --track-id qingtian --config config/neural.json
```

无需处理中人工干预。可选 `--lyrics song.lrc`、`--background background.json`；背景 JSON 格式见 [文本模块文档](descriptions-implementation.md)。未提供文本时标 skipped。

所有配置相对路径以配置文件所在目录为准；`config/neural.json` 内 `../models/...` 指仓库的 models 目录。模型需要另一个 Python 环境时，在对应任务或描述模型中设置 `python`；环境版本参与缓存指纹，虚拟环境符号链接会保留。

## 6. 完整配置：Qwen2-Audio + SOFA

Apple Silicon 上用独立环境，避免 MLX 与旧研究模型依赖冲突：

```sh
.venv/bin/python scripts/setup_qwen.py
.venv/bin/python scripts/setup_sofa.py

.venv/bin/musicprep analyze data/raw/qingtian.flac \
  --output work/qingtian-full --track-id qingtian \
  --config config/full.json --lyrics data/raw/qingtian-lyrics.txt
```

Qwen 直接下载社区 4-bit 权重（约 6.56 GB），按固定 revision 和 SHA-256 校验；不是下载原始 7B 权重再量化。SOFA 下载官方中文演唱模型，自动把中文和支持的唱名转换成音素、对齐整首人声，不需要手工切句或时间戳。模型按顺序运行；本机 16 GB 环境不要同时加载多个大模型。

SOFA 是强制对齐器，**需要参考歌词，不负责听写或联网检索**。歌词缺失时文本任务 skipped；未知语言/读音失败会保留原因，不猜时间。参考文本可附 `source_metadata` 文件，记录来源 URL 与文本 SHA-256；来源文件不提供时间。详细配置：[Qwen](qwen-implementation.md) · [SOFA](sofa-implementation.md)。

`setup_models.py --all` 只安装核心模型组；两个新增安装脚本单独执行。隐式上游缓存发生变化时可用 `--force` 重算。

## 常见结果

- `partial` 且日志显示具体 fallback：模型未装好/加载失败，或使用了不确定的器乐候选；先看任务 provenance 与 warnings。
- 歌词为空：没有提供文本，不代表程序自动知道歌曲歌词。
- 模型错误不会产生伪造结果。下载/依赖初始化失败时默认继续其他模块，严格模式可设置 `fallback:false`。
- 报告打开后音频未载入：点击“选择本地音频”，选择与分析一致的版本；本地报告不上传文件。
