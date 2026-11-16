# 描述路径与可选歌词处理

`descriptions` 按结构段落分析；无可用段落时使用 30 秒窗口。输出的分析范围与应用范围采用原音频秒数。`lyrics_context` 只处理提供的歌词和背景资料；缺失时自动跳过，不生成虚构歌词、歌曲故事或作者意图。

## 当前实现

| 分支 | 输入 → 输出 | 实现与边界 |
|---|---|---|
| 声学质感 | 混音片段 → RMS、谱质心、谱平坦度、包络变异、起音密度 | SciPy + NumPy；明亮/偏暗等只是带来源的阈值候选，不由响度猜效价、曲风或乐器 |
| 曲风、乐器 | 混音片段 → 标签及分数 | Essentia Discogs EffNet 嵌入 + Discogs400 / MTG Jamendo 分类头；读取各模型 JSON 的标签和张量名 |
| 情绪 | 混音片段 → 效价、唤醒度、情绪候选 | Music2Emotion 官方 `Music2emo`，保留其 1–9 输出尺度；推理窗口内的汇总估计，不当成逐帧真值 |
| 音频描述 | 混音片段 → 曲风、乐器、质感候选 | 可选 Qwen2-Audio-7B-Instruct；结构化 JSON 校验，默认禁止自动下载；不生成背景事实 |
| 带时间歌词 | LRC → 句级时间线 | 解析时间戳和 offset；同句不同时间保留；空时间戳保留清屏边界；句尾默认取下一时间边界，明确标记估计 |
| 无时间歌词 | TXT → 文本；配置 SOFA 后 → 句级、词级时间线 | SOFA 使用给定歌词强制对齐；中文自动转拼音；对齐词序不一致即拒绝，不伪造时间 |
| 语义与背景 | 歌词、TXT/MD/JSON 介绍 → 原文摘录、主题/意象候选、来源 | 规则抽取保留证据；背景标记 `unverified`，未接入外部事实核验 |

基础环境无需模型即可完成声学描述。未配置模型的曲风、乐器留空，效价和唤醒度为 `null`；整个任务标记 `partial`。这意味着任务自动执行结束，不代表所有音乐信息均已识别。

## 配置

以下为单个 `descriptions` 任务参数，路径需换成实际绝对路径：

```json
{
  "backend": "auto",
  "fallback": true,
  "window_seconds": 30,
  "model_backends": {
    "music2emo": {
      "python": "/project/.venv/bin/python",
      "repo": "/project/models/vendor/Music2Emotion",
      "weights": "/project/models/vendor/Music2Emotion/saved_models/J_all.ckpt",
      "allow_download": false,
      "threads": 2,
      "threshold": 0.5,
      "timeout_seconds": 3600
    },
    "essentia": {
      "python": "/project/.venv-essentia/bin/python",
      "embedding": "/project/models/essentia/discogs-effnet-bs64-1.pb",
      "genre": {
        "graph": "/project/models/essentia/genre_discogs400-discogs-effnet-1.pb",
        "metadata": "/project/models/essentia/genre_discogs400-discogs-effnet-1.json",
        "threshold": 0.2,
        "top_k": 5
      },
      "instruments": {
        "graph": "/project/models/essentia/mtg_jamendo_instrument-discogs-effnet-1.pb",
        "metadata": "/project/models/essentia/mtg_jamendo_instrument-discogs-effnet-1.json",
        "threshold": 0.2,
        "top_k": 5
      },
      "timeout_seconds": 1800
    }
  }
}
```

`backend` 支持 `auto`、`baseline`、`models` 或某个模型名称。`auto` 仅执行已配置模型，未配置模型不会偷偷下载。`allow_download: true` 才允许首次从 Hugging Face 获取模型。阈值与 top-k 是产品候选筛选参数，不是经过校准的可信度门槛。

CLI 配置接受相对路径，以配置文件所在目录为基准；进入独立模型工作目录前已转为绝对路径。直接调用适配器的 Python API 时应传绝对路径。

Qwen2-Audio 可在 `model_backends.qwen2_audio` 配置 `python`、`model_id`（本地路径或 HF ID）、`allow_download`、`device_map`、`max_new_tokens`、`timeout_seconds`。7B 模型不适合与多个大模型同时挤在 16 GB 机器中；本项目不默认加载它。模型生成的候选标签与其他模型一致，均不视为事实标注。

Music2Emotion 需要仓库自带的 `saved_models/J_all.ckpt`、整个 `inference/data`，另需 `m-a-p/MERT-v1-95M`。主要依赖：PyTorch、torchaudio、Transformers 4.44、pytorch-lightning、music21、pretty_midi、mir_eval、hydra-core、gradio、librosa、NumPy < 2。实际环境建议由项目统一锁定，避免与其他分析模型争用依赖版本。

Essentia 必须具有 `TensorflowPredictEffnetDiscogs` 和 `TensorflowPredict2D`，普通不含 TensorFlow 的构建不足以运行。模型下载：

- [Discogs EffNet](https://essentia.upf.edu/models/music-style-classification/discogs-effnet/discogs-effnet-bs64-1.pb)
- [Discogs400 分类头](https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb) / [元数据](https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json)
- [乐器分类头](https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb) / [元数据](https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.json)

可选 SOFA 参数位于 `lyrics_context` 任务配置：

```json
{
  "backend": "auto",
  "fallback": true,
  "sofa": {
    "python": "/project/.venv-sofa/bin/python",
    "repo": "/project/models/vendor/SOFA",
    "checkpoint": "/project/models/sofa/model.ckpt",
    "dictionary": "/project/models/vendor/SOFA/dictionary/opencpop-extension.txt",
    "text_type": "mandarin",
    "timeout_seconds": 1800
  }
}
```

SOFA 环境需要其官方依赖和 `pypinyin`；`text_type` 也可为 `pinyin`。默认中文路径拒绝混入英文或数字的歌词，避免用错误字典强行对齐。后台环境缺依赖时保留原文、时间留空、返回警告；不要求用户在处理中确认。自动转写与检索背景不在本模块中隐式发生。

## 数据约定与容错

- 标签：`{"label":"acousticguitar","score":0.8,"source":"essentia","status":"candidate"}`；启发式和生成式标签无可靠分数时不编造 `score`。
- 情绪：`{"valence":5.1,"arousal":4.3,"scale":[1,9],"moods":[],"source":"music2emo","status":"estimated"}`；没有估计时两轴为 `null`。
- 段落：`section_id` 引用结构段落；固定窗口回退时为 `null`。每条保留 `analysis_start/end`、`applies_start/end`。
- 每个模型由独立子进程运行，一次加载后遍历片段。输出保存于独立任务目录，保留日志。单段失败不会丢掉其他片段；超时/依赖缺失可自动降级。
- Music2Emotion 上游会删除相对工作目录的 `temp_out` 和 `output`；适配器使用唯一临时工作目录，只链接仓库的 `inference` 资源，避免影响项目与其他任务。
- 语义抽取保留 `lyric_id`、原词证据、来源行号；重复副歌按各自出现时间记录。
- LRC offset 采用常见的“正数提前、负数延后”约定：`实际秒数 = 时间戳 - offset / 1000`；例如 `[offset:500]` 下 `[00:02]` 对应 1.5 秒。依据：[Liricle](https://github.com/mcanam/liricle#offset-gettersetter)、[Paroles](https://github.com/Clarkkkk/paroles#lyrics)。空歌词时间戳只结束上一句，不生成空语义记录。
- 所有模型结果拒绝 NaN、Infinity、越界情绪数值、重复片段索引。段落 ID 从不作为输出文件路径。
- `fallback: false` 会将模型进程失败或输入文件错误抛给任务调度器；单个模型片段的拒绝仍以警告表示。

## 验证

2026-11-16 已在 macOS ARM64、Python 3.11、PyTorch/torchaudio 2.5.1、Transformers 4.44.2 环境跑通 Music2Emotion 官方检查点与 MERT 的真实 8 秒音频冒烟测试，输出有限的 1–9 效价/唤醒度及情绪候选。同一片段已跑通 Essentia Discogs EffNet、Discogs400 与 MTG Jamendo 乐器模型，输出带分数的曲风和乐器候选。8 秒测试验证接入，不证明音乐判断准确；正式处理按结构段落或 30 秒窗口分析。另已对本地《晴天》完整音频跑通 9 个基础声学窗口。Qwen2-Audio、SOFA 的真实模型尚需在对应依赖与权重齐全的环境验证。

基础单元测试覆盖：纯音频谱、静音不被误标成情绪、越界段落、路径穿越 ID、单段模型失败、非有限模型输出、重复 LRC 句子、非法时间戳、无时间文本、背景来源、坏 JSON、SOFA HTK 对齐词序核验。SOFA 测试使用模拟进程输出验证适配协议，不能替代真实模型准确率测试。

```bash
python -m unittest tests.test_descriptions tests.test_lyrics -v
```

方法来源：[Music2Emotion](https://github.com/AMAAI-Lab/Music2Emotion)、[Essentia 模型](https://essentia.upf.edu/models.html)、[Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio)、[SOFA](https://github.com/qiuqiao/SOFA)。模型权重和上游代码各自遵循原项目许可，不随本项目一起重新授权。

## 本地验证使用的公开模型版本

| 项目 | 来源与版本 |
|---|---|
| Music2Emotion 源码 | [commit c5f7984f78d9928b841f25bd05a47377c12c58ca](https://github.com/AMAAI-Lab/Music2Emotion/tree/c5f7984f78d9928b841f25bd05a47377c12c58ca) |
| MERT-v1-95M | [HF snapshot 12af15fef9d0ac838c3f475bfbbf26d2060dd4f5](https://huggingface.co/m-a-p/MERT-v1-95M/tree/12af15fef9d0ac838c3f475bfbbf26d2060dd4f5)；首次从官方仓库下载，后续使用 Hugging Face 本地缓存 |
| Essentia | `essentia-tensorflow==2.1b6.dev1389`，Python 3.11 macOS ARM64 wheel |

本次下载文件 SHA-256：

```text
J_all.ckpt                                    deaceb291f7974deb688167d3639b7f6eb66eb0715824668618393777d1e07a5
discogs-effnet-bs64-1.pb                        3ed9af50d5367c0b9c795b294b00e7599e4943244f4cbd376869f3bfc87721b1
genre_discogs400-discogs-effnet-1.pb            3885ba078a35249af94b8e5e4247689afac40deca4401a4bc888daf5a579c01c
genre_discogs400-discogs-effnet-1.json          2d367319d9b782ffa10f69abf0e805b3ac4e10899025e5bdbaceda3919b243e0
mtg_jamendo_instrument-discogs-effnet-1.pb      2e8c3003c722e098da371b6a1f7ad0ce62fac0dcfc09c7c7997d430941196c2a
mtg_jamendo_instrument-discogs-effnet-1.json    7d02204c6451b5615e2968ec6364bbae3b915c886e608f05f00d3a38dc5177c4
```

适配器在结果 provenance 中记录已配置权重/元数据哈希与可获得的源码 commit。MERT 上游内部下载行为仍由该源码控制；严格离线复现需保留上述 snapshot 并设置 `allow_download: false`。

已知上游限制：该 Music2Emotion commit 的 `music2emo.py` 在调性处理中先将 `key_signature` 清为主音名，随后再用它读取大小调，可能使模式特征回落到 `major`（402–412 行）。当前适配器保留官方检查点的推理行为，没有未验证地改写上游模型逻辑；情绪结果仍需听障参与者评估，不能用运行成功代替效度验证。其 `utils/mir_eval_modules.py` 的现行实现已支持短于 10 秒的 CQT 输入，旧的未初始化版本仅存在于注释中。
