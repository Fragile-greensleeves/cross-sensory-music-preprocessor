# 开源项目与方法来源

以下为方法选型来源，研究记录日期 2026-11-16。链接不是版本锁；模型集成时记录实际 commit、安装环境及权重哈希。论文性能不能直接作为本项目效果。

| 工作 | 项目 | 论文/说明 | 使用边界 |
|---|---|---|---|
| 音频准备 | [FFmpeg](https://github.com/FFmpeg/FFmpeg) | [ffprobe](https://ffmpeg.org/ffprobe.html) | 解码、探测，不提取音乐语义 |
| 声源分离接口 | [audio-separator](https://github.com/nomadkaraoke/python-audio-separator) | README 模型清单 | 固定实际模型，不依赖随版本变化的默认模型 |
| 四轨分离模型 | [Demucs](https://github.com/facebookresearch/demucs) | [Hybrid Transformers](https://arxiv.org/abs/2211.08553) | 原仓库不再维护；四轨不是四类音乐信息 |
| 音符识别 | [Basic Pitch](https://github.com/spotify/basic-pitch) | [ICASSP 2022](https://arxiv.org/abs/2203.09893) | 单一声部更合适；不能自动保证主旋律归属 |
| 连续基频 | [torchcrepe](https://github.com/maxrmorrison/torchcrepe) | [CREPE](https://arxiv.org/abs/1802.06182) | 周期性/静音需过滤，不适合直接解释所有复音 |
| 器乐显著旋律 | [Essentia](https://github.com/MTG/essentia) | [PredominantPitchMelodia](https://essentia.upf.edu/reference/streaming_PredominantPitchMelodia.html) | 显著旋律估计，不一定是设计者希望的主奏 |
| 和弦识别 | [BTC](https://github.com/jayg996/BTC-ISMIR19) | [ISMIR 2019](https://arxiv.org/abs/1907.02698) | 需大词表模式；依赖与权重实际加载待验证 |
| 和弦编码/评估 | [mir_eval](https://github.com/mir-evaluation/mir_eval) | [chord API](https://mir-eval.readthedocs.io/latest/api/chord.html) | 标签解析，不从音频识别和弦 |
| 节拍 | [Beat This!](https://github.com/CPJKU/beat_this) | [ISMIR 2024](https://arxiv.org/abs/2407.21658) | 拍点/第一拍不等于完整拍号 |
| 段落结构 | [All-In-One](https://github.com/mir-aidj/all-in-one) | [ISMIR 2023](https://arxiv.org/abs/2307.16425) | 不识别和弦；维护官方前处理兼容性 |
| 曲风与乐器 | [Essentia](https://github.com/MTG/essentia) | [预训练模型](https://essentia.upf.edu/models.html) | 标签有限；库和权重许可分别核对 |
| 音频情绪 | [Music2Emotion](https://github.com/AMAAI-Lab/Music2Emotion) | [Music2Emo](https://arxiv.org/abs/2502.03979) | 输入级汇总；分段调用需要自研封装与评估 |
| 音频语言描述 | [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) | [技术报告](https://arxiv.org/abs/2407.10759) | 生成描述候选；JSON合法不等于内容正确 |
| 歌词对齐 | [SOFA](https://github.com/qiuqiao/SOFA) | [中文文档](https://github.com/qiuqiao/SOFA/blob/main/README_zh.MD) | 需要歌词、语言模型/字典和切片匹配 |
| 特征辅助 | [librosa](https://github.com/librosa/librosa) | 仓库文档 | 频谱特征不直接等于情绪或质感标签 |
| 开发期诊断 | [Sonic Visualiser](https://github.com/sonic-visualiser/sonic-visualiser) | 仓库 README | 仅研发可选检查工具，不加入生产人工关卡 |

设计框架依据：[Framework PR #1](https://github.com/AngelaYellow/cross-sensory-live-music/pull/1)，提交 `a346e1113b2b485d66eac58df2e714ce93e873f1`。当前项目补充的是自动执行与失败回退，不把新的工程组合宣称为该 Framework 引用论文的原始实现。
