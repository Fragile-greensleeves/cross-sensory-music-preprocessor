# 分析方法与自动规则

当前实现是可运行的离线估计流程。方法实现、参数及限制详见 [即时路径](events-implementation.md) 和 [描述与歌词](descriptions-implementation.md)；下表对应完整配置。

| 分析目标 | 输入与工具 | 输出与处理 |
|---|---|---|
| 保留原曲时间 | FFmpeg / ffprobe | 浮点 WAV、采样率/声道/时长/哈希；不裁静音、不改变速度和响度 |
| 声源 | Demucs `htdemucs` | vocals/drums/bass/other；分轨是辅助输入，不等于信息分类 |
| 旋律音符 | 人声 → Basic Pitch ONNX | 起止、MIDI 音高；与连续基频核对覆盖和音高一致性 |
| 连续音高 | torchcrepe tiny 或 YIN | 20 ms 时间步、基频、半音、有效标记、原始周期性；无效音高为 null |
| 无唱段旋律 | other 声部周期音高 + 起音 | 超过配置时长的无可靠歌声区间才尝试；只标器乐候选，不冒充已确定主奏 |
| 和声 | 完整混音 → BTC 170 类 | 和弦起止与音类；N=no_chord，X=unknown；10 秒分块按采样点恢复时间 |
| 节奏 | 完整混音 → Beat This! final0 | 拍点、第一拍、估计小节位置与局部 BPM；没有证据不推断完整记谱拍号 |
| 段落 | 完整混音 → All-In-One | 功能标签和边界；保留官方前处理；无法覆盖的间隙为 unknown |
| 曲风/乐器 | 分段混音 → Essentia EffNet + 分类头 | top-k 标签及原始分数；候选不是确定乐器清单 |
| 情绪 | 分段混音 → Music2Emo | 效价/唤醒度原生 1–9 量表和 mood 候选；不等于观众真实感受 |
| 音频描述 | 每窗口完整混音 → Qwen2-Audio 4-bit | 曲风、乐器/人声、质感 JSON 候选；保留原始生成审计，不覆盖其他模型标签 |
| 声学质感 | RMS、频谱重心/平坦度、起音密度等 | 声学特征与规则候选；不把响度当情绪 |
| 歌词 | 无时间文本 + 人声 → SOFA 中文演唱模型 | 汉字/唱名起止、逐行时间与模型原始对齐指标；未知读音拒绝，唱名近似显式记录 |
| 语义/背景 | 用户提供文本与来源 | 提取式关键词/意象/文本摘要；不自动编造创作背景 |

Qwen 使用固定权重与独立 MLX 环境；长段落均分成不超过 30 秒的窗口，保留原段落 ID。Qwen 只生成音频描述，不输出情绪数值、歌词或歌曲背景。SOFA 不是歌词识别器，须提供文本；中文多音字及唱名近似仍可能影响对齐。参考来源的时间戳只可作事后诊断，不进入无时间歌词对齐输入。

## 时间与状态

- 局部结果转换为原曲秒数，补零不得延伸成真实事件。描述分别保留分析区间与应用区间。
- 音符是候选，连贯的主旋律归属仍需有标注数据评估；人声伴唱和复音器乐可能混淆。
- `quality` 保存原生指标及含义，不把未校准相似度命名为正确概率。
- 模型不可用的回退会显示实际后端与 partial。静音、未知和无和弦状态分开。
- 参数在配置中统一管理，不针对《晴天》手工写入主歌、副歌、和弦或情绪时间。

## 来源

[Demucs](https://github.com/facebookresearch/demucs) · [Basic Pitch](https://github.com/spotify/basic-pitch) · [torchcrepe](https://github.com/maxrmorrison/torchcrepe) · [BTC](https://github.com/jayg996/BTC-ISMIR19) · [Beat This!](https://github.com/CPJKU/beat_this) · [All-In-One](https://github.com/mir-aidj/all-in-one) · [Essentia](https://essentia.upf.edu/models.html) · [Music2Emo](https://github.com/AMAAI-Lab/Music2Emotion) · [Qwen2-Audio](https://github.com/QwenLM/Qwen2-Audio) · [SOFA](https://github.com/qiuqiao/SOFA)
