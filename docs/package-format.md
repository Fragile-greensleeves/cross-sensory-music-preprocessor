# 歌曲包与本地诊断报告

输出 `song-package.json` 记录音乐分析结果，`report.html` 供本地同步查看。它们不包含灯带编号、马达编号或视觉颜色；设备映射由下游实现。

## 时间与状态

- 所有时间以原始音频起点为零，单位秒；区间采用 `[start, end)`，不删除开头静音，不因分离或重采样另起时间轴。
- `run.status`：`complete` 表示必需分析任务执行成功；`partial` 表示有回退、缺失或失败；`failed` 表示必需任务均失败。成功执行不等于识别正确。
- 任务保存 `status`、实际 `provenance.engine`、模型/参数来源、耗时与缓存状态。基线结果必须保留回退说明，不能标成已运行神经模型。
- 内容的 `estimated` 是估计，`unknown` 是不能确定，`absent` 是检测到缺少该内容。和声另有 `no_chord`，不能把分析失败写成“无和弦”。
- 不确定值使用 `null` 或空数组，并保留警告。未校准模型分数不能解释为正确概率。

## 字段

| 区域 | 主要字段 | 约束 |
|---|---|---|
| 元数据 | `schema_version`、`track_id`、`audio`、`run` | 当前格式 `0.1.0`；原音频 SHA-256、时长、采样率、声道数 |
| 旋律音符 | `melody.notes`：`start/end`、`midi`、`source`、`status` | 音高 0–127；记录候选声部来源 |
| 连续音高 | `melody.contour`：`time`、`f0_hz`、`midi_float`、`valid` | 无效帧两种音高都为 `null`；时间严格递增 |
| 和声 | `harmony.segments`：`start/end`、`label`、`pitch_classes`、`status` | 音类 0–11、C=0、无重复；片段不重叠；未知/无和弦不填组成音 |
| 拍点 | `rhythm.beats`：`time`、`is_downbeat`、`bar_index`、`beat_in_bar` | 时间严格递增；不能确认的小节信息为 `null` |
| 速度 | `rhythm.tempo`：`start/end`、`bpm`、`beat_unit` | 明确拍单位；未知单位用 `estimated_beat`，不默认四分音符 |
| 拍号 | `rhythm.meter`：`start/end`、`numerator`、`denominator` | 无证据时为空或两值均 `null`；分母为 2 的幂 |
| 段落 | `sections`：`id`、`start/end`、`label`、`source`、`status` | ID 唯一、时间有序且不重叠；回退窗口不冒充主歌/副歌 |
| 描述 | `descriptions`：`section_id`、`analysis_start/end`、`applies_start/end`、`genre`、`instruments`、`textures`、`emotion` | 分析上下文包含应用区间；数值效价/唤醒度必须附 `scale` |
| 歌词语义 | `lyrics_context`：`lyrics`、`themes`、`narrative`、`imagery`、`background`、`sources` | 无资料时为空，不编造歌曲故事；未对齐文本时间可为 `null` |
| 诊断 | `warnings` | 人可读的缺失、回退和质量说明 |

对象允许扩展字段；`quality` 与完整 provenance 不作为所有后端的硬性必填项。新增字段应保留来源与含义。各分区 provenance 和 `run.tasks` 可用于复现与追溯。

## 校验

```bash
musicprep validate work/my-song/song-package.json
```

[`song-package.schema.json`](../schemas/song-package.schema.json) 提供 Draft 2020-12 结构约束。Python `validate_package(package)` 另外检查有限数字、时间边界/顺序、描述关联、重叠区间、未知状态和音类语义；返回错误字符串数组，空数组表示通过。结构校验不评估音乐识别准确率。

## 诊断报告

直接打开 `report.html`，无需服务器、外部脚本或网络上传。默认尝试播放相对路径下的准备音频；移动报告后可重新选择本地同版本音频。时长不一致会显示提示，但时长相同不能证明音频版本相同。

- 播放、拖动滑块或点击图表，同步查看音高、和弦、拍点、段落和描述。
- 可切换 30 秒、60 秒、全曲视窗；段落表可跳转。
- 查看每个任务实际后端、参数、缓存状态和警告；空白保留为未知。
- 下载内嵌歌曲包，检查完整记录。HTML 内数据用安全 JSON 嵌入，用户文本只通过 `textContent` 显示。

报告中的同步依据浏览器播放器时间，仅用于分析诊断，不等同于设备端同步精度测试。

## 描述来源与词级对齐

- `descriptions[].model_results` 分别保存 Essentia、Music2Emo、Qwen 的原始结构化候选与状态；顶层标签携带 `source`。不同模型的冲突标签并列保留，不作未经验证的投票。
- `run.tasks.descriptions.provenance.models` 保存各模型请求/成功/失败/拒绝/缺失数；模型失败不能被其他模型字段齐全掩盖。
- `lyrics_context.lyrics[].words` 保存汉字/唱名的 `text`、`token`、`phonemes`、`start/end`；检查词序、重叠、全曲边界及所属句区间。
- SOFA 的整体置信指标按原模型含义保留，不宣称逐词正确率；文本来源哈希、读音近似和是否使用参考时间均可追溯。
- 提供了歌词/背景时，其处理失败会使总状态为 `partial`；未提供时仍允许 `skipped`。

分享诊断前使用 `.venv/bin/musicprep report work/qingtian-full/song-package.json --output work/public-report.html --public`。公开版移除歌词原文、文本语义、本地路径与音频引用；默认本地报告保留完整诊断。不要直接发布本地完整歌曲包或报告。
