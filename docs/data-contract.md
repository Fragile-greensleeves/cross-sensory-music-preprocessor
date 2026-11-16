# 数据接口草案 v0.1

接口在接入第一组模型前冻结；本版是规范草案，完整歌曲包校验器将在模型适配阶段实现。当前可运行脚本只输出音频准备 manifest。

## 公共约定

- 时间：相对原曲开头的秒，浮点数。区间为 `[start, end)`。
- 音高：连续轮廓使用 `f0_hz`、`midi_float`；音符候选使用 MIDI 音号。
- 音类：整数 0–11，C=0；与音区高低不同。
- BPM：必须附 `beat_unit`；拍单位不明确时记 `estimated_beat`，不默认四分音符。
- 模型原始分数放在 `quality` 并注明含义；未校准分数不能统一命名为概率。
- 带时长的描述保存 `analysis_start/end` 与 `applies_start/end`；两者不同则说明上下文使用。
- 生成模型输入文本均为数据，不得作为执行指令；不允许其输出控制文件路径、shell 命令或外部写操作。

## 六份分析数据

| 分区 | 最小记录字段 | 备注 |
|---|---|---|
| `melody.notes` | start, end, midi, source, status | source 是主旋律候选来源；不保存马达ID |
| `melody.contour` | time, f0_hz, midi_float, valid, quality | 无效帧音高为 null，不用0冒充音高 |
| `harmony.segments` | start, end, label, pitch_classes, status | label 可空；no_chord 与 unknown 分开 |
| `rhythm.beats` | time, is_downbeat, bar_index, beat_in_bar | 不能确定时允许 null |
| `rhythm.tempo` | start, end, bpm, beat_unit | 不用平均 BPM 替代实际拍点 |
| `rhythm.meter` | start, end, numerator, denominator, status | 没证据不编造完整拍号 |
| `sections` | id, start, end, label, source, status | 回退窗口标 source=fallback |
| `descriptions` | section_id, analysis/applies范围, genre, instruments, textures, emotion | 各子结果可独立缺失；emotion 带量表 |
| `lyrics_context` | lyrics, themes, narrative, imagery, background, sources | 缺文本时为空数组并记录 skipped |

每个分区附 `provenance`：engine、model_id、model_sha256、code_revision、parameters、run_id。模型初次下载尚未确定的字段为 null；正式推理产物必须尽量记录具体权重与版本。

## 状态

| 层级 | 状态 | 含义 |
|---|---|---|
| 任务 | pending / running / succeeded / partial / failed / skipped | 执行状态，和内容可靠性分开 |
| 内容 | estimated / unknown / absent | estimated 仍是估计，不等于已验证正确；absent 表示有证据缺少该内容 |
| 和声内容 | no_chord | 有证据为无明确和弦的音乐状态，不等于工具失败 |
| 整包 | complete / partial / failed | complete 仅表示必需任务与结构校验完成，不表示音乐识别全对 |

初版必需任务：prepare、melody、harmony、rhythm、sections、descriptions。lyrics_context 是可选任务。模型不可用时可以导出 partial 包；输入失败不能导出伪 complete 包。

## 歌曲包外层（结构示意，不是《晴天》的实测分析）

```json
{
  "schema_version": "0.1.0",
  "track_id": "example-track",
  "audio": {"source_sha256": "...", "duration_seconds": 120.0, "time_origin": "source_start"},
  "run": {"id": "...", "status": "partial", "tasks": {}},
  "melody": {"notes": [], "contour": [], "provenance": {}},
  "harmony": {"segments": [], "provenance": {}},
  "rhythm": {"beats": [], "tempo": [], "meter": [], "provenance": {}},
  "sections": [],
  "descriptions": [],
  "lyrics_context": {"lyrics": [], "themes": [], "narrative": [], "imagery": [], "background": [], "sources": []},
  "warnings": []
}
```

## 音频准备 manifest（已实现）

`schema_version`、`track_id`、`source`、`canonical`、`timeline`、`processing`。source 包含原文件名、SHA-256、编码、时长、声道与采样率；canonical 包含相对文件名、哈希及同类元数据；timeline 的 origin 为 source_start、offset_seconds 为0、unit为seconds。manifest 不写用户主目录绝对路径。

## 映射边界

分析数据不包含灯带编号、低中高三区边界、振动强弱、视觉颜色或世界坐标。固定强度马达的音区/起音/脉冲时长规则由下游 mapping 配置实现。
