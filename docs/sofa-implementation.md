# SOFA 歌词对齐：已完成真实整曲验证

输入音频、无时间歌词与模型配置后，程序自动生成拼音、处理已配置的唱名发音、执行 SOFA 强制对齐，输出逐句和逐字/唱名时间。该模块不自动检索歌词，也不把提供方歌词视作录音真值。

## 安装与配置

```bash
python scripts/setup_sofa.py
# 已安装后，仅验证源码和权重：
python scripts/setup_sofa.py --verify-only
```

安装使用独立 `.venv-sofa`，不修改主项目环境。需要已安装 `uv`；源码来自 [qiuqiao/SOFA](https://github.com/qiuqiao/SOFA)，模型来自作者发布的[普通话歌声模型](https://github.com/qiuqiao/SOFA/discussions/16)。模型压缩包约 377 MB，解压检查点约 406 MB。

在流水线 `tasks.lyrics_context` 中配置：

```json
{
  "backend": "sofa",
  "fallback": true,
  "source_metadata": "data/raw/qingtian-lyrics.source.json",
  "sofa": {
    "python": ".venv-sofa/bin/python",
    "repo": "models/vendor/SOFA",
    "checkpoint": "models/sofa/v1.0.0_mandarin_singing.ckpt",
    "dictionary": "models/vendor/SOFA/dictionary/opencpop-extension.txt",
    "text_type": "mandarin",
    "solfege": true,
    "threads": 2,
    "timeout_seconds": 1800
  }
}
```

`source_metadata` 为可选文件，仅用于该曲的来源归属。其 JSON 必须包含与歌词文件**原始字节**匹配的 `text_sha256` 和 HTTP(S) `source_url`；可附 `source_type`。哈希不匹配时拒绝归属，不标成已验证文本。`solfege` 只在需要识别唱名时启用；普通歌曲可省略。

运行示例（以实际配置文件名为准）：

```bash
musicprep analyze data/raw/qingtian.flac --output work/qingtian-sofa \
  --config config/full.json --lyrics data/raw/qingtian-lyrics.txt
```

歌词来源示例：[酷狗《晴天》页面](https://www.kugou.com/song-36/c7i2yb5.html)。正式验证仅使用其无时间歌词；提供方时间戳单独用于事后比较。

## 实际处理流程

1. 优先使用分离得到的人声，保留原曲时间零点，转成 44.1 kHz 单声道供模型读取。
2. 中文短语由 `pypinyin` 自动转拼音；保留每个原字与拼音的对应关系。纯拼音输入可用 `text_type: pinyin`。
3. 逐词核对官方词典；未知词不能被 SOFA 静默跳过。模型加载后再检查所有发音音素是否在检查点词表中。
4. 使用官方 `DictionaryG2P → LitForcedAlignmentTask → AP detector → post_processing → Exporter`。CPU、2 线程，整曲全局强制对齐；没有用参考 LRC 预给句子位置。
5. 读取 HTK `words`，明确识别 `SP`、`AP` 和 `BND` 等非歌词标记；每个真实词必须与输入顺序逐一对应，再合并为原歌词行。
6. 校验边界有限、非负、不越过曲长、词序单调；输出模型估计时间、检查点/字典哈希、源码版本、置信分数与日志位置。

模型官方 CLI 默认自动选择设备；本项目使用同一套官方推理模块的薄封装，明确固定 CPU，防止与其他模型争用 MPS。官方当前版本的 `BND` 表示乐句边界，可以为零时长；它不是歌词内容。

## 唱名与语言限制

《晴天》的原始 `Re / So / Si / Do / La` 保留为原文，不替换成汉字。启用 `solfege` 后，独立字典 token 使用近似发音：

| 原文 | 模型音素近似 |
|---|---|
| Do | d o |
| Re | r ei |
| Mi | m i |
| Fa | f a |
| So / Sol | s o |
| La | l a |
| Si | s i |
| Ti | t i |

唱名使用 `latin_*` token，避免与汉字拼音的 `re`、`si` 等混淆。`pronunciation_approximations` 记录这些近似；它们没有经过通用多语种发音验证。其他英文词需显式提供 `token_pronunciations`（原词 → 模型支持的音素数组）；数字、未知语言或不支持的字符不会被随意删掉或转写。

自动 G2P 可能读错多音字；歌词版本不同、漏字、重复句不符和和声干扰也可能影响对齐。运行成功只代表完成模型处理与结构校验，不代表歌词内容和每个边界都人工验证正确。

## 《晴天》真实验证

2026-11-16，macOS ARM64 / M5 / 16 GB，使用 269.747 秒的整曲人声和无时间参考歌词：

| 项目 | 结果 |
|---|---|
| 原文行数 | 50，完整保留 |
| 对齐单位 | 494 个汉字/唱名，全部与输入匹配 |
| 适配器状态 | `succeeded` |
| 端到端耗时 | 7.10 秒（模型环境已热启动） |
| 最大常驻内存 | 约 2.94 GB（系统 time 报告） |
| 运行方式 | 整曲 CPU 强制对齐，无参考时间输入 |
| 与提供方 LRC 的起点差 | 中位绝对差 0.153 秒；最大 0.467 秒；50/50 行小于 0.5 秒 |

提供方 LRC 不是人工标注真值；上述差异是独立一致性诊断，不是模型准确率。当前整曲无需切片；更长音频的全局动态规划内存会增加，现阶段不按词数均分时间来假造分段。

本地验证产物（不提交音频或完整歌词）：`work/qingtian-sofa-full/result.json`、`provider-comparison.json` 和对应 `sofa-*/worker.log`、HTK、TextGrid。

## 复现版本

- 源码：[9549c6a86d16019c817eefe4bb8183405da524cc](https://github.com/qiuqiao/SOFA/tree/9549c6a86d16019c817eefe4bb8183405da524cc)。
- 正式权重：`v1.0.0_mandarin_singing.ckpt`，来自官方 release v1.0.1 压缩包。
- 核心环境：Python 3.11、PyTorch/torchaudio 2.5.1、Lightning 2.4.0、NumPy 1.24.4、librosa 0.9.2、pypinyin 0.55.0；`setuptools<81` 保留 librosa 所需 `pkg_resources`。

```text
archive SHA256     63a8c6191fe6bdf29ac40af2955ef96816b8391596d94a4340e0f58dcf47f26e
checkpoint SHA256  612cb69ef3503befc43286f0d90893b4b3ad959df07ed08799dba2bea69d2b51
dictionary SHA256  e912ee4ec1d87fcdfd8b82c0a8f3999f0db7fd731bec08435256b621ebebb5a1
```

可重复验证：`python -m unittest tests.test_lyrics -v`。真实模型已实际运行；单元测试另覆盖缺文本、来源哈希错误、重复歌词、offset、清屏边界、BND、词序不符、未知英文和唱名近似。
