# 《晴天》当前交付版

版本：`qingtian-reviewed-2026-11-16`。视觉与触觉统一读取本目录，不使用历史自动基线重建另一份时间轴。

| 文件 | 用途 |
|---|---|
| [song-package.json](song-package.json) | 当前完整数值分析时间线，公开时移除歌词与文本语义摘录 |
| [manifest.json](manifest.json) | 数据哈希、音源哈希、时长、数量、来源与排除项 |
| [report.html](report.html) | 下载后在浏览器打开；选择同版本音频，即可同步查看 |
| [preview.png](preview.png) | 当前公开报告截图 |

916个音符、165个和弦、305个拍点（77个第一拍）、21个段落及描述。时长269.747208秒，所有秒数从原音频开头计算。

相较自动基线901个音符：加入14个谱面辅助片段及1个复核后的 D4 片段，修正1个人声音符起点与4个连续音高帧。其余数据保持原样。99个器乐候选仍保留原来源；`estimated/partial` 不能当作100%确认。

本地完整版 `qingtian-team-full.zip` 需由项目负责人另发：包含相同音乐时间线、50行/494单位歌词对齐及完整报告；不含音频或模型。拿到后读取 ZIP 内的 `song-package.json`，不要同时合并两份音符，以免重复。

详见 [Framework](../../docs/framework.md) · [接入说明](../../docs/handoff.md) · [字段格式](../../docs/package-format.md)。
