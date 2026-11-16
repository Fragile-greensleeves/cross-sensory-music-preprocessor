# 即时路径与段落分析：实现说明

所有适配器接受统一 `ctx`，输出 `status / data / warnings / provenance`。运行中不要求人工选择音符、和弦或段落；不能确定的信息保留为空或 `unknown`。`succeeded` 表示模型调用完成，不代表音乐学判断已经验证。

## 实际处理内容

| 环节 | 完整后端 | CPU 基线与自动降级 | 输出 |
|---|---|---|---|
| 声源分离 | Demucs `htdemucs`；可配置 `htdemucs_ft` | librosa HPSS，明确只分谐波与打击成分 | 分轨绝对路径、采样率、分离类型 |
| 旋律 | Basic Pitch ONNX 音符候选 + YIN 或 torchcrepe 音高核验 | YIN + 周期性/能量筛选 + 起音检测 | 音符起止/MIDI、连续音高、有效帧、来源 |
| 和声 | BTC 170 类和弦识别 | chroma 与 24 个大小三和弦模板比较 | 和弦时间段、标签、音级集合 |
| 节奏 | Beat This! 拍点与小节起点 | librosa 拍点跟踪；小节起点未知 | 拍点、小节位置、相邻拍间速度 |
| 段落 | All-In-One 功能段落模型 | MFCC 变化检测 + 最长窗口限制 | 时间段；基线不编造主歌/副歌名称 |

HPSS **不能**替代人声/鼓/贝斯/其他乐器分离。未获得人声分轨时，旋律结果标记 `partial`，保留实际输入来源。

## 全自动旋律选择

1. 优先分析分离的人声；不存在人声轨时分析 HPSS 谐波或完整混音。
2. 连续音高通过能量与周期性门限，静音帧输出 `valid=false`、音高为空。
3. Basic Pitch 音符候选需与有效音高帧一致，拒绝无声、低覆盖率和不一致的候选。
4. 人声存在至少 1 秒无可靠音高的区间时，自动尝试 `other` 分轨。只有通过筛选的音高进入输出，来源标记 `other_instrumental_candidate`。
5. 音高变化和再次起音均可切分音符；所有时间截断在原曲范围内。

`other` 中可能同时含多种乐器；上述步骤提供可追溯的器乐候选，**不能保证识别出创作者意图中的主旋律**。因此包含此类补充的结果保留 `partial`。音符、连续音高并非一一对应：模型可识别连续音高，但无法可靠确定音符起止。

## 配置

每个任务都支持 `backend` 与 `fallback`。`auto` 尝试完整后端；失败且 `fallback=true` 时自动执行真实基线，记录失败原因与实际引擎。设置 `fallback=false` 可使模型缺失或失败成为明确任务错误。

```json
{
  "tasks": {
    "separation": {
      "backend": "demucs", "model": "htdemucs", "device": "cpu",
      "threads": 4, "shifts": 0, "overlap": 0.25, "fallback": true
    },
    "melody": {
      "backend": "basic_pitch", "pitch_backend": "yin",
      "fmin": 65, "fmax": 1000, "hop_seconds": 0.02,
      "minimum_note_seconds": 0.08, "instrumental_fallback": true,
      "instrumental_gap_seconds": 1.0, "fallback": true
    },
    "harmony": {
      "backend": "btc", "btc_repo": "../models/vendor/BTC-ISMIR19",
      "fallback": true
    },
    "rhythm": {
      "backend": "beat_this", "model": "final0",
      "checkpoint": "../models/beat_this-final0.ckpt", "device": "cpu",
      "fallback": true
    },
    "sections": {
      "backend": "allin1", "model": "harmonix-all", "device": "cpu",
      "minimum_section_seconds": 12, "fallback_window_seconds": 30,
      "fallback": true
    }
  }
}
```

`melody.pitch_backend="torchcrepe"` 使用神经音高与周期性，默认 `crepe_model="tiny"`；`backend="torchcrepe"` 是 Basic Pitch + torchcrepe 组合的简写。`periodicity_threshold` 可覆盖默认 YIN 0.55 / torchcrepe 0.35，但两者分数不应直接比较。

以上路径以配置文件位于 `config/` 为例，相对于配置文件所在目录解析。

基本依赖：numpy、scipy、librosa、soundfile。完整后端另需 torch、torchaudio、demucs、beat-this、basic-pitch、onnxruntime、torchcrepe；Basic Pitch 0.4.0 的 resampy 依赖还需要 `setuptools<81`。BTC 和 All-In-One 的环境及权重由安装流程准备，不会伪造缺失模型的结果。

BTC 使用项目内 `btc_worker.py` 独立进程加载官方架构与权重，兼容旧 NumPy/YAML 与 GPU 保存权重。保留官方 10 秒 CQT 上下文，以实际采样点偏移和 `hop_length/sample_rate` 映射时间，修复旧 CLI 累计 `10/108` 帧时长的漂移；不执行旧 `test.py` 或生成无关 MIDI 文件。

All-In-One 1.1.0 使用旧 NATTEN API；本机验证组合为 `natten==0.17.3`、`torch==2.5.1`、`madmom==0.17.dev0`。NATTEN 0.17.5 移除了旧名称，0.21.1 又移除了所需 CPU/RPB 内核，不能直接升级替换。Apple 新版 Clang 编译旧 PyTorch 头文件时，本机使用：

```sh
NATTEN_N_WORKERS=2 CXXFLAGS='-Wno-invalid-specialization' \
  uv --no-cache pip install --python .venv/bin/python \
  --no-build-isolation --no-deps 'natten==0.17.3'
```

需要已安装 CMake 与 Command Line Tools。此参数兼容旧 PyTorch 头文件的编译诊断，不改变模型运算。All-In-One 保留自己的官方 `htdemucs` 分离与四轨频谱前处理，缓存于任务目录；权重固定 `taejunkim/allinone@379e5fd010b3fdd0ee8381ff8cbcfa51d70b5c19`，逐文件记录 SHA-256。`harmonix-fold0` 是单折模型，`harmonix-all` 是八折集成，报告中保留实际选型。

## 数据边界

- `quality` 中周期性、模板相似度、候选一致度是未校准指标，均标记 `calibrated=false`，不是正确概率。
- 基线和声只支持大小三和弦；扩展和弦、转位与复杂混合音不能据此可靠区分。
- 基线节拍没有小节信息，`is_downbeat / bar_index / beat_in_bar` 保留空值。
- Beat This! 输出的小节起点可用于分组；不会据此伪造记谱拍号，`meter=[]`，速度单位为 `estimated_beat`。半速/倍速歧义仍需评估。
- 无法确认功能段落时保留 `unknown`。窗口切分只是描述分析的时间容器，不是自动识别出的主歌/副歌。
- All-In-One 的时间缺口以 `unknown` 补齐，重叠裁剪并记录警告；不将相邻标签强行延伸。
- 完整混音保留为和声、节奏与段落的分析依据；分轨不会分别归一化后再用于比较原有声部响度。

## 验证

自动合成信号测试覆盖：静音无音符/拍点、同音停顿后再起、440 Hz → 523 Hz 的音高变化、C 大三和弦、0.5 秒点击间隔、弱起的小节归属、器乐候选路由、真实文件末尾、HPSS 重构与分轨名称、段落完整覆盖、模型失败回退与严格失败。

在 Apple Silicon CPU 上，用本地歌曲的 8 秒片段实际运行了 Demucs、Basic Pitch ONNX + YIN / torchcrepe、Beat This!、All-In-One `harmonix-fold0`。该检查确认依赖、模型、音频解码和适配器链路可运行，**不是准确率评测**；短片段的功能段落尤其不能据此确认。不将歌曲或分轨音频上传仓库。

Beat This! 官方下载地址本轮返回 HTTP 500，因此使用镜像的同一权重，校验 SHA-256 后写入本地缓存：`8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`。适配器优先读取本地文件/缓存，保留权重散列；未执行镜像仓库代码。

## 方法与代码来源

[Demucs](https://github.com/facebookresearch/demucs) · [Basic Pitch](https://github.com/spotify/basic-pitch) · [torchcrepe](https://github.com/maxrmorrison/torchcrepe) · [BTC](https://github.com/jayg996/BTC-ISMIR19) · [Beat This!](https://github.com/CPJKU/beat_this) · [All-In-One](https://github.com/mir-aidj/all-in-one) · [librosa](https://librosa.org/doc/latest/index.html)

[Beat This! 权重镜像与来源记录](https://huggingface.co/AEmotionStudio/muscriptor-models/blob/main/NOTICE)
