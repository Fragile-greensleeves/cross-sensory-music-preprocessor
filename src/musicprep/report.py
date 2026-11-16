"""Write a self-contained, offline diagnostic viewer for an estimated song package."""
import html
import json
import os
import re
from pathlib import Path
from urllib.parse import quote


def _safe_json(value):
    # JSON is read as textContent, never executed. Escape HTML parser delimiters.
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))



def _public_package(package):
    """Allowlisted publication view: no lyrics, raw model text, paths or audio URLs.

    It is intentionally a review snapshot, not a replacement for the private
    reproducibility package. Unknown extension fields are never copied.
    """
    def pick(value, keys):
        return {key: value[key] for key in keys if key in value}

    def safe_id(value):
        if not isinstance(value, str):
            return value
        if (value.startswith(("/", "~", "./", "../", "file:"))
                or any(part in value for part in ("/Users/", "/home/", "/tmp/", "/private/", "/var/"))
                or "\\" in value or re.search(r"[A-Za-z]:[/\\]", value)):
            return "local-model"
        return value

    def provenance(value):
        result = {key: safe_id(value[key]) for key in
                  ("engine", "model_id", "backend", "method_version", "source_revision") if key in value}
        if isinstance(value.get("models"), dict):
            result["models"] = {}
            for name, model in value["models"].items():
                item = provenance(model)
                item.update(pick(model, ("status",)))
                if isinstance(model.get("counts"), dict):
                    item["counts"] = pick(model["counts"], ("requested", "succeeded", "failed", "rejected", "missing", "generated", "silent_audio"))
                # Runtime can contain prompts and absolute paths; only publish identity/hash.
                runtime = model.get("runtime", {})
                item["runtime"] = {key: safe_id(runtime[key]) for key in
                                   ("engine", "model_id", "model_revision", "model_sha256") if key in runtime}
                result["models"][safe_id(name)] = item
        return result

    def rows(values, keys):
        return [pick(value, keys) for value in values]

    result = pick(package, ("schema_version",))
    result["track_id"] = safe_id(package.get("track_id", "music-analysis"))
    result["audio"] = pick(package["audio"], ("source_sha256", "duration_seconds", "sample_rate", "channels", "time_origin"))
    result["run"] = {"id": "public-" + package["audio"]["source_sha256"][:16],
                     "status": package["run"]["status"], "tasks": {}}
    for name, task in package["run"].get("tasks", {}).items():
        item = pick(task, ("status", "cached", "duration_seconds"))
        item["provenance"] = provenance(task.get("provenance", {}))
        result["run"]["tasks"][name] = item
    definitions = {
        "melody": {"notes": ("start", "end", "midi", "source", "status"),
                   "contour": ("time", "f0_hz", "midi_float", "valid")},
        "harmony": {"segments": ("start", "end", "label", "pitch_classes", "status")},
        "rhythm": {"beats": ("time", "is_downbeat", "bar_index", "beat_in_bar"),
                   "tempo": ("start", "end", "bpm", "beat_unit"),
                   "meter": ("start", "end", "numerator", "denominator", "status")},
    }
    for field, arrays in definitions.items():
        result[field] = {key: rows(package.get(field, {}).get(key, []), keys) for key, keys in arrays.items()}
        result[field]["provenance"] = provenance(package.get(field, {}).get("provenance", {}))
    result["sections"] = rows(package.get("sections", []), ("id", "start", "end", "label", "source", "status"))
    result["descriptions"] = []
    for description in package.get("descriptions", []):
        item = pick(description, ("section_id", "analysis_start", "analysis_end", "applies_start", "applies_end", "status"))
        for key in ("genre", "instruments", "textures"):
            item[key] = [pick(tag, ("label", "score", "source", "status")) if isinstance(tag, dict) else tag
                         for tag in description.get(key, [])]
        item["emotion"] = pick(description.get("emotion", {}), ("valence", "arousal", "scale", "status", "source"))
        item["model_results"] = {name: pick(value, ("status", "inference_status")) for name, value in description.get("model_results", {}).items()}
        result["descriptions"].append(item)
    lyrics = package.get("lyrics_context", {})
    result["lyrics_context"] = {key: [] for key in ("lyrics", "themes", "narrative", "imagery", "background", "sources")}
    result["publication"] = {
        "redacted": True, "lyrics_line_count": len(lyrics.get("lyrics", [])),
        "lyrics_word_count": sum(len(line.get("words", [])) for line in lyrics.get("lyrics", []) if isinstance(line, dict)),
        "warning_count": len(package.get("warnings", [])),
    }
    review = package.get("melody", {}).get("provenance", {}).get("melody_review")
    if review:
        # Public readers need transition semantics, not private before/after
        # records, raw evidence text, or local artifact paths.
        decisions = [pick(d, ("start", "end", "resolution")) for d in review.get("decisions", [])
                     if d.get("resolution") in ("upper_voice_selected", "no_stable_instrumental_lead", "vocal_onset_corrected")]
        result["melody"]["provenance"]["melody_review"] = {
            "summary": "本版本包含声部与起音复核；完整证据保存在本地交接包。",
            "decisions": decisions,
        }
        result["publication"]["review_applied"] = True
    result["warnings"] = ["公开预览已移除歌词、叙事与背景原文、模型原始输出、本地路径及音频链接；请在本地完整报告查看运行详情。"]
    # Even a source or label extension carrying a local path must not leak it.
    def scrub(value):
        if isinstance(value, dict):
            return {safe_id(key): scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return safe_id(value)
    return scrub(result)


def write_report(package, output_path, audio_path=None, *, public=False):
    """Write an offline report; public=True exports a redacted, audio-free snapshot."""
    if public:
        package = _public_package(package)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_url = ""
    if audio_path is not None and not public:
        audio_file = Path(audio_path).resolve()
        if audio_file.is_file():
            relative = os.path.relpath(audio_file, output_path.parent.resolve())
            audio_url = quote(Path(relative).as_posix(), safe="/")
    replacements = {
        "__TRACK_TITLE__": html.escape(str(package.get("track_id", "音乐分析"))),
        "__AUDIO_URL__": html.escape(audio_url, quote=True),
        "__PACKAGE_JSON__": _safe_json(package),
    }
    document = re.sub(r"__TRACK_TITLE__|__AUDIO_URL__|__PACKAGE_JSON__",
                      lambda match: replacements[match.group()], _HTML)
    output_path.write_text(document, encoding="utf-8")
    return output_path


_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light"><title>__TRACK_TITLE__ · 音乐分析诊断</title>
<style>
:root{--navy:#173862;--blue:#497eb7;--purple:#8970ad;--muted:#66748a;--line:#dce4ef;--bg:#f4f7fc;--pale:#edf3fb;--violet:#f0ebf7;--green:#197459;--amber:#926313;--red:#a53846}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#25364b;font:15px/1.6 system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif}main{max-width:1240px;margin:auto;padding:30px 26px 50px}h1{font-size:28px;line-height:1.3;color:var(--navy);margin:8px 0}h2{font-size:18px;margin:0 0 14px;color:var(--navy)}h3{font-size:15px;margin:0 0 9px}p{margin:6px 0}.eyebrow{font-size:12px;letter-spacing:2px;color:var(--blue);font-weight:700}.muted{color:var(--muted)}.header{display:flex;justify-content:space-between;gap:18px;align-items:center;margin-bottom:22px}.badge{display:inline-block;border:1px solid var(--line);border-radius:20px;padding:5px 12px;background:white;font-size:13px;white-space:nowrap}.badge.partial{background:#fff6e6;color:var(--amber);border-color:#efd7aa}.badge.complete,.badge.succeeded{background:#eaf6ef;color:var(--green);border-color:#bde0cd}.badge.failed{background:#fbecef;color:var(--red);border-color:#edc5ce}.badge.skipped,.badge.unknown{color:var(--muted);background:#f0f3f7}.card{background:white;border:1px solid var(--line);border-radius:16px;padding:20px;margin-bottom:18px;box-shadow:0 4px 18px #17386205}.summary{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:18px}.metric{border:1px solid var(--line);background:white;border-radius:13px;padding:15px 18px}.metric strong{font-size:24px;display:block;color:var(--navy)}.metric span{font-size:12px;color:var(--muted)}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:12px}audio{width:min(540px,100%);height:40px}button,.filebutton,select{font:inherit;font-size:13px;border:1px solid #bccfe3;background:white;border-radius:8px;padding:8px 12px;color:var(--navy);cursor:pointer}button:hover,.filebutton:hover{background:var(--pale)}button:focus-visible,.filebutton:focus-within,select:focus-visible{outline:3px solid #9bb6dc;outline-offset:2px}.filebutton input{position:absolute;width:1px;height:1px;opacity:0}.scrub{display:flex;align-items:center;gap:12px;margin-top:16px}input[type=range]{width:100%;accent-color:var(--blue)}#time{min-width:112px;font-variant-numeric:tabular-nums}.charttop{display:flex;justify-content:space-between;align-items:center;gap:10px;margin:12px 0}.legend{display:flex;gap:15px;flex-wrap:wrap;color:var(--muted);font-size:12px}.legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.chart-wrap{border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fcfdff}canvas{display:block;width:100%;height:360px;touch-action:none}.tip{font-size:12px;color:var(--muted);margin-top:9px}.current{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.current>div{padding:15px;background:var(--pale);border-radius:10px}.current>div:last-child{background:var(--violet)}.lyric-current{border-top:1px solid var(--line);margin-top:16px;padding-top:13px}.lyric-current .line{font-size:17px;color:var(--navy);overflow-wrap:anywhere;margin:10px 0 5px}.lyric-focus{margin-top:10px}.lyric-word[hidden]{display:none}.lyric-word{display:inline-block;min-width:80px;max-width:100%;padding:8px 18px;background:var(--violet);color:#624d80;border-radius:12px;font-size:clamp(40px,5vw,52px);font-weight:600;line-height:1.3;text-align:center;overflow-wrap:anywhere}.label{font-size:12px;color:var(--muted);display:block}.value{font-size:20px;font-weight:600;color:var(--navy)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}.grid .card{min-width:0}.chips{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}.chip{padding:4px 9px;border-radius:6px;background:var(--pale);font-size:12px;max-width:100%;overflow-wrap:anywhere}.chip small{display:block;color:var(--muted);font-size:11px}.model-summary{padding:9px 0;border-top:1px dashed var(--line);margin-top:7px}.model-heading{display:flex;justify-content:space-between;align-items:center;gap:8px}.model-summary .badge{font-size:11px;padding:2px 8px}.chip.purple{background:var(--violet);color:#624d80}.description{white-space:pre-wrap;overflow-wrap:anywhere}.task{border-bottom:1px solid var(--line);padding:10px 0;display:flex;justify-content:space-between;gap:12px}.task>.badge{align-self:flex-start}.task:last-child{border:0}.taskname{font-weight:600}.engine{font-size:12px;overflow-wrap:anywhere;color:var(--muted)}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{background:var(--pale);font-weight:600}td button{padding:3px 7px;font-variant-numeric:tabular-nums}details{margin-top:8px}summary{cursor:pointer;font-weight:600;color:var(--navy)}pre{white-space:pre-wrap;word-break:break-word;font-size:12px;background:var(--bg);padding:12px;border-radius:8px;max-height:300px;overflow:auto}.warning{padding:8px 10px;border-left:3px solid #cda159;background:#fffbf4;margin:8px 0;overflow-wrap:anywhere;font-size:13px}.scroll{max-height:300px;overflow:auto}.footer{font-size:12px;color:var(--muted);padding:2px 4px}.empty{color:var(--muted);font-size:13px}#audioNotice{font-size:12px;color:var(--muted)}@media(max-width:760px){main{padding:20px 14px}.header{align-items:flex-start}.summary{grid-template-columns:repeat(2,1fr)}.grid{grid-template-columns:1fr}.current{grid-template-columns:1fr}.metric:last-child{grid-column:1/-1}.card{padding:15px}h1{font-size:23px}.charttop{align-items:flex-start;flex-direction:column}}
</style></head>
<body><main>
<header class="header"><div><div class="eyebrow">CROSS-SENSORY · ANALYSIS REVIEW</div><h1 id="track"></h1><p class="muted">原曲时间轴上的自动估计 · 即时事件与音乐描述</p></div><span id="status" class="badge"></span></header>
<div id="metrics" class="summary"></div>

<section class="card"><h2>同步查看分析结果</h2><div class="controls"><audio id="audio" controls preload="metadata" src="__AUDIO_URL__"></audio><label class="filebutton">选择本地音频<input id="audioFile" type="file" accept="audio/*"></label><button id="download">下载歌曲数据包</button></div><p id="audioNotice">音频只在本机播放，不会上传。若浏览器未载入相对路径，请选择对应原曲文件。</p><div class="scrub"><span id="time">0:00 / 0:00</span><input id="seek" type="range" min="0" max="1" value="0" step="0.01" aria-label="原曲时间"></div>
<div class="charttop"><div class="legend"><span><i style="background:#497eb7"></i>音高轮廓 / 音符</span><span><i style="background:#8970ad"></i>和弦片段</span><span><i style="background:#1b7964"></i>拍点 / 第一拍</span></div><label>显示范围 <select id="window"><option value="30">30 秒</option><option value="60">60 秒</option><option value="all">全曲</option></select></label></div><div class="chart-wrap"><canvas id="chart" aria-label="旋律、和弦和拍点同步图"></canvas></div><p class="tip">点击图表可跳转。空白或“未知”表示没有可靠输出；未把模型分数视为人工确认。</p></section>
<section class="card"><div class="current"><div><span class="label">当前旋律</span><span class="value" id="currentPitch">—</span><p class="muted" id="pitchSource"></p></div><div><span class="label">当前和弦</span><span class="value" id="currentChord">—</span><p class="muted" id="chordPcs"></p></div><div><span class="label">当前段落</span><span class="value" id="currentSection">—</span><p class="muted" id="currentTempo"></p></div></div><div class="lyric-current"><span class="label">当前歌词与字词对齐</span><div class="lyric-focus"><span id="currentWord" class="lyric-word">—</span></div><p id="currentLyric" class="line">—</p><span id="lyricTiming" class="engine"></span></div></section>
<div class="grid"><section class="card"><h2>当前片段描述</h2><div id="description"></div><p class="tip">质感与情绪是音频线索的模型/规则估计，不表示观众的实际感受。</p></section><section class="card"><h2>任务与实际后端</h2><div id="tasks"></div></section></div>
<div class="grid"><section class="card"><h2>段落与边界</h2><div class="scroll"><table><thead><tr><th>时间</th><th>段落</th><th>来源 / 状态</th></tr></thead><tbody id="sections"></tbody></table></div></section><section class="card"><h2>歌词与背景</h2><div id="lyrics"></div><p class="tip">无歌词或资料时显示缺失，不从音频编造创作故事。</p></section></div>
<section class="card"><h2>运行说明与不确定项</h2><div id="warnings" class="scroll"></div><details id="scoreSupplement" hidden></details><details id="melodyReview" hidden></details><details><summary>运行元数据</summary><pre id="metadata"></pre></details></section>
<p class="footer">complete 仅表示必需任务执行完成，不代表全部音乐信息识别正确。图表是诊断界面，不是最终视觉或触觉映射。</p>
</main><script id="package-data" type="application/json">__PACKAGE_JSON__</script>
<script>
'use strict';
const P=JSON.parse(document.getElementById('package-data').textContent), $=id=>document.getElementById(id);
const duration=P.audio.duration_seconds, notes=P.melody.notes||[], contour=P.melody.contour||[], chords=P.harmony.segments||[], beats=P.rhythm.beats||[], sections=P.sections||[], descriptions=P.descriptions||[];
const labels={complete:'处理完成',partial:'部分结果 / 有回退',failed:'处理失败',succeeded:'已执行',skipped:'已跳过',unknown:'未知',estimated:'估计',absent:'未检测到',no_chord:'无和弦',pending:'待处理',running:'处理中',rejected:'输出不合规',missing:'缺少输出',candidate:'候选',generated:'模型已生成',silent_audio:'静音跳过生成'};
const taskNames={separation:'声部分离',melody:'旋律',harmony:'和声',rhythm:'节奏',sections:'段落',descriptions:'音乐描述',lyrics_context:'歌词与背景',prepare:'音频准备'};
const names=['C','C♯','D','D♯','E','F','F♯','G','G♯','A','A♯','B'];
const text=(node,value)=>{node.textContent=value==null?'—':String(value);};
const el=(tag,value,cls)=>{const n=document.createElement(tag);if(value!==undefined)text(n,value);if(cls)n.className=cls;return n;};
const fmt=t=>`${Math.floor(Math.max(0,t)/60)}:${(Math.max(0,t)%60).toFixed(1).padStart(4,'0')}`;
const itemText=value=>typeof value==='string'?value:(value&&typeof value==='object'?(value.label||value.tag||value.name||value.text||JSON.stringify(value)):String(value));
function badge(status){return el('span',labels[status]||status,'badge '+(['complete','partial','failed','succeeded','skipped','unknown'].includes(status)?status:'unknown'));}
const modelNames={essentia:'Essentia',music2emo:'Music2Emo',qwen2_audio:'Qwen2-Audio',acoustic_rules:'声学规则'};
function sourceName(source){return modelNames[source]||source||'来源未记录';}
function tagChip(value,purple=false){const tag=el('span',itemText(value),'chip'+(purple?' purple':''));const source=value&&typeof value==='object'?value.source:null;const score=value&&typeof value==='object'&&Number.isFinite(value.score)?` · 模型分数 ${value.score.toFixed(3)}`:'';tag.append(el('small',sourceName(source)+score));return tag;}
function modelSummary(models,parent,coverage){for(const [name,model] of Object.entries(models||{})){const row=el('div',undefined,'model-summary'),heading=el('div',undefined,'model-heading');heading.append(el('strong',sourceName(name)),badge(model.inference_status==='silent_audio'?'silent_audio':model.status||'unknown'));row.append(heading);if(model.inference_status==='generated')row.append(el('div','模型已实际生成候选','engine'));if(model.inference_status==='silent_audio')row.append(el('div','静音规则返回空标签，未执行生成','engine'));if(coverage){const c=model.counts;if(c&&Number.isFinite(c.requested)&&Number.isFinite(c.succeeded)){row.append(el('div',`已接受 ${c.succeeded} / 请求 ${c.requested} 片段`,'engine'));row.append(el('div',`失败 ${c.failed??'—'} · 缺失 ${c.missing??'—'} · 拒绝结果 ${c.rejected??'—'} 条`,'engine'));}else row.append(el('div','覆盖数量未记录','engine'));if(c&&('generated' in c||'silent_audio' in c))row.append(el('div',`生成窗口 ${c.generated??0} · 静音跳过 ${c.silent_audio??0}`,'engine'));}parent.append(row);}}
function at(rows,time,start='start',end='end'){return rows.find(r=>time>=r[start]&&time<r[end]);}
function nearestFrame(time){let lo=0,hi=contour.length;while(lo<hi){const mid=(lo+hi)>>1;if(contour[mid].time<time)lo=mid+1;else hi=mid;}const prev=contour[Math.max(0,lo-1)],next=contour[lo];const point=(!prev||(next&&Math.abs(next.time-time)<Math.abs(prev.time-time)))?next:prev;return point&&Math.abs(point.time-time)<0.15?point:null;}
text($('track'),P.track_id);$('status').replaceWith(Object.assign(badge(P.run.status),{id:'status'}));
for(const [value,label] of [[fmt(duration),'原曲时长'],[notes.length,'音符候选'],[chords.length,'和弦片段'],[beats.length,'拍点候选'],[sections.length,'段落 / 窗口']]){const box=el('div',undefined,'metric');box.append(el('strong',value),el('span',label));$('metrics').append(box);}
$('seek').max=duration;text($('metadata'),JSON.stringify({audio:P.audio,run:P.run},null,2));
const melodyReview=P.melody.provenance?.melody_review;
if(melodyReview){const box=$('melodyReview');box.hidden=false;box.append(el('summary','声部与起音复核记录'),el('p',melodyReview.summary,'muted'),el('pre',JSON.stringify(melodyReview,null,2)));}
const scoreNotes=notes.filter(n=>n.source==='score_assisted_gap');
if(scoreNotes.length){const box=$('scoreSupplement');box.hidden=false;box.append(el('summary',`参考资料与补充记录 · ${scoreNotes.length} 个片段`),el('p','所有音符使用统一样式。补充来源在此保留；参考音符不代表已测得连续音高。','muted'));const list=el('div',undefined,'chips');for(const n of scoreNotes){const button=el('button',`${n.start.toFixed(2)}–${n.end.toFixed(2)} 秒 · ${names[n.midi%12]}${Math.floor(n.midi/12)-1}`);button.onclick=()=>seek(n.start+(n.end-n.start)/2);list.append(button);}box.append(list);const ref=P.melody.provenance?.score_supplement;if(ref){box.append(el('p',melodyReview?.summary||ref.limitations||'仅补参考覆盖的空缺，其他不确定项保留。','tip'));const detail=el('details');detail.append(el('summary',melodyReview?'此前补充记录（后续处理见复核记录）':'补充方法与逐项证据'),el('pre',JSON.stringify(ref,null,2)));box.append(detail);}}

for(const [name,t] of Object.entries(P.run.tasks||{})){const row=el('div',undefined,'task'),body=el('div');body.append(el('div',taskNames[name]||name,'taskname'));const prov=t.provenance||{};const engine=prov.engine||t.engine||'未记录';body.append(el('div',[engine,prov.model_id||prov.backend||'',t.cached?'缓存结果':'',Number.isFinite(t.duration_seconds)?`${t.duration_seconds.toFixed(2)}s`:''].filter(Boolean).join(' · '),'engine'));row.append(body,badge(t.status));if(name==='descriptions')modelSummary(prov.models,body,true);const detail=el('details'),summary=el('summary','参数与来源'),pre=el('pre',JSON.stringify(prov,null,2));detail.append(summary,pre);body.append(detail);$('tasks').append(row);}
for(const row of sections){const tr=el('tr'),cell=el('td'),button=el('button',`${fmt(row.start)}–${fmt(row.end)}`);button.onclick=()=>seek(row.start);cell.append(button);tr.append(cell,el('td',labels[row.label]||row.label),el('td',`${row.source||'未知'} · ${labels[row.status]||row.status||'估计'}`));$('sections').append(tr);}
if(!sections.length){const row=el('tr'),cell=el('td','尚无段落结果','empty');cell.colSpan=3;row.append(cell);$('sections').append(row);}
for(const warning of P.warnings||[])$('warnings').append(el('div',warning,'warning'));
if(!(P.warnings||[]).length)$('warnings').append(el('p','没有记录运行警告；仍需独立评估识别效果。','empty'));
const lyrics=P.lyrics_context||{};
for(const [key,title] of [['lyrics','歌词'],['themes','主题'],['narrative','叙事'],['imagery','意象'],['background','背景'],['sources','来源']]){const values=lyrics[key]||[];if(!values.length)continue;const detail=el('details'),sum=el('summary',`${title} · ${values.length}`);detail.append(sum);for(const value of values){detail.append(el('p',itemText(value),'description'));}$('lyrics').append(detail);}
if(P.publication?.redacted){text($('download'),'下载公开数据摘要');$('lyrics').replaceChildren(el('p',`公开预览已隐藏原文 · ${P.publication.lyrics_line_count} 行 / ${P.publication.lyrics_word_count} 个已对齐字词`,'empty'));text($('audioNotice'),'公开预览不附带原曲音频。可选择本地对应原曲，仅在本机播放。');}else if(!$('lyrics').children.length)$('lyrics').append(el('p','未提供或未成功提取歌词与背景。','empty'));
let current=0,objectUrl=null,viewStart=0,viewEnd=Math.min(30,duration),lastDescription=Symbol();
const audio=$('audio'),canvas=$('chart'),ctx=canvas.getContext('2d');
audio.addEventListener('error',()=>text($('audioNotice'),P.publication?.redacted?'公开预览不附带原曲音频。可选择本地对应原曲，仅在本机播放。':'浏览器未载入音频。请选择对应的本地原曲；时间轴仍可独立拖动查看。'));
audio.addEventListener('loadedmetadata',()=>{const delta=Math.abs(audio.duration-duration);text($('audioNotice'),delta>0.1?`所选音频时长与分析相差 ${delta.toFixed(2)} 秒，请确认版本。`:'音频已载入本地播放器。播放或拖动时间轴查看同步估计结果。');});
$('audioFile').addEventListener('change',event=>{const file=event.target.files[0];if(!file)return;if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(file);audio.src=objectUrl;audio.load();});
$('download').onclick=()=>{const url=URL.createObjectURL(new Blob([JSON.stringify(P,null,2)],{type:'application/json'})),a=el('a');a.href=url;a.download=String(P.track_id).replace(/[^a-zA-Z0-9_-]/g,'_')+(P.publication?.redacted?'-public-summary.json':'-song-package.json');a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
function seek(time){current=Math.min(duration,Math.max(0,time));if(audio.readyState>0&&Number.isFinite(audio.duration))audio.currentTime=Math.min(current,audio.duration);update();}
$('seek').addEventListener('input',()=>seek(Number($('seek').value)));$('window').addEventListener('change',draw);
audio.addEventListener('timeupdate',()=>{current=Math.min(duration,audio.currentTime);update();});
let raf=0;function tick(){current=Math.min(duration,audio.currentTime);update();if(!audio.paused)raf=requestAnimationFrame(tick);}audio.addEventListener('play',()=>{cancelAnimationFrame(raf);raf=requestAnimationFrame(tick);});audio.addEventListener('pause',()=>cancelAnimationFrame(raf));
function showDescription(d){if(d===lastDescription)return;lastDescription=d;$('description').replaceChildren();if(!d){$('description').append(el('p','此时没有描述结果。','empty'));return;}$('description').append(el('p',`${fmt(d.applies_start)}–${fmt(d.applies_end)} · ${d.section_id||'未关联段落'}`,'muted'));for(const [key,title] of [['genre','曲风'],['instruments','乐器'],['textures','质感']]){const group=el('div'),tags=el('div',undefined,'chips');group.append(el('span',title,'label'));for(const value of d[key]||[])tags.append(tagChip(value,key==='textures'));if(!tags.children.length)tags.append(el('span','未确定','empty'));group.append(tags);$('description').append(group);}const emo=d.emotion;const emotionText=emo?`效价：${emo.valence==null?'未知':Number(emo.valence).toFixed(2)} ｜ 唤醒度：${emo.arousal==null?'未知':Number(emo.arousal).toFixed(2)}${emo.scale?' ｜ 量表：'+(typeof emo.scale==='string'?emo.scale:JSON.stringify(emo.scale)):''}`:'未生成效价 / 唤醒度；不会以声学能量冒充情绪。';$('description').append(el('p',emotionText,'description'));if(emo?.source)$('description').append(el('p','情绪来源：'+sourceName(emo.source),'engine'));if(emo&&(emo.predicted_moods||emo.moods)){const moodTags=el('div',undefined,'chips');for(const mood of (emo.predicted_moods||emo.moods))moodTags.append(tagChip(mood,true));$('description').append(moodTags);}if(d.caption)$('description').append(el('p',d.caption,'description'));modelSummary(d.model_results,$('description'),false);$('description').append(el('p','标签可来自不同模型，冲突候选保留来源；模型分数不是识别准确率。','tip'));}
function lyricState(time){
  if(P.publication?.redacted)return {hidden:true};
  const validSpan=row=>Number.isFinite(row.start)&&Number.isFinite(row.end)&&row.start<=time&&time<row.end;
  const line=(lyrics.lyrics||[]).find(validSpan);
  if(!line)return {line:null,word:null};
  const word=(line.words||[]).find(validSpan)||null;
  return {line,word};
}
function showLyric(time){
  const state=lyricState(time);
  if(state.hidden){text($('currentLyric'),`公开预览已隐藏原文 · ${P.publication.lyrics_line_count} 行 / ${P.publication.lyrics_word_count} 个已对齐字词`);$('currentWord').hidden=true;text($('lyricTiming'),'');return;}
  $('currentWord').hidden=false;
  if(!state.line){text($('currentLyric'),'此时无已对齐歌词 / 句间空白');text($('currentWord'),'—');text($('lyricTiming'),'');return;}
  const line=state.line,word=state.word,span=word||line,source=String(line.source||'');
  text($('currentLyric'),line.text||'');
  text($('currentWord'),word?(word.text||word.token||'—'):'—');
  const origin=source.toLowerCase().includes('sofa')?'SOFA 估计':(source||'来源未记录')+(line.timing_status?' · '+(labels[line.timing_status]||line.timing_status):'');
  text($('lyricTiming'),`${origin} · 原曲 ${span.start.toFixed(2)}–${span.end.toFixed(2)} 秒${word?'':((line.words||[]).length?' · 词间空白':' · 仅行级时间')}`);
}
function update(){showLyric(current);text($('time'),`${fmt(current)} / ${fmt(duration)}`);$('seek').value=current;const n=at(notes,current),f=nearestFrame(current),transition=at((melodyReview?.decisions||[]).filter(d=>d.resolution==='no_stable_instrumental_lead'),current);const pitch=f&&f.valid?f.midi_float:(n?n.midi:null);text($('currentPitch'),Number.isFinite(pitch)?`${names[((Math.round(pitch)%12)+12)%12]}${Math.floor(Math.round(pitch)/12)-1} · ${pitch.toFixed(1)}`:(transition?'器乐 → 人声过渡':'未知 / 停顿'));text($('pitchSource'),n?`候选来源：${n.source==='score_assisted_gap'?'谱图辅助补充（估计）':(n.source==='instrumental_lead_reconciled'?'吉他高声部（复核）':n.source||'未记录')}`:(transition?'此段无稳定主旋律，不强制填入音高':'无有效音符候选'));const chord=at(chords,current);text($('currentChord'),chord?(chord.status==='estimated'?chord.label:labels[chord.status]||chord.status):'未知');text($('chordPcs'),chord&&chord.pitch_classes.length?'组成音：'+chord.pitch_classes.map(pc=>names[pc]).join(' · '):'');const section=at(sections,current);text($('currentSection'),section?(labels[section.label]||section.label):'未知');const tempo=at(P.rhythm.tempo||[],current);text($('currentTempo'),tempo?`${tempo.bpm.toFixed(1)} BPM · ${tempo.beat_unit}`:'速度未确定');showDescription(at(descriptions,current,'applies_start','applies_end'));draw();}
function draw(){const rect=canvas.getBoundingClientRect(),w=Math.max(240,rect.width),h=360,dpr=window.devicePixelRatio||1;if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)){canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);const full=$('window').value==='all',span=full?duration:Math.min(Number($('window').value),duration);viewStart=full?0:Math.max(0,Math.min(current-span*0.3,duration-span));viewEnd=viewStart+span;const left=48,right=w-14,x=t=>left+(t-viewStart)/Math.max(0.001,span)*(right-left);const pitches=notes.filter(n=>n.end>=viewStart&&n.start<=viewEnd&&Number.isFinite(n.midi)).map(n=>n.midi);const min=Math.min(48,...pitches)-2,max=Math.max(84,...pitches)+2,y=p=>185-(p-min)/(max-min)*155;ctx.font='11px system-ui';ctx.strokeStyle='#e0e7f0';ctx.fillStyle='#7a8797';for(let p=Math.ceil(min/12)*12;p<=max;p+=12){ctx.beginPath();ctx.moveTo(left,y(p));ctx.lineTo(right,y(p));ctx.stroke();ctx.fillText('C'+(p/12-1),8,y(p)+4);}ctx.fillText('音高',8,18);for(let i=0;i<=6;i++){let t=viewStart+span*i/6;ctx.fillText(fmt(t),Math.min(right-32,x(t)),346);ctx.beginPath();ctx.moveTo(x(t),28);ctx.lineTo(x(t),326);ctx.stroke();}ctx.save();ctx.beginPath();ctx.rect(left,0,right-left,330);ctx.clip();for(const n of notes){if(n.end<viewStart||n.start>viewEnd||!Number.isFinite(n.midi))continue;ctx.fillStyle='#bbd4ee';ctx.fillRect(x(n.start),y(n.midi)-3,Math.max(1,x(n.end)-x(n.start)),6);}ctx.strokeStyle='#3979b6';ctx.lineWidth=1.6;ctx.beginPath();let connected=false,previousTime=null;for(const f of contour){if(f.time<viewStart-0.15||f.time>viewEnd+0.15)continue;if(!f.valid||!Number.isFinite(f.midi_float)){connected=false;continue;}if(!connected||previousTime===null||f.time-previousTime>0.15)ctx.moveTo(x(f.time),y(f.midi_float));else ctx.lineTo(x(f.time),y(f.midi_float));connected=true;previousTime=f.time;}ctx.stroke();for(let i=0;i<chords.length;i++){const c=chords[i];if(c.end<viewStart||c.start>viewEnd)continue;const start=Math.max(left,x(c.start)),end=Math.min(right,x(c.end));ctx.fillStyle=c.status==='estimated'?(i%2?'#e5daf2':'#d5c5e8'):'#edf0f5';ctx.fillRect(start,211,Math.max(1,end-start-1),33);if(end-start>22){ctx.save();ctx.beginPath();ctx.rect(start,211,end-start,33);ctx.clip();ctx.fillStyle='#594873';ctx.fillText(c.status==='estimated'?c.label:(labels[c.status]||'未知'),start+5,232);ctx.restore();}}for(const beat of beats){if(beat.time<viewStart||beat.time>viewEnd)continue;ctx.strokeStyle=beat.is_downbeat?'#166b54':'#87b8ab';ctx.lineWidth=beat.is_downbeat?2:1;ctx.beginPath();ctx.moveTo(x(beat.time),beat.is_downbeat?259:272);ctx.lineTo(x(beat.time),293);ctx.stroke();}for(let i=0;i<sections.length;i++){const s=sections[i];if(s.end<viewStart||s.start>viewEnd)continue;ctx.fillStyle=i%2?'#e6edf6':'#d6e3f3';ctx.fillRect(x(s.start),308,x(s.end)-x(s.start)-1,17);}ctx.strokeStyle='#c65c59';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x(current),6);ctx.lineTo(x(current),328);ctx.stroke();ctx.restore();ctx.fillStyle='#7a8797';ctx.fillText('和弦',8,230);ctx.fillText('节拍',8,282);ctx.fillText('段落',8,321);}
canvas.addEventListener('click',event=>{const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left;seek(viewStart+(x-48)/(rect.width-62)*(viewEnd-viewStart));});new ResizeObserver(draw).observe(canvas);update();
</script></body></html>'''
