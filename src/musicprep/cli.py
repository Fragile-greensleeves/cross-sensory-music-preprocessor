import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(prog="musicprep", description="自动音乐预处理：事件、描述与统一时间轴")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("analyze", help="自动分析歌曲并生成JSON和交互报告")
    run.add_argument("audio", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--track-id")
    run.add_argument("--profile", choices=("auto", "baseline"), default="auto")
    run.add_argument("--config", type=Path)
    run.add_argument("--lyrics", type=Path)
    run.add_argument("--background", type=Path)
    run.add_argument("--force", action="store_true", help="重新计算任务，不复用模型结果缓存")
    run.add_argument("--require-complete", action="store_true", help="部分结果时退出码为2")
    check = sub.add_parser("validate", help="校验歌曲数据包")
    check.add_argument("package", type=Path)
    report = sub.add_parser("report", help="从歌曲数据包重新生成本地报告")
    report.add_argument("package", type=Path)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--audio", type=Path)
    report.add_argument("--public", action="store_true", help="导出可分享的诊断报告，去除歌词原文、本地路径和音频链接")
    sub.add_parser("doctor", help="检查依赖，不下载模型")
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            from .pipeline import environment_versions
            data = {"python": sys.executable, "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"),
                    "versions": environment_versions(), "modules": {name: importlib.util.find_spec(name) is not None for name in
                    ("numpy", "scipy", "librosa", "soundfile", "jsonschema", "demucs", "torchcrepe", "basic_pitch", "beat_this", "allin1", "essentia", "transformers")}}
            print(json.dumps(data, ensure_ascii=False, indent=2))
            if not data["ffmpeg"] or not data["ffprobe"]:
                raise SystemExit(1)
        elif args.command == "analyze":
            from .pipeline import analyze
            result = analyze(args.audio, args.output, args.track_id, args.profile, args.config, args.lyrics, args.background, args.force,
                             progress=lambda task, status: print(f"[{task}] {status}", file=sys.stderr, flush=True))
            print(json.dumps({"status": result["run"]["status"], "package": str(args.output / "song-package.json"),
                              "report": str(args.output / "report.html")}, ensure_ascii=False, indent=2))
            if result["run"]["status"] == "failed":
                raise SystemExit(1)
            if args.require_complete and result["run"]["status"] != "complete":
                raise SystemExit(2)
        else:
            from .validation import validate_package
            package = json.loads(args.package.read_text(encoding="utf-8"))
            errors = validate_package(package)
            if errors:
                print(json.dumps({"valid": False, "errors": errors}, ensure_ascii=False, indent=2))
                raise SystemExit(1)
            if args.command == "report":
                from .report import write_report
                write_report(package, args.output, args.audio, public=args.public)
                print(args.output)
            else:
                print(json.dumps({"valid": True, "status": package["run"]["status"]}))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"musicprep: {exc}\n")
