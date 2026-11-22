"""Offline-friendly quality checks for Face Fortune Raspberry Pi deployments.

Examples:
  python verify_pi.py consistency --last 5
  python verify_pi.py consistency --captures captures/20261007_100000 captures/20261007_100100

The consistency test compares repeated captures of ONE person. It cannot prove
that traditional fortune interpretations are accurate or facial recognition is valid.
No photographs, coordinates, or raw frames are uploaded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
CAPTURES = ROOT / "captures"
METRICS = (
    "face_ratio", "eye_len", "eye_gap", "nose_len", "ala_w",
    "mouth_w", "jaw_taper", "brow_tilt", "eye_tilt", "jaw_angle",
)


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def captures_available() -> list[Path]:
    if not CAPTURES.is_dir():
        return []
    return sorted((p for p in CAPTURES.iterdir()
                   if p.is_dir() and (p / "analysis_result.json").is_file()),
                  key=lambda p: p.name)


def capture_args(paths: list[str], last: int) -> list[Path]:
    if paths:
        candidates = [Path(p).expanduser().resolve() for p in paths]
    else:
        candidates = captures_available()[-last:]
    if len(candidates) < 2:
        raise ValueError("Need at least 2 captures of the SAME person; 5 is recommended")
    for capture in candidates:
        if not (capture / "analysis_result.json").is_file():
            raise ValueError(f"Missing analysis_result.json: {capture}")
    return candidates


def metric_values(doc: dict) -> dict[str, dict]:
    levels = (doc.get("reference_10level") or {}).get("metrics") or {}
    if not isinstance(levels, dict):
        return {}
    return {key: row for key, row in levels.items()
            if key in METRICS and isinstance(row, dict)}


def calculate_consistency(captures: list[Path]) -> dict:
    rows = [load(p / "analysis_result.json") for p in captures]
    record = []
    warnings = []
    for p, doc in zip(captures, rows):
        q = doc.get("quality") or {}
        item = {"capture": p.name, "samples": q.get("sample_count"),
                "stable": q.get("stable"), "average_cv": q.get("average_cv")}
        record.append(item)
        if q.get("stable") is False or (isinstance(q.get("sample_count"), (int, float)) and q["sample_count"] < 30):
            warnings.append(f"{p.name}: capture quality needs review")

    metrics = []
    for key in METRICS:
        observations = []
        for doc in rows:
            val = metric_values(doc).get(key, {})
            value, level = val.get("value"), val.get("level")
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                observations.append((float(value), int(level) if isinstance(level,(int,float)) else None))
        if len(observations) < 2:
            continue
        raw = [x[0] for x in observations]
        normalized_mean = sum(abs(v) for v in raw) / len(raw)
        change_pct = 100*(max(raw)-min(raw))/normalized_mean if normalized_mean > 1e-9 else None
        deciles = [x[1] for x in observations if x[1] is not None]
        level_span = max(deciles)-min(deciles) if deciles else None
        # A screening threshold, not a statistical guarantee of correctness.
        review = level_span is not None and level_span > 2
        if change_pct is not None and change_pct > 20 and key not in {"eye_tilt", "brow_tilt", "jaw_angle"}:
            review = True
        if review:
            warnings.append(f"{key}: repeated measurement varies substantially")
        metrics.append({"key":key,"count":len(observations),
                        "min":round(min(raw),5),"max":round(max(raw),5),
                        "relative_span_percent":round(change_pct,1) if change_pct is not None else None,
                        "decile_span":level_span,"review":review})

    comparison = {}
    if all((p / "fortune_result.json").is_file() for p in captures):
        stories = [load(p / "fortune_result.json") for p in captures]
        for key in ("face_ratio", "eye_gap", "nose_len", "mouth_w"):
            labels=[]
            for story in stories:
                items={v.get("key"):v for v in (story.get("items") or []) if isinstance(v,dict)}
                labels.append(items.get(key,{}).get("label"))
            comparison[key] = {"labels":labels, "distinct":len(set(str(x) for x in labels))}
            if comparison[key]["distinct"] > 2:
                warnings.append(f"{key}: categorical descriptions change across captures")
    report={"test":"same_person_repeatability_screening","count":len(captures),
            "assumption":"All captures must belong to the same person in similar conditions",
            "result":"review_needed" if warnings else "within_screening_limits",
            "captures":record,"metrics":metrics,"labels":comparison,
            "warnings":warnings,
            "disclaimer":"These are heuristic stability checks, not ground-truth accuracy or life prediction validation."}
    return report


def run_consistency(args):
    paths = capture_args(args.captures or [], args.last)
    print(f"Comparing {len(paths)} scans. Verify they are ALL of the same person.")
    report = calculate_consistency(paths)
    print(f"Screening result: {report['result']}")
    for r in report["metrics"]:
        span = "n/a" if r["decile_span"] is None else str(r["decile_span"])
        relative = "n/a" if r["relative_span_percent"] is None else f"{r['relative_span_percent']}%"
        flag = "REVIEW" if r["review"] else "ok"
        print(f"  {r['key']:<14} level range={span:<3} relative span={relative:<8} {flag}")
    for warning in report["warnings"]:
        print("  WARNING:", warning)
    if args.output:
        target=Path(args.output).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8")
        print("JSON report:",target)
    return 1 if report["warnings"] else 0


def main():
    p=argparse.ArgumentParser(description="Face Fortune Raspberry Pi release quality checks")
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("consistency",help="Compare repeated captures of the same person")
    c.add_argument("--last",type=int,default=5,help="Number of most recent captures (default: 5)")
    c.add_argument("--captures",nargs="+",help="Explicit capture directory paths")
    c.add_argument("--output",help="Optional JSON report path (no photographs)")
    args=p.parse_args()
    try:
        if args.last<2:
            raise ValueError("--last must be at least 2")
        return run_consistency(args)
    except (OSError,ValueError,json.JSONDecodeError) as exc:
        print("Validation setup error:",exc,file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
