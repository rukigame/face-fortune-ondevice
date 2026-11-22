"""Connect existing Face Fortune measurements to the independent reading rules.

Only precomputed measurements are used here. Life stories are generated locally
by deterministic rules. Vision and Animal/Pokemon are unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import face_fortune_labels18 as reading

ROOT = Path(__file__).resolve().parent


def _summaries(doc: dict[str, Any]) -> list[str]:
    """Select measured eye type and a few existing rule-based explanations."""
    sections = doc.get("sections", [])
    for title, blocks in sections:
        if title.startswith("한눈에 보기"):
            return [str(x["text"]) for x in blocks
                    if isinstance(x, dict) and x.get("t") == "p" and x.get("text")][:5]
    return []


def _simplify_rules(rules: dict[str, Any]) -> dict[str, Any]:
    if rules.get("status") not in {"ok", "no_reference"}:
        return {"status": "missing_rules", "message": "Optional gwansang_rules.json was not provided"}
    return {
        "status": rules.get("status"),
        "ohaeng": rules.get("ohaeng"),
        "grade": rules.get("grade"),
        "zones": rules.get("zones", []),
        "palaces": rules.get("palaces", []),
        "fortunes": rules.get("fortunes", []),
    }



SUMMARY_VERSION = 7


def _overall_summary(labels: dict[str, Any], reference_available: bool) -> dict[str, Any]:
    """Summarize only measured labels and the team's stored traditional rules."""
    items = labels.get("items") or {}
    rules = labels.get("rules") or {}
    shape = rules.get("ohaeng") or {}
    eye = labels.get("eye_shape") or {}

    def label(key: str) -> str | None:
        row = items.get(key)
        if not isinstance(row, dict) or row.get("reliability") == "low":
            return None
        return str(row.get("label")) if row.get("label") else None

    segments = []
    for key, prefix in (("eye_tilt", "눈꼬리는"),
                        ("nose_len", "코 길이는"),
                        ("mouth_w", "입 크기는"),
                        ("brow_tilt", "눈썹 각도는")):
        value = label(key)
        if value:
            segments.append(f"{prefix} {value}")
    segments = segments[:3]

    paragraphs = []
    if shape.get("name") and shape.get("tag"):
        paragraphs.append(
            f"얼굴 비율을 바탕으로 전통 오행 분류의 ‘{shape['name']}’ "
            f"({shape['tag']})에 가깝게 분류됐어요."
        )
    elif label("face_ratio"):
        paragraphs.append(f"얼굴 가로세로 비율은 {label('face_ratio')}으로 측정됐어요.")

    if eye.get("distinct") and eye.get("best"):
        paragraphs.append(f"눈 형태는 비교 가능한 전통 눈 유형 중 ‘{eye['best']}’에 가장 가까웠어요.")

    if segments:
        if len(segments) == 1:
            paragraphs.append(segments[0] + "으로 측정됐어요.")
        else:
            paragraphs.append(", ".join(segments[:-1]) + ", " + segments[-1] + "으로 측정됐어요.")

    if shape.get("one"):
        paragraphs.append(f"전통 관상에서는 이러한 얼굴형을 ‘{shape['one']}’으로 비유하기도 해요.")

    if not paragraphs:
        paragraphs.append("얼굴의 세부 측정값을 확인한 뒤 관상 이야기를 구성할 수 있어요.")

    tags = []
    if shape.get("name"):
        tags.append({"name": "오행 얼굴형", "value": str(shape["name"])})
    if eye.get("distinct") and eye.get("best"):
        tags.append({"name": "눈 유형", "value": str(eye["best"])})
    for key, name in (("eye_tilt", "눈꼬리"), ("nose_len", "코"), ("mouth_w", "입")):
        if label(key):
            tags.append({"name": name, "value": label(key)})

    return {
        "headline": "전통 관상 속 나의 얼굴 이야기",
        "paragraphs": paragraphs,
        "tags": tags[:5],
        "basis": "team_rules_plus_vision_labels",
        "reference_available": reference_available,
        "notice": "전통적인 표현을 재구성한 오락용 풀이예요. 실제 성격이나 미래를 판단하지 않습니다.",
    }



def _life_story(labels: dict[str, Any]) -> dict[str, Any]:
    """Compose distinct life stories from multiple independently measured rules."""
    from fortune_story_composer import compose
    return compose(labels)


def generate(capture: Path) -> dict[str, Any]:
    """Generate 18 labels and traditional interpretation from saved numeric JSON."""
    source = capture / "analysis_result.json"
    if not source.is_file():
        return {"status":"unavailable", "reason":"analysis_result.json missing"}
    a = json.loads(source.read_text(encoding="utf-8"))
    m = a.get("feature_values_ipd")
    if not isinstance(m, dict) or not m:
        return {"status":"unavailable", "reason":"feature_values_ipd missing"}

    ref = reading.find_reference()
    labels = a.get("face_labels")
    if not isinstance(labels, dict) or labels.get("status") != "measured":
        labels = reading.face_labels(
            m, a.get("eyelid"), ref=ref,
            std=a.get("feature_std_ipd"),
            n=(a.get("quality") or {}).get("sample_count"),
            tilts=reading.eye_tilts(a.get("eyes_wflw"))
        )
    if not labels:
        return {"status":"unavailable", "reason":"No face labels could be computed"}

    # Recalculate optional detailed rules even for previously saved labels.
    labels = dict(labels)
    labels["rules"] = reading.rules_report(m, ref)

    doc = reading.build_reading(
        {**a, "face_labels": labels},
        folder_name=capture.name,
        ref=ref,
    )
    eye = labels.get("eye_shape") or {}
    items = labels.get("items") or {}
    eyelid = a.get("eyelid") or {}
    if not isinstance(eyelid, dict):
        eyelid = {}

    return {
        "status": "ready",
        "summary_version": SUMMARY_VERSION,
        "summary": _overall_summary(labels, ref is not None),
        "life_story": _life_story(labels),
        "source": "face_fortune_labels18.py",
        "rules_source": "embedded reading rules + optional gwansang_rules.json",
        "reference_available": ref is not None,
        "label_scale": "5_levels",
        "eye_shape": {
            "name": eye.get("best"),
            "similarity": eye.get("similarity"),
            "distinct": eye.get("distinct", False),
            "description": eye.get("description"),
            "ranking": eye.get("ranking", []),
            "caveat": eye.get("caveat"),
        },
        "eyelid": {
            "status": eyelid.get("status", "unavailable"),
            "label": eyelid.get("label"),
            "validated": eyelid.get("validated", False),
        },
        "items": [
            {"key": k, "name": x.get("name_ko", k),
             "label": x.get("label", ""),
             "level": x.get("level"),
             "of": x.get("of", 5),
             "basis": x.get("basis"),
             "reliability": x.get("reliability")}
            for k,x in items.items()
        ],
        "lines": _summaries(doc or {}),
        "traditional_luck": labels.get("luck", []),
        "rules": _simplify_rules(labels.get("rules") or {}),
        "disclaimer": "전통 관상의 오락용 풀이이며, 성격·능력·건강·미래에 대한 과학적 판단이 아닙니다.",
    }


def load_or_generate(capture: Path) -> dict[str, Any]:
    """Load or generate a purely rule-based reading, with no language model."""
    p = capture / "fortune_result.json"
    if p.is_file():
        try:
            cached = json.loads(p.read_text(encoding="utf-8"))
            rules_path = ROOT / "gwansang_rules.json"
            rule_stale = (rules_path.is_file()
                          and cached.get("rules", {}).get("status") != "ok")
            if cached.get("summary_version") == SUMMARY_VERSION and not rule_stale:
                return cached
            reading._RULES.clear()
        except (json.JSONDecodeError, OSError):
            pass
    result = generate(capture)
    if result.get("status") == "ready":
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build a reading from saved face measurements")
    parser.add_argument("capture", type=Path, help="Path to a captures/YYYYMMDD_HHMMSS folder")
    args = parser.parse_args()
    result = load_or_generate(args.capture)
    print(json.dumps({"status": result.get("status"),
                      "eye_shape": result.get("eye_shape", {}).get("name"),
                      "items": len(result.get("items", [])),
                      "rules": result.get("rules", {}).get("status")}, ensure_ascii=False, indent=2))
