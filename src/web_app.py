from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

import face_fortune as ff

ROOT = Path(__file__).resolve().parent
CAPTURE_DIR = ROOT / "captures"
STATIC_IMAGES = ROOT / "static" / "images"
SAFE_IMAGE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def match_image_url(category: str, name: str | None) -> str | None:
    # Do not construct arbitrary filesystem paths from result JSON values.
    if category not in {"animals", "pokemon"} or not name:
        return None
    if not SAFE_IMAGE_NAME.fullmatch(str(name)):
        return None
    for basename in (str(name), str(name).lower()):
        for ext in (".png", ".webp", ".jpg", ".jpeg"):
            path = STATIC_IMAGES / category / f"{basename}{ext}"
            if path.is_file():
                return f"/static/images/{category}/{basename}{ext}"
    return None


def latest_saved_capture() -> Path | None:
    if not CAPTURE_DIR.is_dir():
        return None
    captures = [p for p in CAPTURE_DIR.iterdir()
                if p.is_dir() and (p / "analysis_result.json").is_file()]
    return max(captures, key=lambda p: p.name) if captures else None

app = Flask(__name__)


def _kiosk_authenticated() -> bool:
    # Password is optional for backward compatibility with offline kiosks.
    # Enable only on trusted networks; plain HTTP does not encrypt credentials.
    password = os.environ.get("FORTUNE_ACCESS_PASSWORD", "")
    if not password:
        return True
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    try:
        raw = base64.b64decode(auth[6:], validate=True).decode("utf-8")
        username, separator, supplied = raw.partition(":")
    except (ValueError, UnicodeError):
        return False
    return bool(separator) and hmac.compare_digest(username, "fortune") and hmac.compare_digest(supplied, password)


@app.before_request
def require_kiosk_password():
    if not _kiosk_authenticated():
        return Response("Authentication required", status=401,
                        headers={"WWW-Authenticate": 'Basic realm="Face Fortune"',
                                 "Cache-Control": "no-store"})


@app.after_request
def prevent_sensitive_cache(response):
    if request.path.startswith(("/api/", "/capture/", "/video_feed")):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
    return response


STATUS_KO = {
    "IDLE": "분석 준비 완료",
    "ONE FACE REQUIRED": "얼굴을 카메라에 맞춰 주세요",
    "KEEP FACE INSIDE FRAME": "얼굴 전체가 화면 안에 들어오게 해 주세요",
    "MOVE CLOSER": "카메라에 조금 더 가까이 와 주세요",
    "LOOK STRAIGHT AT CAMERA": "카메라를 정면으로 봐 주세요",
    "BLINK - HOLD ON": "눈을 뜬 상태로 잠시 유지해 주세요",
    "LOOK AT THE CAMERA LENS": "카메라 렌즈를 바라봐 주세요",
    "COLLECTING": "좋아요. 얼굴 특징을 측정하고 있어요",
    "READY - AUTO SAVING": "측정 완료. 결과를 저장하고 있어요",
    "UNSTABLE - R TO RETRY": "측정이 흔들렸어요. 다시 시도해 주세요",
    "SAVING": "결과를 분석하고 있어요",
    "DONE": "분석이 완료됐어요",
    "ERROR": "분석 중 오류가 발생했어요",
}

KEY_METRICS = [
    ("face_ratio", "얼굴 세로 비율"),
    ("eye_len", "눈 길이"),
    ("eye_tilt", "눈꼬리 기울기"),
    ("eye_gap", "눈 사이 간격"),
    ("nose_len", "코 길이"),
    ("ala_w", "콧볼 폭"),
    ("mouth_w", "입 너비"),
    ("lip_h", "입술 두께"),
    ("jaw_angle", "턱선 각도"),
    ("jaw_taper", "턱 좁아짐"),
]


def level_descriptor(level: int) -> str:
    if level <= 2:
        return "매우 낮은 편"
    if level <= 4:
        return "낮은 편"
    if level <= 6:
        return "평균 범위"
    if level <= 8:
        return "높은 편"
    return "매우 높은 편"


def fortune_lines(metrics: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    recipes = [
        ("eye_len", "눈의 가로 길이"),
        ("eye_tilt", "눈꼬리 기울기"),
        ("nose_len", "코 길이"),
        ("mouth_w", "입의 가로 폭"),
        ("jaw_taper", "턱선이 좁아지는 정도"),
    ]
    for key, label in recipes:
        item = metrics.get(key)
        if not item:
            continue
        level = int(item.get("level", 5))
        lines.append(
            f"{label}는 기준 분포에서 {level}/10 단계로, {level_descriptor(level)}에 해당합니다."
        )
    return lines[:4]


class FaceFortuneEngine:
    def __init__(self, camera_index: int = 1, eye_refine: bool = True):
        self.camera_index = camera_index
        self.eye_refine = eye_refine
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.cap = None
        self.landmarker = None
        self.hair = None
        self.reference_table = None

        self.frame_jpeg: bytes | None = None
        self.camera_ready = False
        self.scan_active = False
        self.status = "IDLE"
        self.count = 0
        self.error: str | None = None
        self.latest_capture: Path | None = None

        self.face_ok = False
        self.pose_ok = False
        self.gaze_ok = False
        self.hair_ok = False

        self._reset_measurement_state()

    def _reset_measurement_state(self):
        self.measurement = ff.Measurement()
        self.coordinates: list[np.ndarray] = []
        self.analyses: list[dict[str, Any]] = []
        self.cnn_frames: list[np.ndarray] = []
        self.cnn_face_crops: list[np.ndarray] = []
        self.cnn_face_candidates: list[dict[str, Any]] = []
        self.last_hairline = None
        self.invalid_streak = 0
        self.ready_frame = None
        self.ready_screen = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def shutdown(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=3)

    def start_scan(self):
        with self.lock:
            if not self.camera_ready:
                return False, "카메라가 아직 준비되지 않았습니다."
            if self.scan_active:
                return False, "이미 분석 중입니다."
            self._reset_measurement_state()
            self.scan_active = True
            self.status = "ONE FACE REQUIRED"
            self.count = 0
            self.error = None
            self.latest_capture = None
            return True, None

    def cancel_scan(self):
        with self.lock:
            self.scan_active = False
            self.status = "IDLE"
            self.count = 0
            self._reset_measurement_state()

    def _initialize(self):
        self.reference_table = ff.load_reference_10level(ff.REFERENCE_CSV)
        self.landmarker = ff.Landmarker(eye_refine=self.eye_refine)
        self.hair = ff.Hairline(enabled=True)
        self.cap = ff.open_camera(self.camera_index)
        self.camera_ready = True

    def _encode_preview(self, frame, pts, status, hairline):
        # User-facing web preview intentionally shows the clean camera frame only.
        # Landmarks, boxes, hairline lines and debug HUD remain internal.
        preview = frame.copy()

        ok, buf = cv2.imencode(
            ".jpg",
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), 88],
        )
        if ok:
            self.frame_jpeg = buf.tobytes()

    def _run(self):
        try:
            self._initialize()
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("Camera read failed")
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                pts = self.landmarker(frame)
                analysis = None
                valid = False
                blink = False
                status = "ONE FACE REQUIRED"
                hairline_for_preview = self.last_hairline

                self.face_ok = pts is not None
                self.pose_ok = False
                self.gaze_ok = False
                self.hair_ok = self.last_hairline is not None

                if pts is not None:
                    x1, y1, x2, y2 = ff.face_bbox(pts)
                    fits = x1 > width * 0.02 and y1 > height * 0.02 and x2 < width * 0.98 and y2 < height * 0.98
                    large = (pts[ff.P["cheek_r"], 0] - pts[ff.P["cheek_l"], 0]) >= width * ff.MIN_FACE_WIDTH_RATIO
                    eyes = {side: ff.eye_metrics(pts, side) for side in ff.EYES}
                    blink = not ff.eyes_open(eyes)
                    pose = ff.pose_of(pts)
                    frontal = ff.is_frontal(pose)
                    gaze_ok = True if blink else ff.gaze_centered(eyes)
                    valid = fits and large and frontal and gaze_ok and not blink

                    self.pose_ok = bool(frontal)
                    self.gaze_ok = bool(gaze_ok and not blink)

                    if not fits:
                        status = "KEEP FACE INSIDE FRAME"
                    elif not large:
                        status = "MOVE CLOSER"
                    elif not frontal:
                        status = "LOOK STRAIGHT AT CAMERA"
                    elif blink:
                        status = "BLINK - HOLD ON"
                    elif not gaze_ok:
                        status = "LOOK AT THE CAMERA LENS"
                    else:
                        status = "COLLECTING"

                    if self.scan_active and valid:
                        if self.hair.ok and (self.last_hairline is None or len(self.measurement.samples) % ff.FOREHEAD_EVERY == 0):
                            self.last_hairline = self.hair(frame, pts)
                        self.hair_ok = self.last_hairline is not None
                        hairline_for_preview = self.last_hairline
                        analysis = ff.analyze_frame(
                            pts,
                            self.last_hairline,
                            self.hair.last_mask if self.last_hairline is not None else None,
                        )
                    else:
                        analysis = ff.analyze_frame(pts, None)

                if self.scan_active:
                    if blink:
                        pass
                    elif not valid:
                        self.invalid_streak += 1
                        if self.invalid_streak >= ff.RESET_AFTER_INVALID:
                            self._reset_measurement_state()
                            self.count = 0
                    else:
                        self.invalid_streak = 0
                        before = len(self.measurement.samples)
                        merged = dict(analysis["feature_values"])
                        merged.update({"ipd__" + k: v for k, v in analysis["feature_values_ipd"].items()})
                        completed = self.measurement.observe(merged, ff.face_center(pts), width)
                        after = len(self.measurement.samples)

                        if after > before:
                            self.coordinates.append(pts.copy())
                            self.analyses.append(analysis)
                            face_crop = ff.cnn_face_crop(frame, pts)
                            self.cnn_face_candidates.append({
                                "sample_number": int(after),
                                "crop": face_crop.copy(),
                                "sharpness": ff.cnn_face_sharpness(face_crop),
                            })
                            if after in ff.CNN_MULTI_FRAME_SAMPLE_NUMBERS:
                                self.cnn_frames.append(frame.copy())
                                self.cnn_face_crops.append(face_crop.copy())

                        if after < before:
                            self.coordinates.clear()
                            self.analyses.clear()
                            self.cnn_frames.clear()
                            self.cnn_face_crops.clear()
                            self.cnn_face_candidates.clear()
                            if after == 1:
                                self.coordinates.append(pts.copy())
                                self.analyses.append(analysis)
                                face_crop = ff.cnn_face_crop(frame, pts)
                                self.cnn_face_candidates.append({
                                    "sample_number": int(after),
                                    "crop": face_crop.copy(),
                                    "sharpness": ff.cnn_face_sharpness(face_crop),
                                })
                                if after in ff.CNN_MULTI_FRAME_SAMPLE_NUMBERS:
                                    self.cnn_frames.append(frame.copy())
                                    self.cnn_face_crops.append(face_crop.copy())

                        self.count = len(self.measurement.samples)

                        if self.count == ff.TARGET_FRAMES:
                            if self.measurement.stable:
                                status = "READY - AUTO SAVING"
                                self.ready_frame = frame.copy()
                            else:
                                status = "UNSTABLE - R TO RETRY"

                    if self.count == ff.TARGET_FRAMES and self.measurement.stable and self.ready_frame is not None:
                        self.status = "SAVING"
                        self._save_result(self.ready_frame)
                        self.scan_active = False
                        status = "DONE"

                with self.lock:
                    self.status = status if self.scan_active else ("DONE" if self.latest_capture else "IDLE")
                    self.count = len(self.measurement.samples) if self.scan_active else self.count
                    self._encode_preview(frame, pts, status, hairline_for_preview if self.scan_active else None)

                time.sleep(0.001)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.status = "ERROR"
                self.scan_active = False
                self.camera_ready = False
        finally:
            if self.cap is not None:
                self.cap.release()

    def _save_result(self, frame):
        # Build a screenshot compatible with the existing save function.
        try:
            screen = ff.draw(
                frame,
                self.coordinates[-1] if self.coordinates else None,
                self.analyses[-1] if self.analyses else None,
                "READY - AUTO SAVING",
                len(self.measurement.samples),
                self.last_hairline,
            )
        except Exception:
            screen = frame.copy()

        capture = ff.save_capture(
            CAPTURE_DIR,
            self.coordinates,
            self.analyses,
            frame,
            screen,
            self.measurement,
            self.reference_table,
            self.cnn_frames,
            self.cnn_face_crops,
            self.cnn_face_candidates,
        )
        try:
            ff.analyze_and_save_animal_pokemon(capture)
        except Exception as exc:
            print("Animal/Pokemon web analysis failed:", exc)
        try:
            from fortune_bridge import load_or_generate
            load_or_generate(capture)
        except Exception as exc:
            print("Fortune reading analysis failed:", exc)
        self.latest_capture = capture
        self.status = "DONE"

    def status_payload(self):
        with self.lock:
            return {
                "camera_ready": self.camera_ready,
                "scan_active": self.scan_active,
                "status": self.status,
                "status_ko": STATUS_KO.get(self.status, self.status),
                "count": self.count,
                "target": ff.TARGET_FRAMES,
                "progress": round(100 * self.count / max(ff.TARGET_FRAMES, 1)),
                "error": self.error,
                "capture_id": self.latest_capture.name if self.latest_capture else None,
                "checks": {
                    "face": self.face_ok,
                    "pose": self.pose_ok,
                    "gaze": self.gaze_ok,
                    "hairline": self.hair_ok,
                },
            }


engine: FaceFortuneEngine | None = None


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/video_feed")
def video_feed():
    def generate():
        while True:
            if engine is None:
                time.sleep(0.1)
                continue
            frame = engine.frame_jpeg
            if frame is None:
                time.sleep(0.08)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.04)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    return jsonify(engine.status_payload() if engine else {"camera_ready": False})


@app.post("/api/scan/start")
def api_scan_start():
    if engine is None:
        return jsonify({"ok": False, "error": "Engine not initialized"}), 500
    ok, error = engine.start_scan()
    return jsonify({"ok": ok, "error": error})


@app.post("/api/scan/cancel")
def api_scan_cancel():
    if engine:
        engine.cancel_scan()
    return jsonify({"ok": True})


def build_result_payload(capture: Path) -> dict[str, Any]:
    analysis_path = capture / "analysis_result.json"
    animal_path = capture / "animal_pokemon_result.json"

    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) if analysis_path.exists() else {}
    animal = json.loads(animal_path.read_text(encoding="utf-8")) if animal_path.exists() else {}

    reference = analysis.get("reference_10level", {})
    metrics_raw = reference.get("metrics", {}) if isinstance(reference, dict) else {}
    metrics = []
    for key, label in KEY_METRICS:
        item = metrics_raw.get(key)
        if not item:
            continue
        level = int(item.get("level", 5))
        metrics.append({
            "key": key,
            "label": label,
            "level": level,
            "of": 10,
            "descriptor": level_descriptor(level),
            "value": item.get("value"),
            "percentile_band": item.get("percentile_band"),
        })

    raw_matches = animal.get("animal_pokemon_top3", [])
    match_semantics = str(animal.get("animal_score_semantics", "")).lower()
    diagnostics = animal.get("animal_score_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    production = diagnostics.get("production_selection", {})
    if not isinstance(production, dict):
        production = {}
    fallback_used = bool(production.get("fallback_used", False))
    top3 = []
    for i, item in enumerate(raw_matches if isinstance(raw_matches,list) else [], start=1):
        if not isinstance(item,dict):
            continue
        animal_name = str(item.get("animal", ""))
        pokemon_name = str(item.get("pokemon", ""))

        # New team model uses an independent 0-100 match score, NOT a probability.
        # Legacy files used normalized shares. Keep that distinction explicit.
        has_match_score = not fallback_used and (
            item.get("animal_match_score") is not None
            or "independent" in match_semantics
        )
        score = item.get("animal_match_score")
        if score is None:
            score = item.get("animal_percent")
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
        top3.append({
            "rank": item.get("rank",i),
            "animal": animal_name,
            "pokemon": pokemon_name,
            "match_score": score,
            "score_kind": "match" if has_match_score else "legacy_share",
            "animal_image_url": match_image_url("animals",animal_name),
            "pokemon_image_url": match_image_url("pokemon",pokemon_name),
        })

    try:
        from fortune_bridge import load_or_generate
        fortune = load_or_generate(capture)
    except Exception as exc:
        print("Fortune reading payload failed:", exc)
        fortune = {"status": "unavailable", "reason": str(exc)}

    # The 10-level reference eye-gap metric is IPD-normalized, while the
    # physiognomy rule uses eye-gap / average eye width. Showing both as the
    # same 'eye gap' card was confusing, so the user-facing card follows the
    # dedicated 5-level rule used by the reading. Other metrics stay 10-level.
    if fortune.get("status") == "ready":
        gap_item = next((x for x in fortune.get("items", []) if x.get("key") == "eye_gap"), None)
        if isinstance(gap_item, dict) and gap_item.get("level") is not None:
            for metric in metrics:
                if metric.get("key") == "eye_gap":
                    metric["level"] = int(gap_item.get("level", 3))
                    metric["of"] = int(gap_item.get("of") or 5)
                    metric["descriptor"] = str(gap_item.get("label") or "표준")
                    metric["percentile_band"] = None
                    break

    return {
        "ready": True,
        "capture_id": capture.name,
        "image_url": f"/capture/{capture.name}/face_measurement.jpg",
        "screen_url": f"/capture/{capture.name}/measurement_screen.jpg",
        "quality": analysis.get("quality", {}),
        "metrics": metrics,
        "fortune": fortune,
        "fortune_lines": fortune.get("lines", []) if fortune.get("status") == "ready" else fortune_lines(metrics_raw),
        "top3": top3,
        "human_style": animal.get("human_style", {}),
        "animal_score_semantics": match_semantics,
    }


@app.get("/api/result")
def api_result():
    if engine is None or engine.latest_capture is None:
        return jsonify({"ready": False})
    return jsonify(build_result_payload(engine.latest_capture))


@app.get("/preview-result")
def preview_result():
    # Load the most recent saved capture without starting another scan.
    return render_template("index.html")


@app.get("/api/latest-result")
def api_latest_result():
    saved = latest_saved_capture()
    if saved is None:
        return jsonify({"ready": False})
    return jsonify(build_result_payload(saved))


@app.get("/capture/<capture_id>/<filename>")
def capture_file(capture_id: str, filename: str):
    # Never serve JSON, landmark arrays, stored face crops, or previous sessions.
    # A face photo can still be viewed by LAN clients unless access protection is on.
    if not SAFE_IMAGE_NAME.fullmatch(capture_id):
        return Response("Not found", status=404)
    if filename not in {"face_measurement.jpg", "measurement_screen.jpg"}:
        return Response("Not found", status=404)
    saved = latest_saved_capture()
    active = engine.latest_capture if engine is not None else None
    allowed_ids = {p.name for p in (saved, active) if p is not None}
    if capture_id not in allowed_ids:
        return Response("Not found", status=404)
    return send_from_directory(CAPTURE_DIR / capture_id, filename)


def main():
    global engine
    parser = argparse.ArgumentParser(description="Face Fortune Web UI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-eye-refine", action="store_true",
                        help="Use WFLW eye points without optional eye refinement")
    args = parser.parse_args()
    if args.debug and args.host not in ("127.0.0.1", "localhost", "::1"):
        parser.error("--debug is allowed only with a loopback --host")

    engine = FaceFortuneEngine(camera_index=args.camera,
                               eye_refine=not args.no_eye_refine)
    engine.start()

    print()
    print("FACE FORTUNE WEB")
    if not os.environ.get("FORTUNE_ACCESS_PASSWORD") and args.host not in ("127.0.0.1", "localhost", "::1"):
        print("PRIVACY WARNING: No access password. LAN clients can view the camera and latest result.")
        print("Set FORTUNE_ACCESS_PASSWORD on a trusted network or bind to 127.0.0.1.")
    print(f"Open: http://<Raspberry-Pi-IP>:{args.port}")
    print("Stop any other program using the USB camera before launching this server.")
    print()

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
