"""
server.py – Flask application serving the single-page CPT demo.

Routes
------
GET  /                   → templates/index.html
GET  /encoder_stream     → MJPEG stream of the encoder signal
GET  /encoder_state      → JSON: current encoder symbol / phase info
POST /update_encoder     → JSON body: {message, window_type, symbol_frames}
GET  /decoder_settings   → JSON: current decoder settings
POST /update_decoder     → JSON body: partial settings patch
POST /decoder_process    → body: {frame: <base64 PNG/JPEG>}
                           returns: full decoder result JSON

Run with:
    python server.py

Then open http://localhost:5000 in a browser (preferably Chromium/Chrome
for WebRTC webcam support).

Dependencies (pip install):
    flask numpy opencv-python scipy Pillow
"""

import io, base64, time, threading, os
import numpy as np
import cv2
from flask import Flask, Response, render_template, request, jsonify


def _to_python(obj):
    """Recursively convert numpy scalars / arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

from encoder          import Encoder, DATA_FREQS, PILOT_FREQS
from decoder_processor import DecoderProcessor

# ─────────────────────────────────────────────────────────────────── init
app      = Flask(__name__)
enc      = Encoder(message="Google.com", window_type="hann", symbol_frames=90)
dec_proc = DecoderProcessor()

# ────────────────────────────────────────── video background state
_vid_lock    = threading.Lock()
_vid_cap     = None    # cv2.VideoCapture | None
_vid_enabled = False
_vid_opacity = 1.0     # signal opacity: 1.0 = fully opaque signal
UPLOAD_DIR   = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ────────────────────────────────────────────────── encoder MJPEG stream
_encode_lock   = threading.Lock()
_target_fps    = 30
_frame_period  = 1.0 / _target_fps


def _mjpeg_generator():
    """Infinite generator of MJPEG frames (signal | FFT side-by-side)."""
    while True:
        t0    = time.monotonic()
        # ── inject next video background frame (looping) ──────────────────
        with _vid_lock:
            if _vid_enabled and _vid_cap is not None:
                ok, vframe = _vid_cap.read()
                if not ok:                          # loop back to start
                    _vid_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, vframe = _vid_cap.read()
                enc.set_bg_frame(vframe if ok else None, _vid_opacity)
            else:
                enc.set_bg_frame(None)
        frame = enc.next_frame()   # 256×768 BGR (3 panels)
        # Scale up for display clarity (3 panels × 256 wide × 2× = 1536×512)
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
        ok, buf = cv2.imencode(
            ".jpg", frame,
            [cv2.IMWRITE_JPEG_QUALITY, 90]
        )
        if not ok:
            continue
        jpg = buf.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        elapsed = time.monotonic() - t0
        sleep   = max(0.0, _frame_period - elapsed)
        time.sleep(sleep)


# ─────────────────────────────────────────────────────────────── routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/encoder_stream")
def encoder_stream():
    return Response(
        _mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/encoder_state", methods=["GET"])
def encoder_state():
    state = enc.get_state()
    # Serialise numpy types
    out = {
        "message":        state["message"],
        "n_symbols":      state["n_symbols"],
        "symbol_idx":     state["symbol_idx"],
        "frame_in_sym":   state["frame_in_sym"],
        "symbol_frames":  state["symbol_frames"],
        "total_frames":   state["total_frames"],
        "current_bytes":  state["current_sym"],
        "current_phases": [float(p) for p in state["current_phases"]],
        "bytes_data":     state["bytes_data"],
        "pilot_freqs":    [list(f) for f in PILOT_FREQS],
        "data_freqs":     [list(f) for f in DATA_FREQS],
        "mod_mode":       state.get("mod_mode", "psk"),
        "cpm_h":          state.get("cpm_h", 0.5),
        "cpm_pulse":      state.get("cpm_pulse", "rect"),
    }
    return jsonify(_to_python(out))


@app.route("/update_encoder", methods=["POST"])
def update_encoder():
    body = request.get_json(force=True) or {}
    if "message" in body:
        enc.update_message(str(body["message"])[:200])
    if "window_type" in body and body["window_type"] in ("hann", "blackman", "none"):
        enc.update_window(body["window_type"])
    if "symbol_frames" in body:
        try:
            n = int(body["symbol_frames"])
            enc.update_symbol_frames(n)
        except (ValueError, TypeError):
            pass
    if "redundancy_mode" in body:
        enc.update_redundancy_mode(str(body["redundancy_mode"]))
    if "mod_mode" in body or "cpm_h" in body or "cpm_pulse" in body:
        enc.update_mod_mode(
            body.get("mod_mode",   enc._mod_mode),
            cpm_h    = body.get("cpm_h"),
            cpm_pulse= body.get("cpm_pulse"),
        )
    return jsonify({"status": "ok"})


@app.route("/update_freqs", methods=["POST"])
def update_freqs():
    """Update DATA_FREQS and/or PILOT_FREQS at runtime.
    Body: { data_freqs: [[ky,kx], ...], pilot_freqs: [[ky,kx], ...] }
    Each list must have exactly the same length as the current array."""
    body = request.get_json(force=True) or {}
    df = body.get("data_freqs")
    pf = body.get("pilot_freqs")
    try:
        enc.update_freqs(
            data_freqs  = [[int(v) for v in p] for p in df]  if df else None,
            pilot_freqs = [[int(v) for v in p] for p in pf]  if pf else None,
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"status": "ok",
                    "data_freqs":  [list(f) for f in DATA_FREQS],
                    "pilot_freqs": [list(f) for f in PILOT_FREQS]})


@app.route("/upload_video", methods=["POST"])
def upload_video():
    """Accept a multipart video file upload and prepare it as a looping background."""
    global _vid_cap
    f = request.files.get("video")
    if not f or not f.filename:
        return jsonify({"error": "no file"}), 400
    _, ext = os.path.splitext(f.filename)
    save_path = os.path.join(UPLOAD_DIR, f"bg_video{ext or '.mp4'}")
    f.save(save_path)
    with _vid_lock:
        if _vid_cap is not None:
            _vid_cap.release()
        _vid_cap = cv2.VideoCapture(save_path)
        if not _vid_cap.isOpened():
            _vid_cap = None
            return jsonify({"error": "could not open video file"}), 400
        fps      = _vid_cap.get(cv2.CAP_PROP_FPS) or 30
        n_frames = int(_vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return jsonify({"status": "ok", "filename": f.filename,
                    "fps": fps, "frames": n_frames})


@app.route("/video_bg_settings", methods=["POST"])
def video_bg_settings():
    """Enable/disable video background and set signal opacity (0.0–1.0)."""
    global _vid_enabled, _vid_opacity
    body = request.get_json(force=True) or {}
    with _vid_lock:
        if "enabled" in body:
            _vid_enabled = bool(body["enabled"])
        if "opacity" in body:
            _vid_opacity = float(max(0.0, min(1.0, body["opacity"])))
        return jsonify({"status": "ok",
                        "enabled": _vid_enabled,
                        "opacity": _vid_opacity,
                        "has_video": _vid_cap is not None})


@app.route("/detect_aruco", methods=["POST"])
def detect_aruco():
    """
    Lightweight ArUco-only detection.  Accepts a raw JPEG body (no base64).
    Returns aruco_pose JSON at full speed — no FFT work performed.
    """
    data = request.get_data()
    if not data:
        return jsonify({"detected": False, "error": "no data"})
    arr   = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"detected": False, "error": "decode failed"})
    result = dec_proc.detect_aruco_only(frame)
    return jsonify(_to_python(result))


@app.route("/decoder_settings", methods=["GET"])
def decoder_settings():
    s = dec_proc.get_settings()
    out = {k: (v if not isinstance(v, np.ndarray) else list(v))
           for k, v in s.items()}
    return jsonify(_to_python(out))


@app.route("/update_decoder", methods=["POST"])
def update_decoder():
    body = request.get_json(force=True) or {}
    # Whitelist of permitted keys and their types
    allowed = {
        "window_type":          str,
        "phase_offset":         float,
        "geo_correction":       bool,
        "pilot_search_r":       int,
        "temporal_avg":         int,
        "mag_threshold":        float,
        "fft_zoom_bins":        int,
        "carrier_gain":         float,
        "phase_nudge":          list,
        "redundancy_mode":      str,
        "aruco_decode_enabled": bool,
        "mod_mode":             str,
        "cpm_h":                float,
    }
    patch = {}
    for k, T in allowed.items():
        if k in body:
            try:
                if T is bool:
                    patch[k] = bool(body[k])
                elif T is list:
                    raw = body[k]
                    if isinstance(raw, list):
                        patch[k] = [float(x) for x in raw]
                else:
                    patch[k] = T(body[k])
            except (ValueError, TypeError):
                pass
    dec_proc.update_settings(patch)
    return jsonify({"status": "ok"})


@app.route("/decoder_process", methods=["POST"])
def decoder_process():
    """
    Accepts JSON: {frame: "data:image/...;base64,..."}
    Returns full decoder result as JSON.
    """
    body = request.get_json(force=True) or {}
    frame_b64 = body.get("frame", "")
    if not frame_b64:
        return jsonify({"error": "no frame"}), 400

    # Strip data URL prefix if present
    if "," in frame_b64:
        frame_b64 = frame_b64.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(frame_b64)
        arr       = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": f"decode error: {e}"}), 400

    if frame is None:
        return jsonify({"error": "could not decode image"}), 400

    result = dec_proc.process(frame)
    return jsonify(_to_python(result))


# ──────────────────────────────────────────────────────────────────── main
if __name__ == "__main__":
    print("=" * 60)
    print("  CPT Multi-Frequency Phase Encoder / Decoder")
    print("  Open http://localhost:5000 in your browser.")
    print("  Point your webcam at the encoder panel to decode.")
    print("=" * 60)
    # Use threaded=True so the encoder stream doesn't block decoder calls
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
