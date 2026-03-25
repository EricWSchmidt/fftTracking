import cv2
import cv2.aruco as aruco
import numpy as np
import io
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

# ── ArUco config ───────────────────────────────────────────────────────────────
ARUCO_DICT   = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
ARUCO_PARAMS = aruco.DetectorParameters()
# Faster detection: narrow threshold window range, skip subpixel refinement
ARUCO_PARAMS.adaptiveThreshWinSizeMin  = 3
ARUCO_PARAMS.adaptiveThreshWinSizeMax  = 23
ARUCO_PARAMS.adaptiveThreshWinSizeStep = 10
ARUCO_PARAMS.cornerRefinementMethod    = aruco.CORNER_REFINE_NONE
ARUCO_PARAMS.minMarkerPerimeterRate    = 0.05
DETECTOR     = aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

MARKER_ID   = 0      # which marker ID to generate and detect
MARKER_SIZE = 0.05   # physical side length in metres (edit to match your screen)

# half-size corners in marker-local frame (Z = 0 plane)
MARKER_OBJ_PTS = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
], dtype=np.float32)


def camera_matrix(w: int, h: int):
    """Approximate intrinsics (works well without calibration)."""
    f = float(max(w, h))
    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0, 1    ]], dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    return K, D


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/marker.png")
def marker_image():
    """Generate and return a DICT_6X6_250 marker (id=MARKER_ID) as PNG."""
    inner = 512
    border = inner // 8
    img = np.zeros((inner, inner), dtype=np.uint8)
    aruco.generateImageMarker(ARUCO_DICT, MARKER_ID, inner, img, 1)
    canvas = np.full((inner + border * 2, inner + border * 2), 255, dtype=np.uint8)
    canvas[border:border + inner, border:border + inner] = img
    _, buf = cv2.imencode(".png", canvas)
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


@app.route("/detect", methods=["POST"])
def detect():
    """
    Accept a raw JPEG frame (request body), run ArUco detection + solvePnP,
    return pose JSON.
    """
    data = request.get_data()
    if not data:
        return jsonify({"found": False, "error": "no data"})

    nparr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"found": False, "error": "decode failed"})

    h, w = frame.shape[:2]
    K, D = camera_matrix(w, h)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(gray)

    if ids is None or len(ids) == 0:
        return jsonify({"found": False})

    # Prefer MARKER_ID, fall back to first detected
    target_idx = 0
    for i, mid in enumerate(ids.flatten()):
        if mid == MARKER_ID:
            target_idx = i
            break

    c = corners[target_idx].reshape(4, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(MARKER_OBJ_PTS, c, K, D,
                                   flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return jsonify({"found": False})

    R, _ = cv2.Rodrigues(rvec)
    return jsonify({
        "found"    : True,
        "marker_id": int(ids.flatten()[target_idx]),
        "tvec"     : tvec.flatten().tolist(),   # [x, y, z] metres
        "rvec"     : rvec.flatten().tolist(),
        "R"        : R.flatten().tolist(),       # 3×3 row-major
        "corners"  : c.tolist(),                 # [[x,y]×4] image pixels
        "frame_w"  : w,
        "frame_h"  : h,
    })


if __name__ == "__main__":
    print("Open  http://localhost:5001  in your browser")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
