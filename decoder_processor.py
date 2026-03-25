"""
decoder_processor.py – Server-side 2-D FFT decoding pipeline.

Pipeline
--------
1. Receive a grayscale (or colour) webcam frame.
2. Crop / normalise.
3. Apply chosen 2-D window (Hann or Blackman).
4. Compute 2-D FFT; convert to centred magnitude/phase maps.
5. Find pilot peaks near their expected frequency positions.
6. Estimate and apply affine de-skew transform in frequency domain.
7. Read amplitude & phase at each data carrier bin.
8. Apply phase corrections (global offset, per-carrier pilot-derived offset).
9. Convert phases → bytes → ASCII string.
10. Generate annotated FFT panel (PNG bytes for the browser).
11. Produce guidance hints.
"""

from __future__ import annotations
import threading, base64, math
from typing import Optional
import numpy as np
import cv2
from numpy.fft import fft2, fftshift, ifft2
from scipy.ndimage import maximum_position
from encoder import (
    FRAME_SIZE, PILOT_FREQS, PILOT_PHASE, PILOT_AMPLITUDE,
    DATA_FREQS, DATA_AMPLITUDE, CARRIER_COLOURS, PILOT_COLOUR,
    FFT_ZOOM_BINS, ARROW_LENGTH, FONT,
    _byte_to_phase, _phase_to_byte, _val_to_phase, _phase_to_val, _make_window,
    ARUCO_DICT_ID, ARUCO_MARKER_ID, ARUCO_MARKER_PX, ARUCO_OFFSET_PX, ARUCO_REF_HALF,
    ARUCO_PHYSICAL_SIZE_M,
    _get_redundancy_groups,
)

# ──────────────────────────────────────────────────────────────────────────────
# Decoder defaults (all mutable via /update_settings)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "window_type":          "hann",       # "hann" | "blackman"
    "phase_offset":         0.0,          # global phase correction (radians)
    "geo_correction":       True,         # affine de-skew using pilots
    "pilot_search_r":       6,            # ±bins to search for each pilot
    "temporal_avg":         5,            # frames to time-average before decode
    "mag_threshold":        0.05,         # fraction of max mag to accept a peak
    "fft_zoom_bins":        FFT_ZOOM_BINS,
    "carrier_gain":         1.0,          # manual amplitude scale
    "phase_nudge":          [0.0] * len(DATA_FREQS),  # per-carrier phase trim
    "redundancy_mode":      "none",       # "none"|"pairs"|"quads"|"all8"
    "aruco_decode_enabled": False,        # run ArUco-referenced FFT decode path
    "mod_mode":             "psk",        # "psk" | "cpm"
    "cpm_h":                0.5,          # CPM modulation index (must match encoder)
    "bits_per_carrier":     8,            # 1-8: quantisation levels = 2^bits
}


def _encode_png(bgr: np.ndarray) -> str:
    """Encode a BGR uint8 image as a base64-encoded PNG data URL."""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _affine_correct_fft(
    F_shift: np.ndarray,
    src_pts: list[tuple[float, float]],
    dst_pts: list[tuple[float, float]],
) -> np.ndarray:
    """
    Apply an affine (or perspective) warp to the complex frequency-domain map.
    src_pts: measured full-FFT pixel positions of pilot peaks
    dst_pts: ideal full-FFT pixel positions
    Returns a warped copy of F_shift (same shape).
    """
    if len(src_pts) < 3:
        return F_shift
    src = np.array(src_pts[:4], dtype=np.float32)
    dst = np.array(dst_pts[:4], dtype=np.float32)

    H, W   = F_shift.shape
    mag    = np.abs(F_shift).astype(np.float32)
    phase  = np.angle(F_shift).astype(np.float32)

    if len(src_pts) >= 4:
        # Perspective (homography) – needs exactly 4 points
        M, mask = cv2.findHomography(src[:4], dst[:4], method=0)
        if M is None:
            return F_shift
        mag_w   = cv2.warpPerspective(mag,   M, (W, H),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
        phase_w = cv2.warpPerspective(phase, M, (W, H),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
    else:
        # Affine – needs 3 points (2×3 matrix → use warpAffine)
        M = cv2.getAffineTransform(src[:3], dst[:3])
        if M is None:
            return F_shift
        mag_w   = cv2.warpAffine(mag,   M, (W, H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
        phase_w = cv2.warpAffine(phase, M, (W, H),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)

    return mag_w * np.exp(1j * phase_w)


# ── ArUco detector: built once, shared across all frames ─────────────────────
# Re-creating the dictionary on every frame is expensive (~5-15 ms each).
# We also tune the parameters for speed: fewer adaptive-threshold window
# sizes, no subpixel corner refinement, and a tighter perimeter range
# (the marker occupies a large fraction of the frame during normal use).
def _build_aruco_detector():
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        params = cv2.aruco.DetectorParameters()
        # Adaptive threshold: scan only 3 window sizes instead of the
        # default ~11 (wins 3,7,11,...,53 step 4 → 3,13,23 step 10)
        params.adaptiveThreshWinSizeMin  = 3
        params.adaptiveThreshWinSizeMax  = 23
        params.adaptiveThreshWinSizeStep = 10
        # Accept only large markers (perimeter > 10 % of image perimeter)
        params.minMarkerPerimeterRate = 0.10
        params.maxMarkerPerimeterRate = 4.0
        # Skip subpixel refinement – saves ~2 ms and we don't need sub-pixel accuracy
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_NONE
        return cv2.aruco.ArucoDetector(dictionary, params), None
    except AttributeError:
        return None, "new-api-unavailable"
    except Exception as e:
        return None, str(e)

_ARUCO_DETECTOR, _ARUCO_DETECTOR_ERR = _build_aruco_detector()


class DecoderProcessor:
    """Stateful multi-frame decoder."""

    def __init__(self):
        self._lock    = threading.Lock()
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings["phase_nudge"] = [0.0] * len(DATA_FREQS)
        # CPM decoder state — accumulated byte and previous phase per carrier
        self._cpm_prev_phases = [None] * len(DATA_FREQS)
        self._cpm_byte_acc    = [0]    * len(DATA_FREQS)
        # Ring buffer for temporal averaging
        self._avg_buffer: list[np.ndarray] = []   # list of complex F_shift arrays
        # Ring buffer for multi-symbol message reconstruction
        self._symbol_history: list[list] = []     # list of decoded byte lists
        # Last results (returned if a new frame hasn't been processed yet)
        self._last_result: Optional[dict] = None
        # Max side-length for ArUco detection (downsample large frames)
        self._aruco_max_side = 640

    # ---------------------------------------------------------------- settings
    def update_settings(self, patch: dict):
        with self._lock:
            for k, v in patch.items():
                if k in self.settings:
                    self.settings[k] = v

    def get_settings(self) -> dict:
        with self._lock:
            return dict(self.settings)

    def detect_aruco_only(self, raw_frame: np.ndarray) -> dict:
        """Fast path: run only ArUco pose detection, skip all FFT work."""
        if raw_frame is None or raw_frame.size == 0:
            return {"detected": False, "error": "empty frame"}
        return self._detect_aruco_pose(raw_frame)

    # ----------------------------------------------------------------- process
    def process(self, raw_frame: np.ndarray) -> dict:
        """
        Process one webcam frame.  Returns a dict with:
          fft_image   – base64 PNG of annotated FFT panel
          raw_image   – base64 PNG of pre-processed input frame
          bytes_dec   – list of decoded byte values
          ascii_dec   – decoded ASCII string (non-printable → '?')
          correct     – True if surrounding 0-bytes detected
          phases_meas – list of measured phases (radians)
          phases_true – list of true encoder phases (radians) [from pilot offset]
          phase_errs  – phase errors in degrees
          pilot_info  – list of {expected, measured, displacement, phase_err}
          guidance    – list of hint strings
          snr_db      – estimated per-carrier SNR (dB)
        """
        with self._lock:
            s = dict(self.settings)   # snapshot settings

        # ── 1. Pre-process ──────────────────────────────────────────────────
        if raw_frame is None or raw_frame.size == 0:
            return {"error": "empty frame"}

        # ── 0. ArUco ground-truth pose (on full-res frame before any resize) ──
        aruco_pose = self._detect_aruco_pose(raw_frame)

        # ── 1. Pre-process ──────────────────────────────────────────────────
        # Resize to FRAME_SIZE × FRAME_SIZE
        frame = cv2.resize(raw_frame, (FRAME_SIZE, FRAME_SIZE),
                           interpolation=cv2.INTER_AREA)
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32)
        # Normalize to [0, 1]
        lo, hi = frame.min(), frame.max()
        if hi > lo:
            frame = (frame - lo) / (hi - lo)
        else:
            frame[:] = 0.5

        raw_u8  = (frame * 255).astype(np.uint8)
        raw_bgr = cv2.cvtColor(raw_u8, cv2.COLOR_GRAY2BGR)

        # ── 2. Window ───────────────────────────────────────────────────────
        win     = _make_window(FRAME_SIZE, s["window_type"])
        f_win   = frame * win

        # ── 3. 2-D FFT ──────────────────────────────────────────────────────
        N       = FRAME_SIZE
        F       = fft2(f_win)
        F_shift = fftshift(F)

        # ── 4. Temporal averaging ───────────────────────────────────────────
        with self._lock:
            self._avg_buffer.append(F_shift.copy())
            avg_n = max(1, s["temporal_avg"])
            if len(self._avg_buffer) > avg_n:
                self._avg_buffer = self._avg_buffer[-avg_n:]
            # Average in complex plane
            F_avg = np.mean(self._avg_buffer, axis=0)

        # ── 5. Find pilot peaks ─────────────────────────────────────────────
        pilot_info = []
        src_pts    = []   # measured pixel positions (full FFT coords)
        dst_pts    = []   # ideal pixel positions

        mag_avg  = np.abs(F_avg)
        c        = N // 2
        sr       = s["pilot_search_r"]
        global_phase_offset = 0.0
        per_pilot_phase_errs = []

        for (fy, fx) in PILOT_FREQS:
            # ideal position in full-size shifted FFT
            idy = c + fy
            idx = c + fx

            # clamp search window
            y0 = max(0, idy - sr);  y1 = min(N, idy + sr + 1)
            x0 = max(0, idx - sr);  x1 = min(N, idx + sr + 1)
            patch = mag_avg[y0:y1, x0:x1]

            if patch.size == 0:
                pilot_info.append({
                    "expected":      (fy, fx),
                    "measured_bin":  (fy, fx),
                    "displacement":  (0, 0),
                    "phase_measured": 0.0,
                    "phase_expected": PILOT_PHASE,
                    "phase_err_deg":  0.0,
                })
                src_pts.append((float(idx), float(idy)))
                dst_pts.append((float(idx), float(idy)))
                continue

            local_y, local_x = np.unravel_index(np.argmax(patch), patch.shape)
            my = y0 + local_y
            mx = x0 + local_x

            meas_fy = my - c
            meas_fx = mx - c
            disp_y  = meas_fy - fy
            disp_x  = meas_fx - fx

            meas_ph = float(np.angle(F_avg[my, mx]))
            ph_err  = _wrap_angle(meas_ph - PILOT_PHASE)
            per_pilot_phase_errs.append(ph_err)

            pilot_info.append({
                "expected":       (fy, fx),
                "measured_bin":   (meas_fy, meas_fx),
                "displacement":   (int(disp_y), int(disp_x)),
                "phase_measured": float(meas_ph),
                "phase_expected": float(PILOT_PHASE),
                "phase_err_deg":  float(np.degrees(ph_err)),
            })
            src_pts.append((float(mx), float(my)))
            dst_pts.append((float(idx), float(idy)))

        if per_pilot_phase_errs:
            # Global phase offset = mean pilot phase error
            global_phase_offset = float(np.angle(
                np.mean([np.exp(1j * e) for e in per_pilot_phase_errs])
            ))

        # ── 5.5. Camera pose estimation from pilot correspondences ────────────
        # Work in DC-centred bin coordinates: subtract c from full-FFT positions.
        pose = {"rotation_deg": 0.0, "zoom_scale": 1.0,
                "translation_x": 0.0, "translation_y": 0.0,
                "keystone_magnitude": 0.0, "pose_quality": "unknown"}
        if len(src_pts) >= 3:
            # src_pts / dst_pts are in full-FFT pixel coords (0…N-1);
            # convert to bin-offset coords relative to DC.
            exp_xy = np.array([(x - c, y - c) for (x, y) in dst_pts],
                              dtype=np.float64)   # ideal (fx, fy) per pilot
            meas_xy = np.array([(x - c, y - c) for (x, y) in src_pts],
                               dtype=np.float64)  # measured (fx, fy)
            pose = self._estimate_camera_pose(exp_xy, meas_xy)

        # ── 6. Geometric de-skew ─────────────────────────────────────────────
        if s["geo_correction"] and len(src_pts) >= 3:
            F_corrected = _affine_correct_fft(F_avg, src_pts, dst_pts)
        else:
            F_corrected = F_avg

        # ── 7. Read data carrier phases ──────────────────────────────────────
        total_phase_offset = global_phase_offset + s["phase_offset"]
        mag_corr   = np.abs(F_corrected)

        phases_meas = []
        snr_db_list = []
        nudges      = s["phase_nudge"]
        # Threshold: accept a bin if signal > threshold_factor * local noise floor
        # This avoids the DC-dominated global-max problem.
        snr_threshold = max(1.5, s["mag_threshold"] * 20)  # user 0–50% → SNR 1.5–10×

        for i, (fy, fx) in enumerate(DATA_FREQS):
            idy = c + fy
            idx = c + fx
            if 0 <= idy < N and 0 <= idx < N:
                val      = F_corrected[idy, idx]
                meas_mag = float(np.abs(val))
                meas_ph  = float(np.angle(val))
                # Noise estimate: mean magnitude in annulus around this bin
                bg_mag   = _estimate_noise(mag_corr, idy, idx, radius=4, width=3)
                snr      = meas_mag / (bg_mag + 1e-12)
                snr_db_list.append(float(20 * np.log10(snr + 1e-9)))
                if snr < snr_threshold:
                    phases_meas.append(None)
                else:
                    corrected = meas_ph - total_phase_offset - nudges[i]
                    phases_meas.append(float(corrected % (2 * np.pi)))
            else:
                phases_meas.append(None)
                snr_db_list.append(None)

        # ── 7.5 SNR-weighted redundancy group averaging ──────────────────────
        red_groups = _get_redundancy_groups(len(DATA_FREQS), s["redundancy_mode"])
        snr_linear = [
            10 ** (db / 20.0) if (db is not None) else 0.0
            for db in snr_db_list
        ]
        phases_decode = []   # one phase per group (averaged)
        snr_decode    = []   # one SNR per group (dB, log-weighted)

        for group in red_groups:
            z_sum   = 0.0 + 0.0j
            w_total = 0.0
            for ci in group:
                ph = phases_meas[ci] if ci < len(phases_meas) else None
                w  = snr_linear[ci]  if ci < len(snr_linear)  else 0.0
                if ph is not None and w > 0:
                    z_sum  += w * np.exp(1j * ph)
                    w_total += w
            if w_total > 0:
                avg_ph  = float(np.angle(z_sum)) % (2 * np.pi)
                # group SNR ≈ geom-mean of member SNRs (dB)
                avg_snr = float(20 * np.log10(w_total / len(group) + 1e-9))
            else:
                avg_ph  = None
                avg_snr = -30.0
            phases_decode.append(avg_ph)
            snr_decode.append(avg_snr)

        # ── 8. Decode bytes ──────────────────────────────────────────────────
        mod_mode = s.get("mod_mode", "psk")
        cpm_h    = float(s.get("cpm_h", 0.5))
        bits     = int(s.get("bits_per_carrier", 8))
        bytes_dec = []
        for gi, ph in enumerate(phases_decode):
            if ph is None:
                bytes_dec.append(None)
                # Reset CPM state for this redundancy group on dropout
                if mod_mode == "cpm" and gi < len(red_groups):
                    for ci in red_groups[gi]:
                        self._cpm_prev_phases[ci] = None
            elif mod_mode == "cpm":
                ci      = red_groups[gi][0] if gi < len(red_groups) else gi
                prev_ph = self._cpm_prev_phases[ci]
                if prev_ph is None:
                    b = 0  # first frame after dropout
                else:
                    dphi  = (ph - prev_ph + np.pi) % (2 * np.pi) - np.pi
                    delta = int(round(dphi / (np.pi * cpm_h) * 255))
                    b     = int((self._cpm_byte_acc[ci] + delta) % 256)
                    if gi < len(red_groups):
                        for ci2 in red_groups[gi]:
                            self._cpm_byte_acc[ci2] = b
                if gi < len(red_groups):
                    for ci2 in red_groups[gi]:
                        self._cpm_prev_phases[ci2] = ph
                bytes_dec.append(b)
            else:
                # PSK: level-aware decode using bits_per_carrier
                bytes_dec.append(_phase_to_val(ph, bits))

        ascii_dec = ""
        maxval = (1 << bits) - 1
        for b in bytes_dec:
            if b is None:
                ascii_dec += "·"
            elif bits == 8 and 32 <= b <= 126:
                ascii_dec += chr(b)
            elif bits == 8 and b == 0:
                ascii_dec += "∅"
            elif bits < 8:
                ascii_dec += str(b) + " "
            else:
                ascii_dec += f"[{b:02X}]"

        # ── 8b. Multi-symbol ring buffer for full-message detection ──────────
        with self._lock:
            self._symbol_history.append(list(bytes_dec))
            if len(self._symbol_history) > 32:
                self._symbol_history = self._symbol_history[-32:]
            # Flatten every combination of consecutive symbols and scan for
            # the 0x00 … 0x00 sentinel frame.
            correct, full_message = self._scan_for_message()

        # Check instant symbol for sentinel presence (convenience)
        nn = [b for b in bytes_dec if b is not None]
        instant_correct = len(nn) >= 2 and nn[0] == 0 and nn[-1] == 0

        # ── 9. Phase errors vs. ideal ────────────────────────────────────────
        phase_errs = []
        for ph in phases_decode:
            if ph is None:
                phase_errs.append(None)
            else:
                phase_errs.append(None)   # can't know true without encoder state

        # Expand grouped results back to per-carrier arrays for FFT panel display.
        # In "none" mode this is a no-op.
        bytes_dec_8  = [None] * len(DATA_FREQS)
        phases_dec_8 = [None] * len(DATA_FREQS)
        for g_idx, group in enumerate(red_groups):
            b  = bytes_dec[g_idx]  if g_idx < len(bytes_dec)    else None
            ph = phases_decode[g_idx] if g_idx < len(phases_decode) else None
            for ci in group:
                bytes_dec_8[ci]  = b
                phases_dec_8[ci] = ph

        # ── 10. Annotated FFT panel ──────────────────────────────────────────
        fft_panel = self._make_fft_panel(
            F_corrected, phases_dec_8, bytes_dec_8, pilot_info,
            s["fft_zoom_bins"], s["window_type"],
        )

        # ── 11. Guidance ─────────────────────────────────────────────────────
        guidance = self._generate_guidance(
            pilot_info, snr_db_list, bytes_dec_8, phases_dec_8,
            s["phase_offset"], global_phase_offset, pose,
        )

        fft_png = _encode_png(fft_panel)
        raw_png = _encode_png(raw_bgr)

        # ── 11.5 ArUco-referenced parallel FFT decode ─────────────────────────
        aruco_fft_result = None
        if s.get("aruco_decode_enabled", False) and aruco_pose.get("detected"):
            aruco_fft_result = self._aruco_fft_decode(raw_frame, aruco_pose, s)

        result = {
            "fft_image":      fft_png,
            "raw_image":      raw_png,
            "bytes_dec":      [b if b is not None else -1 for b in bytes_dec],
            "ascii_dec":      ascii_dec,
            "correct":        correct,            # True = full sentinel-bounded msg found
            "full_message":   full_message,       # extracted ASCII between sentinels
            "instant_correct": instant_correct,   # True = THIS symbol starts/ends with 0
            "phases_meas":    [float(p) if p is not None else None
                               for p in phases_dec_8],
            "phases_raw":     [float(p) if p is not None else None
                               for p in phases_meas],
            "phase_errs":     phase_errs,
            "pilot_info":     pilot_info,
            "guidance":       guidance,
            "snr_db":         snr_db_list,
            "snr_db_grouped": snr_decode,
            "redundancy_mode":   s["redundancy_mode"],
            "redundancy_groups": [list(g) for g in red_groups],
            "global_phase_offset_deg": float(np.degrees(global_phase_offset)),
            "window_type":    s["window_type"],
            "mod_mode":       s.get("mod_mode", "psk"),
            "bits_per_carrier": s.get("bits_per_carrier", 8),
            "pose":           pose,
            "aruco_pose":     aruco_pose,
            "aruco_fft_result": aruco_fft_result,
        }
        with self._lock:
            self._last_result = result
        return result

    # ─────────────────────────────────────────── ArUco-referenced FFT decode
    def _aruco_fft_decode(self, raw_frame: np.ndarray, aruco_pose: dict, s: dict) -> dict:
        """
        Use the detected ArUco marker to warp the signal panel out of the
        webcam frame, run FFT, and decode the data carriers.

        The composite encoder frame is [signal 256×256 | ArUco 256×256 | FFT 256×256].
        We find the signal panel by projecting its corners through the homography
        derived from the ArUco marker's known and detected positions.
        """
        try:
            meas_c = np.array(aruco_pose["meas_corners_px"], dtype=np.float32)

            # Marker corners in ArUco-panel pixel coords
            o = ARUCO_OFFSET_PX
            m = ARUCO_MARKER_PX
            panel_marker_pts = np.float32([
                [o,     o    ],   # TL
                [o + m, o    ],   # TR
                [o + m, o + m],   # BR
                [o,     o + m],   # BL
            ])

            # H: ArUco-panel-space → webcam-pixel-space
            H_to_webcam, _ = cv2.findHomography(panel_marker_pts, meas_c)
            if H_to_webcam is None:
                return {"error": "homography failed"}

            # Signal panel corners in ArUco-panel-space.
            # The signal panel is one panel-width (256 px) to the LEFT of the
            # ArUco panel, so subtract 256 from the x coordinate.
            sig_in_aruco = np.float32([
                [-256,  0  ],   # signal TL  (= aruco_panel TL - 256 in X)
                [  0,   0  ],   # signal TR  (= aruco_panel TL)
                [  0,  255 ],   # signal BR  (= aruco_panel BL)
                [-256, 255 ],   # signal BL
            ]).reshape(1, -1, 2)

            sig_webcam = cv2.perspectiveTransform(sig_in_aruco, H_to_webcam).reshape(4, 2)

            # Warp signal region to standard 256×256
            dst_pts = np.float32([[0,0],[255,0],[255,255],[0,255]])
            H_warp  = cv2.getPerspectiveTransform(sig_webcam, dst_pts)

            if raw_frame.ndim == 3:
                warp_gray = cv2.warpPerspective(
                    cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY), H_warp, (256, 256))
            else:
                warp_gray = cv2.warpPerspective(raw_frame, H_warp, (256, 256))

            # Normalize → window → FFT
            warp_f = warp_gray.astype(np.float32)
            lo, hi = warp_f.min(), warp_f.max()
            if hi > lo:
                warp_f = (warp_f - lo) / (hi - lo)
            else:
                warp_f[:] = 0.5

            win  = _make_window(FRAME_SIZE, s["window_type"])
            F_w  = fftshift(fft2(warp_f * win))
            N    = FRAME_SIZE
            c    = N // 2
            mag  = np.abs(F_w)

            # Decode pilots for phase offset
            gpo  = 0.0
            for (fy, fx) in PILOT_FREQS:
                idy = c + fy;  idx = c + fx
                if 0 <= idy < N and 0 <= idx < N:
                    ph = float(np.angle(F_w[idy, idx]))
                    gpo += ph
            if PILOT_FREQS:
                gpo /= len(PILOT_FREQS)

            # Read data carriers
            a_phases, a_snrs, a_bytes = [], [], []
            snr_thr = max(1.5, s["mag_threshold"] * 20)
            for (fy, fx) in DATA_FREQS:
                idy = c + fy;  idx = c + fx
                if 0 <= idy < N and 0 <= idx < N:
                    val = F_w[idy, idx]
                    m_v = float(np.abs(val))
                    ph  = float(np.angle(val))
                    bg  = _estimate_noise(mag, idy, idx, radius=4, width=3)
                    snr = m_v / (bg + 1e-12)
                    snr_db = float(20 * np.log10(snr + 1e-9))
                    a_snrs.append(snr_db)
                    if snr >= snr_thr:
                        corrected = (ph - gpo) % (2 * np.pi)
                        a_phases.append(float(corrected))
                        a_bytes.append(_phase_to_byte(corrected))
                    else:
                        a_phases.append(None)
                        a_bytes.append(None)
                else:
                    a_phases.append(None)
                    a_snrs.append(None)
                    a_bytes.append(None)

            # Build ASCII
            ascii_out = ""
            for b in a_bytes:
                if b is None:         ascii_out += "·"
                elif b == 0:          ascii_out += "∅"
                elif 32 <= b <= 126:  ascii_out += chr(b)
                else:                 ascii_out += f"[{b:02X}]"

            # Annotated FFT panel for ArUco-referenced path
            fft_panel_a = self._make_fft_panel(
                F_w, a_phases, a_bytes, [], s["fft_zoom_bins"], s["window_type"])
            fft_png_a = _encode_png(fft_panel_a)

            # Warped spatial image for display
            warp_bgr = cv2.cvtColor(warp_gray, cv2.COLOR_GRAY2BGR)
            warp_png = _encode_png(warp_bgr)

            return {
                "bytes_dec": [b if b is not None else -1 for b in a_bytes],
                "ascii_dec": ascii_out,
                "phases_meas": [float(p) if p is not None else None for p in a_phases],
                "snr_db":    a_snrs,
                "fft_image": fft_png_a,
                "warp_image": warp_png,
            }
        except Exception as e:
            return {"error": str(e)}

    # ─────────────────────────────────────────── ArUco ground-truth pose
    def _detect_aruco_pose(self, raw_frame: np.ndarray) -> dict:
        """
        Detect the large ArUco marker panel in raw_frame and estimate camera pose.

        The ArUco marker sits in a 256×256 panel; its four outer corners in that
        panel’s DC-centred coordinates are (±ARUCO_REF_HALF, ±ARUCO_REF_HALF).
        The measured corners from the webcam frame are also normalised to those
        units (scale: cam_half_px → ARUCO_REF_HALF), making zoom_scale and
        rotation_deg directly comparable to the pilot FFT pose.
        """
        H_f, W_f = raw_frame.shape[:2]

        # -- Downscale for fast detection -----------------------------------
        max_side  = self._aruco_max_side
        long_side = max(H_f, W_f)
        if long_side > max_side:
            scale_d   = max_side / long_side
            det_frame = cv2.resize(raw_frame,
                                   (int(W_f * scale_d), int(H_f * scale_d)),
                                   interpolation=cv2.INTER_AREA)
        else:
            scale_d   = 1.0
            det_frame = raw_frame

        # -- Detect using cached detector -----------------------------------
        global _ARUCO_DETECTOR
        try:
            if _ARUCO_DETECTOR is not None:
                corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(det_frame)
            else:
                raise AttributeError("new API unavailable")
        except AttributeError:
            # Fallback to old OpenCV 4.x API
            try:
                dictionary  = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
                parameters  = cv2.aruco.DetectorParameters_create()
                corners, ids, _ = cv2.aruco.detectMarkers(
                    det_frame, dictionary, parameters=parameters)
            except Exception as e:
                return {"detected": False, "error": str(e)}
        except Exception as e:
            return {"detected": False, "error": str(e)}

        # -- Scale corners back to original resolution ----------------------
        if scale_d != 1.0 and corners:
            corners = [c / scale_d for c in corners]

        if ids is None or ARUCO_MARKER_ID not in ids.flatten():
            return {"detected": False}

        # Pick the correct marker
        idx = int(np.where(ids.flatten() == ARUCO_MARKER_ID)[0][0])
        meas_corners = corners[idx][0].astype(np.float64)  # (4,2) TL TR BR BL

        # Expected corners of the 210×210 marker image inside the 256×256 panel,
        # in DC-centred "panel-pixel" coords, normalised to ARUCO_REF_HALF units.
        # Marker outer edge sits at ±(ARUCO_MARKER_PX/2) from panel centre.
        H_ref   = ARUCO_MARKER_PX / 2.0   # = 105 px
        scale_f = ARUCO_REF_HALF / H_ref  # stretch to ±128 reference units
        exp_corners = np.array([
            [-1, -1],   # TL
            [ 1, -1],   # TR
            [ 1,  1],   # BR
            [-1,  1],   # BL
        ], dtype=np.float64) * H_ref * scale_f  # = ±128

        # Webcam corners: DC-centred, normalised so half-webcam = ARUCO_REF_HALF
        cam_half   = min(H_f, W_f) / 2.0
        cam_scale  = ARUCO_REF_HALF / cam_half
        meas_norm  = (meas_corners - np.array([W_f / 2.0, H_f / 2.0])) * cam_scale

        pose = DecoderProcessor._estimate_camera_pose(exp_corners, meas_norm)

        # Better keystone from full homography residual
        src32 = exp_corners.astype(np.float32)
        dst32 = meas_norm.astype(np.float32)
        H_hom, _ = cv2.findHomography(src32, dst32, method=0)
        if H_hom is not None:
            proj = cv2.perspectiveTransform(
                src32.reshape(1, -1, 2), H_hom).reshape(-1, 2)
            kst  = float(np.max(np.sqrt(np.sum((proj - dst32) ** 2, axis=1))))
            pose["keystone_magnitude"] = round(kst, 3)

        meas_c = meas_corners.mean(axis=0)
        pose["detected"]      = True
        pose["marker_id"]     = ARUCO_MARKER_ID
        pose["center_px"]     = [round(float(meas_c[0]), 1),
                                  round(float(meas_c[1]), 1)]
        pose["apparent_size_px"] = round(float(
            np.mean([
                np.linalg.norm(meas_corners[1] - meas_corners[0]),
                np.linalg.norm(meas_corners[2] - meas_corners[1]),
            ])
        ), 1)

        # ── True 3-D position via solvePnP ────────────────────────────────────
        try:
            half_m = ARUCO_PHYSICAL_SIZE_M / 2.0
            obj_pts = np.float32([
                [-half_m, -half_m, 0],   # TL
                [ half_m, -half_m, 0],   # TR
                [ half_m,  half_m, 0],   # BR
                [-half_m,  half_m, 0],   # BL
            ])
            H_f, W_f2 = raw_frame.shape[:2]
            focal = float(max(H_f, W_f2))
            cam_mat = np.float32([
                [focal, 0,     W_f2 / 2.0],
                [0,     focal, H_f  / 2.0],
                [0,     0,     1.0        ],
            ])
            dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            ret, rvec, tvec = cv2.solvePnP(
                obj_pts, meas_corners.astype(np.float32),
                cam_mat, dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            if ret:
                t = tvec.flatten()
                R, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
                if sy > 1e-6:
                    rx = math.degrees(math.atan2( R[2, 1], R[2, 2]))
                    ry = math.degrees(math.atan2(-R[2, 0], sy))
                    rz = math.degrees(math.atan2( R[1, 0], R[0, 0]))
                else:
                    rx = math.degrees(math.atan2(-R[1, 2], R[1, 1]))
                    ry = math.degrees(math.atan2(-R[2, 0], sy))
                    rz = 0.0
                pose["pos3d_x"]  = round(float(t[0]), 4)
                pose["pos3d_y"]  = round(float(t[1]), 4)
                pose["pos3d_z"]  = round(float(t[2]), 4)
                pose["euler_rx"] = round(rx, 2)
                pose["euler_ry"] = round(ry, 2)
                pose["euler_rz"] = round(rz, 2)
        except Exception:
            pass   # solvePnP optional — don't abort pose result

        # also store raw corners for the ArUco-referenced FFT decode path
        pose["meas_corners_px"] = meas_corners.tolist()

        return pose

    # ─────────────────────────────────────────── camera pose estimation
    @staticmethod
    def _estimate_camera_pose(
        exp_xy:  np.ndarray,   # (N,2) ideal pilot positions in bin-offset coords
        meas_xy: np.ndarray,   # (N,2) measured pilot positions in bin-offset coords
    ) -> dict:
        """
        Fit a similarity transform (rotation + uniform scale + translation)
        from expected → measured pilot positions via least squares.

        Model per pilot:
            [x  -y  1  0] [a b tx ty]ᵀ = [x']
            [y   x  0  1]               = [y']

        where  a = s·cos(θ),  b = s·sin(θ)
        → θ = arctan2(b, a),  s = √(a²+b²)
        """
        n = len(exp_xy)
        A = np.zeros((2 * n, 4))
        b_vec = np.zeros(2 * n)
        for i, ((x, y), (xp, yp)) in enumerate(zip(exp_xy, meas_xy)):
            A[2*i]     = [ x, -y, 1, 0]
            A[2*i + 1] = [ y,  x, 0, 1]
            b_vec[2*i]     = xp
            b_vec[2*i + 1] = yp

        params, residuals_arr, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
        a_p, b_p, tx, ty = params

        theta_deg = float(np.degrees(np.arctan2(b_p, a_p)))
        scale     = float(np.sqrt(a_p**2 + b_p**2))

        # Keystone = max absolute residual after similarity fit;
        # non-zero only when perspective distortion is present.
        reconstructed      = A @ params
        per_point_residuals = np.sqrt(
            (b_vec[0::2] - reconstructed[0::2])**2 +
            (b_vec[1::2] - reconstructed[1::2])**2
        )
        keystone_mag = float(per_point_residuals.max()) if len(per_point_residuals) else 0.0

        if keystone_mag < 1.0:
            quality = "good"
        elif keystone_mag < 3.0:
            quality = "moderate"
        else:
            quality = "poor"

        return {
            "rotation_deg":       round(theta_deg, 2),
            "zoom_scale":         round(scale, 4),
            "translation_x":      round(float(tx), 2),
            "translation_y":      round(float(ty), 2),
            "keystone_magnitude": round(keystone_mag, 3),
            "pose_quality":       quality,
            # Per-pilot residuals for detailed UI display
            "pilot_residuals":    [round(float(r), 3) for r in per_point_residuals],
        }

    # ─────────────────────────────────────────── multi-symbol message scan
    def _scan_for_message(self) -> tuple[bool, str]:
        """
        Flatten symbol history, find sentinel 0x00 bytes (with ±8 byte tolerance),
        and extract the ASCII content between them.
        Called with self._lock held.
        """
        SENTINEL_TOL = 8   # bytes within [0,8] or [247,255] are treated as 0x00

        def is_sentinel(b):
            return b is not None and (b <= SENTINEL_TOL or b >= (255 - SENTINEL_TOL))

        flat = []
        for sym in self._symbol_history:
            flat.extend(sym)   # may include None

        # Remove None values for scanning
        clean = [b for b in flat if b is not None]
        if len(clean) < 2:
            return False, ""

        zeros = [i for i, b in enumerate(clean) if is_sentinel(b)]
        if len(zeros) < 2:
            return False, ""

        for start_i in zeros:
            for end_i in zeros:
                if end_i <= start_i + 1:
                    continue
                chunk = clean[start_i + 1 : end_i]
                if len(chunk) == 0:
                    continue
                # Chunk must contain at least one printable byte
                if not any(32 <= b <= 126 for b in chunk):
                    continue
                text = ""
                for b in chunk:
                    if is_sentinel(b):
                        continue   # skip embedded padding nulls
                    if 32 <= b <= 126:
                        text += chr(b)
                    else:
                        text += f"\\x{b:02x}"
                if text:
                    return True, text
        return False, ""

    # ─────────────────────────────────────────── annotated FFT helper
    def _make_fft_panel(
        self,
        F_shift:     np.ndarray,
        phases_meas: list,
        bytes_dec:   list,
        pilot_info:  list,
        zoom_bins:   int,
        win_type:    str,
    ) -> np.ndarray:
        N    = FRAME_SIZE
        disp = 512
        half = disp // 2

        mag  = np.abs(F_shift)
        mag_log = np.log1p(mag)

        c    = N // 2
        zoom = max(4, zoom_bins)
        crop = mag_log[c - zoom : c + zoom, c - zoom : c + zoom]
        if crop.max() > 0:
            crop = crop / crop.max() * 255.0
        crop_u8  = crop.astype(np.uint8)
        crop_big = cv2.resize(crop_u8, (disp, disp),
                              interpolation=cv2.INTER_NEAREST)
        panel    = cv2.applyColorMap(crop_big, cv2.COLORMAP_INFERNO)

        scale = disp / (2.0 * zoom)   # px / bin

        def freq_to_px(fy, fx):
            px = int(half + fx * scale)
            py = int(half + fy * scale)
            return px, py

        def draw_arrow(img, cx, cy, phase_rad, length, colour, thickness=2,
                       linetype=cv2.LINE_AA):
            ax = cx + int(length * np.cos(phase_rad))
            ay = cy - int(length * np.sin(phase_rad))
            cv2.arrowedLine(img, (cx, cy), (ax, ay), colour, thickness,
                            tipLength=0.4, line_type=linetype)

        # ── draw pilots ──
        for pinfo in pilot_info:
            efy, efx = pinfo["expected"]
            mfy, mfx = pinfo["measured_bin"]
            # ideal dot (green empty circle)
            epx, epy = freq_to_px(efy, efx)
            mpx, mpy = freq_to_px(mfy, mfx)
            cv2.circle(panel, (epx, epy), 10, PILOT_COLOUR, 1)        # ideal
            cv2.circle(panel, (mpx, mpy), 10, PILOT_COLOUR, 2)         # measured
            # arrow for measured phase
            draw_arrow(panel, mpx, mpy, pinfo["phase_measured"],
                       ARROW_LENGTH + 2, PILOT_COLOUR)
            # arrow for expected phase (dashed look: draw short line)
            ex_ax = epx + int(ARROW_LENGTH * np.cos(PILOT_PHASE))
            ex_ay = epy - int(ARROW_LENGTH * np.sin(PILOT_PHASE))
            cv2.line(panel, (epx, epy), (ex_ax, ex_ay), (0, 180, 0), 1)
            # displacement line
            if (epx, epy) != (mpx, mpy):
                cv2.line(panel, (epx, epy), (mpx, mpy), (0, 200, 100), 1)
            # phase error label
            label = f"Δφ={pinfo['phase_err_deg']:.1f}°"
            cv2.putText(panel, label, (mpx + 12, mpy + 4),
                        FONT, 0.30, PILOT_COLOUR, 1, cv2.LINE_AA)

        # ── draw data carriers ──
        for i, (fy, fx) in enumerate(DATA_FREQS):
            px, py  = freq_to_px(fy, fx)
            colour  = CARRIER_COLOURS[i]
            if not (5 < px < disp-5 and 5 < py < disp-5):
                continue
            cv2.circle(panel, (px, py), 9, colour, 2)
            ph = phases_meas[i]
            bv = bytes_dec[i]
            bv_str = f"{bv:3d}" if bv is not None else "???"
            if ph is not None:
                draw_arrow(panel, px, py, ph, ARROW_LENGTH, colour, 2)
            lbl = f"B{i}:{bv_str}"
            cv2.putText(panel, lbl, (px + 11, py + 4),
                        FONT, 0.28, colour, 1, cv2.LINE_AA)

        # ── labels ──
        cv2.putText(panel, f"WIN:{win_type.upper()}  ZOOM:±{zoom}bins",
                    (4, 14), FONT, 0.32, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, "SOLID ARROW=meas  DASHED=expected  LINE=displacement",
                    (4, 26), FONT, 0.28, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(panel, "GREEN circles=pilots  COLOURED=data",
                    (4, 38), FONT, 0.28, (200, 200, 200), 1, cv2.LINE_AA)

        return panel

    # ─────────────────────────────────────────── guidance engine
    def _generate_guidance(
        self,
        pilot_info:   list,
        snr_db_list:  list[float],
        bytes_dec:    list,
        phases_meas:  list,
        user_offset:  float,
        auto_offset:  float,
        pose:         dict | None = None,
    ) -> list[str]:
        hints = []
        pose = pose or {}

        # ── Pose-based hints (rotation, zoom, translation, keystone) ──
        rot   = pose.get("rotation_deg", 0.0)
        scale = pose.get("zoom_scale", 1.0)
        tx    = pose.get("translation_x", 0.0)
        ty    = pose.get("translation_y", 0.0)
        kst   = pose.get("keystone_magnitude", 0.0)
        qual  = pose.get("pose_quality", "unknown")

        if abs(rot) > 2.0:
            direction = "counter-clockwise" if rot > 0 else "clockwise"
            hints.append(
                f"Camera/screen is rotated {abs(rot):.1f}° off-axis. "
                f"Rotate camera {direction} to straighten the signal."
            )

        if scale < 0.85:
            pct = round((1.0 - scale) * 100)
            hints.append(
                f"Zoom scale is {scale:.3f}× — signal is {pct}% smaller than expected. "
                "Move the camera closer or zoom in."
            )
        elif scale > 1.18:
            pct = round((scale - 1.0) * 100)
            hints.append(
                f"Zoom scale is {scale:.3f}× — signal is {pct}% larger than expected. "
                "Move the camera further away or zoom out."
            )

        if abs(tx) > 2.0 or abs(ty) > 2.0:
            hints.append(
                f"Signal is off-centre by ({tx:+.1f}, {ty:+.1f}) bins. "
                "Re-centre the camera on the signal."
            )

        if kst > 3.0:
            hints.append(
                f"Keystone distortion residual is {kst:.2f} bins (quality: {qual}). "
                "Square up your camera angle — look straight at the screen, not from an angle."
            )
        elif kst > 1.0:
            hints.append(
                f"Mild keystone detected ({kst:.2f} bins). "
                "Enable Geometric Correction to compensate."
            )

        # Phase-offset hint
        auto_deg = float(np.degrees(auto_offset))
        if abs(auto_deg) > 5:
            hints.append(
                f"Global pilot phase error is {auto_deg:.1f}°. "
                f"Set Phase Offset slider to ~{np.degrees(user_offset) - auto_deg:.1f}° "
                "to compensate."
            )

        # Rotation hint from per-pilot phase spread (catches non-uniform distortion)
        if len(pilot_info) >= 2:
            ph_errs = [p["phase_err_deg"] for p in pilot_info]
            skew    = max(ph_errs) - min(ph_errs)
            if skew > 20:
                hints.append(
                    f"Pilot phase spread is {skew:.1f}°. "
                    "Enable Geometric Correction if not already on."
                )

        # SNR hints
        if snr_db_list:
            mean_snr = float(np.mean(snr_db_list))
            low_carriers = [i for i, s in enumerate(snr_db_list) if s < 6]
            if mean_snr < 6:
                hints.append(
                    f"Average carrier SNR is {mean_snr:.1f} dB (very low). "
                    "Move the camera closer to the signal, reduce ambient light, "
                    "or increase Temporal Averaging."
                )
            elif low_carriers:
                hints.append(
                    f"Carriers {low_carriers} have SNR < 6 dB. "
                    "Those bytes will be unreliable. Try Blackman window to reduce leakage."
                )

        # Codec / compression hint
        none_count = sum(1 for p in phases_meas if p is None)
        if none_count > 0:
            hints.append(
                f"{none_count} carrier(s) are below magnitude threshold "
                "(shown as '???' bytes). Lower Magnitude Threshold slider or move closer."
            )

        # Sentinel check — single-symbol level
        nn = [b for b in bytes_dec if b is not None]
        # Note: the full message spans multiple symbols; 0x00 sentinels are only
        # at the very first and very last bytes of the whole encoded payload.
        # Here we just check for carrier-0 being 0x00 (likely symbol 0).
        if nn and 0 not in nn:
            hints.append(
                "No 0x00 byte in this symbol. This is expected mid-message; "
                "wait for the symbol that starts/ends with 0x00."
            )

        # Window suggestion
        if snr_db_list and np.mean(snr_db_list) < 10:
            hints.append(
                "Blackman window gives better sidelobe suppression (≈−57 dB vs −31 dB for Hann) "
                "at the cost of a wider main lobe — try switching."
            )

        if not hints:
            hints.append("Signal looks good — no obvious corrections needed.")

        return hints


# ─────────────────────────────────────────────────────────────────── helpers
def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _estimate_noise(mag: np.ndarray, cy: int, cx: int,
                    radius: int = 3, width: int = 2) -> float:
    """
    Estimate background noise magnitude as the mean in an annulus
    centred on (cy, cx), from radius to radius+width bins away.
    """
    N  = mag.shape[0]
    ys = np.arange(max(0, cy - radius - width), min(N, cy + radius + width + 1))
    xs = np.arange(max(0, cx - radius - width), min(N, cx + radius + width + 1))
    yy, xx = np.meshgrid(ys, xs, indexing='ij')
    dist    = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    mask    = (dist >= radius) & (dist <= radius + width)
    if mask.sum() == 0:
        return 0.0
    return float(mag[ys[:, None], xs[None, :]][mask].mean())
