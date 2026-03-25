"""
encoder.py – Multi-frequency Continuous Phase Tracking encoder.

Architecture
------------
* The message string is wrapped in 0x00 sentinel bytes.
* The bytes are chunked across N_CARRIERS independent frequency bins.
* Each (fy, fx) carrier encodes a single byte as a phase:
      φ = byte_value / 255.0 * 2π
* Four pilot carriers at known (fy, fx) positions carry a fixed known
  phase so the decoder can correct geometric distortion and phase offset.
* The spatial-domain frame is generated via 2D IFFT of the frequency map.
* A 2D cosine-bell (Hann or Blackman) window is *multiplied* in the
  spatial domain to suppress spectral leakage BEFORE FFT analysis.
* Symbols are held for SYMBOL_FRAMES frames; the last BLEND_FRAC fraction
  of each symbol smoothly transitions to the next via complex-plane lerp.
* The returned encode frame is a side-by-side (signal | annotated FFT) image.
"""

from __future__ import annotations
import threading
import numpy as np
import cv2
from numpy.fft import ifft2, fft2, fftshift

# ──────────────────────────────────────────────────────────────────────────────
# Layout constants (all in units of FFT bin indices relative to the DC centre)
# ──────────────────────────────────────────────────────────────────────────────
FRAME_SIZE    = 256          # square frame (pixels)
SYMBOL_FRAMES = 90           # frames per symbol (3 s @ 30 fps)
BLEND_FRAC    = 0.20         # fraction of symbol used for phase transition
FFT_ZOOM_BINS = 36           # ±bins shown in the zoomed FFT panel
ARROW_LENGTH  = 11           # pixels for phase arrow in FFT display

# Pilot carrier positions (fy, fx) — NOT used as data carriers.
# These form symmetric corners in frequency space for geometric de-skew.
# Placed at ±9 bins so they're >9 bins from DC (search_radius=6 won't hit DC).
# Note: (9,9) and (-9,-9) are Hermitian conjugates; since pilot phase=0,
# the amplitude doubles at both positions but phase stays 0 — harmless.
# These are MUTABLE lists so the server can update them at runtime.
PILOT_FREQS = [
    [ 9,  9],
    [ 9, -9],
    [-9,  9],
    [-9, -9],
]
PILOT_PHASE     = 0.0   # known reference phase for all pilots
PILOT_AMPLITUDE = 1.5

# Data carrier positions — all in the "positive" half-plane so no carrier
# overlaps with another's Hermitian conjugate.  The encoder automatically
# writes the conjugate at (-fy,-fx) to keep the spatial signal real-valued.
# Rule: for every (fy, fx) here, (-fy, -fx) must NOT also appear here.
# These are MUTABLE lists so the server can update them at runtime.
DATA_FREQS = [
    [ 14,   0],   # conjugate: (-14,  0) – not in list ✓
    [  0,  14],   # conjugate: (  0,-14) – not in list ✓
    [ 14,  14],   # conjugate: (-14,-14) – not in list ✓
    [ 14, -14],   # conjugate: (-14, 14) – not in list ✓
    [ 20,   0],   # conjugate: (-20,  0) – not in list ✓
    [  0,  20],   # conjugate: (  0,-20) – not in list ✓
    [ 20,  10],   # conjugate: (-20,-10) – not in list ✓
    [ 10,  20],   # conjugate: (-10,-20) – not in list ✓
]

DATA_AMPLITUDE = 1.0   # amplitude of each data carrier

_FREQS_LOCK = threading.Lock()

def update_freq_bins(data_freqs=None, pilot_freqs=None):
    """Mutate DATA_FREQS / PILOT_FREQS in-place so all consumers see the change."""
    with _FREQS_LOCK:
        if data_freqs is not None and len(data_freqs) == len(DATA_FREQS):
            for i, (ky, kx) in enumerate(data_freqs):
                DATA_FREQS[i][0] = int(ky)
                DATA_FREQS[i][1] = int(kx)
        if pilot_freqs is not None and len(pilot_freqs) == len(PILOT_FREQS):
            for i, (ky, kx) in enumerate(pilot_freqs):
                PILOT_FREQS[i][0] = int(ky)
                PILOT_FREQS[i][1] = int(kx)

# Colour palette for the 8 data carriers (BGR)
CARRIER_COLOURS = [
    (255,  80,  80),   # blue-ish
    ( 80, 255,  80),   # green
    ( 80,  80, 255),   # red-ish
    (255, 255,  80),   # cyan
    (255,  80, 255),   # magenta
    ( 80, 255, 255),   # yellow
    (200, 130, 255),   # lavender
    (255, 160,  80),   # light blue
]

PILOT_COLOUR  = (0, 255, 0)   # bright green
FONT          = cv2.FONT_HERSHEY_SIMPLEX

# ── ArUco marker panel for reliable ground-truth spatial tracking ───────────────
# We render a single large ArUco marker (DICT_4X4_50, ID=0) at 200×200 px
# centred in a 256×256 white panel.  The 28 px quiet-zone on each side is
# included automatically.  The four outer corners of the 200×200 marker
# region are the reference points for pose estimation, expressed as
# ±ARUCO_REF_HALF in DC-centred panel coordinates.
ARUCO_DICT_ID         = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_ID       = 0
ARUCO_MARKER_PX       = 210     # rendered marker size inside 256×256 panel
ARUCO_OFFSET_PX       = (FRAME_SIZE - ARUCO_MARKER_PX) // 2   # = 23 px quiet zone
ARUCO_REF_HALF        = 128     # half-size used as reference for pose (bin units)
ARUCO_PHYSICAL_SIZE_M = 0.20    # assumed real-world marker side length (metres)

_ARUCO_PANEL: "np.ndarray | None" = None


def _get_aruco_panel() -> np.ndarray:
    """Return a cached 256×256 BGR ArUco marker panel (lazily generated)."""
    global _ARUCO_PANEL
    if _ARUCO_PANEL is not None:
        return _ARUCO_PANEL
    gray = np.ones((FRAME_SIZE, FRAME_SIZE), dtype=np.uint8) * 255  # white bg
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        marker_img = np.zeros((ARUCO_MARKER_PX, ARUCO_MARKER_PX), dtype=np.uint8)
        cv2.aruco.generateImageMarker(dictionary, ARUCO_MARKER_ID,
                                      ARUCO_MARKER_PX, marker_img, borderBits=1)
        o = ARUCO_OFFSET_PX
        gray[o : o + ARUCO_MARKER_PX, o : o + ARUCO_MARKER_PX] = marker_img
    except Exception:
        # Fallback: 8×8 checkerboard
        s = FRAME_SIZE // 8
        for iy in range(FRAME_SIZE):
            for ix in range(FRAME_SIZE):
                gray[iy, ix] = 255 if (iy // s + ix // s) % 2 == 0 else 0
    _ARUCO_PANEL = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _ARUCO_PANEL


def _byte_to_phase(byte_val: int) -> float:
    """Map 0-255 → [0, 2π)."""
    return (byte_val / 255.0) * 2.0 * np.pi


def _phase_to_byte(phase: float) -> int:
    """Map phase in [–π, π] → 0-255."""
    phase = phase % (2.0 * np.pi)
    return int(round(phase / (2.0 * np.pi) * 255)) % 256


def _make_window(size: int, win_type: str) -> np.ndarray:
    if win_type == "blackman":
        w1d = np.blackman(size)
    elif win_type == "none":
        return np.ones((size, size), dtype=np.float64)  # rectangular – no windowing
    else:                          # default: hann
        w1d = np.hanning(size)
    return np.outer(w1d, w1d)


def _cpm_pulse_kernel(shape: str, symbol_frames: int) -> np.ndarray:
    """
    Unit-area pulse shaping kernel of length symbol_frames.
    rect  – flat (classical CPFSK)
    rc    – raised cosine (smoother, less splatter)
    gaussian – GMSK-equivalent BT=0.3 (most compact spectrum)
    """
    t = np.arange(symbol_frames) / max(1, symbol_frames)
    if shape == "rc":
        g = 1.0 - np.cos(np.pi * t)
        g /= g.sum()
    elif shape == "gaussian":
        BT    = 0.3
        sigma = np.sqrt(np.log(2)) / (2 * np.pi * BT)
        g     = np.exp(-0.5 * ((t - 0.5) / sigma) ** 2)
        g    /= g.sum()
    else:   # rect
        g = np.ones(symbol_frames) / max(1, symbol_frames)
    return g.astype(np.float64)


def _get_redundancy_groups(n_carriers: int, mode: str) -> list[list[int]]:
    """
    Return a list of carrier-index groups that share the same encoded byte.

    mode "none"  → [[0],[1],...,[n-1]]       (no redundancy)
    mode "pairs" → [[0,1],[2,3],...]          (pair every two carriers)
    mode "quads" → [[0,1,2,3],[4,5,6,7],...]  (four carriers per byte)
    mode "all8"  → [[0,1,...,n-1]]            (all carriers carry one byte)
    """
    if mode == "pairs":
        return [[i, i + 1] for i in range(0, n_carriers, 2)]
    if mode == "quads":
        return [[i + j for j in range(4)] for i in range(0, n_carriers, 4)]
    if mode == "all8":
        return [list(range(n_carriers))]
    # default: "none"
    return [[i] for i in range(n_carriers)]


class Encoder:
    """Generates looping encoded video frames."""

    def __init__(
        self,
        message:          str   = "Google.com",
        window_type:      str   = "hann",
        symbol_frames:    int   = SYMBOL_FRAMES,
        redundancy_mode:  str   = "none",
        mod_mode:         str   = "psk",
        cpm_h:            float = 0.5,
        cpm_pulse:        str   = "rect",
    ):
        self._lock            = threading.Lock()
        self.message          = message
        self.window_type      = window_type
        self.symbol_frames    = symbol_frames
        self.redundancy_mode  = redundancy_mode
        self._mod_mode        = mod_mode
        self._cpm_h           = float(cpm_h)
        self._cpm_pulse       = cpm_pulse

        self._build_payload()
        self._window = _make_window(FRAME_SIZE, window_type)
        self._frame_counter  = 0
        self._bg_frame       = None
        self._bg_opacity     = 1.0
        # CPM accumulators — one per data carrier
        n = len(DATA_FREQS)
        self._cpm_acc        = np.zeros(n, dtype=np.float64)
        self._cpm_prev_bytes = np.zeros(n, dtype=np.float64)
        self._cpm_kernel     = _cpm_pulse_kernel(cpm_pulse, symbol_frames)
        self._cpm_cumint     = np.cumsum(self._cpm_kernel)

    # ------------------------------------------------------------------ build
    def _build_payload(self):
        """
        Wrap message in 0x00 sentinels, chunk into symbols, and expand
        each symbol across all carriers according to the redundancy mode.

        With redundancy mode "pairs": 4 unique bytes per symbol, each byte
        carried by two adjacent carriers.  The carrier-level symbol list
        always has len(DATA_FREQS) entries.
        """
        groups = _get_redundancy_groups(len(DATA_FREQS), self.redundancy_mode)
        n_eff  = len(groups)   # effective bytes per symbol

        raw = [0] + list(self.message.encode("utf-8")) + [0]
        # pad to multiple of n_eff
        while len(raw) % n_eff:
            raw.append(0)
        self.bytes_data = raw

        # Build carrier-level symbols: each entry has len(DATA_FREQS) bytes
        carrier_symbols = []
        for i in range(0, len(raw), n_eff):
            eff_bytes = raw[i : i + n_eff]
            carr = [0] * len(DATA_FREQS)
            for g_idx, group in enumerate(groups):
                for c_idx in group:
                    carr[c_idx] = eff_bytes[g_idx]
            carrier_symbols.append(carr)

        self.symbols   = carrier_symbols
        self.n_symbols = len(carrier_symbols)
        # Groups exposed for decoder sync
        self._redundancy_groups = groups

    # -------------------------------------------------------------- accessors
    def update_message(self, message: str):
        with self._lock:
            self.message = message
            self._build_payload()
            self._frame_counter = 0

    def update_window(self, win_type: str):
        with self._lock:
            self.window_type = win_type
            self._window = _make_window(FRAME_SIZE, win_type)

    def update_symbol_frames(self, n: int):
        with self._lock:
            self.symbol_frames = max(10, n)

    def update_redundancy_mode(self, mode: str):
        with self._lock:
            if mode in ("none", "pairs", "quads", "all8"):
                self.redundancy_mode = mode
                self._build_payload()
                self._frame_counter = 0

    def update_mod_mode(self, mod_mode: str,
                        cpm_h: float = None,
                        cpm_pulse: str = None):
        """Switch between PSK and CPM at runtime and reset CPM state."""
        with self._lock:
            self._mod_mode = mod_mode
            if cpm_h    is not None: self._cpm_h    = float(cpm_h)
            if cpm_pulse is not None:
                self._cpm_pulse  = cpm_pulse
                self._cpm_kernel = _cpm_pulse_kernel(self._cpm_pulse,
                                                      self.symbol_frames)
                self._cpm_cumint = np.cumsum(self._cpm_kernel)
            self._cpm_acc        = np.zeros(len(DATA_FREQS), dtype=np.float64)
            self._cpm_prev_bytes = np.zeros(len(DATA_FREQS), dtype=np.float64)
            self._frame_counter  = 0

    def update_freqs(self, data_freqs=None, pilot_freqs=None):
        """Update DATA_FREQS / PILOT_FREQS globally and rebuild payload."""
        update_freq_bins(data_freqs, pilot_freqs)
        with self._lock:
            self._build_payload()
            self._frame_counter = 0

    # ── CPM helpers (called inside next_frame while lock is held) ──────────
    def _cpm_phases(self, byte_vals: np.ndarray, frame_in_symbol: int) -> np.ndarray:
        """Instantaneous CPM phase at frame_in_symbol within the current symbol.

        φ(n) = φ_acc + π·h·Δb·∫₀ⁿ g(τ)dτ
        """
        n       = min(frame_in_symbol, len(self._cpm_cumint) - 1)
        cumint  = self._cpm_cumint[n]
        delta   = byte_vals - self._cpm_prev_bytes
        return self._cpm_acc + np.pi * self._cpm_h * delta * cumint

    def _advance_cpm(self, byte_vals: np.ndarray):
        """Update CPM accumulators at the end of a symbol."""
        delta             = byte_vals - self._cpm_prev_bytes
        self._cpm_acc     = (self._cpm_acc +
                             np.pi * self._cpm_h * delta) % (2 * np.pi)
        self._cpm_prev_bytes = byte_vals.copy()

    def get_state(self) -> dict:
        with self._lock:
            total  = self.n_symbols * self.symbol_frames
            fidx   = self._frame_counter % total
            sidx   = fidx // self.symbol_frames
            finsy  = fidx  % self.symbol_frames
            sym    = self.symbols[sidx]
            phases = [_byte_to_phase(b) for b in sym]
            return {
                "message":            self.message,
                "bytes_data":         self.bytes_data,
                "symbols":            self.symbols,
                "n_symbols":          self.n_symbols,
                "symbol_idx":         sidx,
                "frame_in_sym":       finsy,
                "symbol_frames":      self.symbol_frames,
                "total_frames":       total,
                "current_sym":        sym,
                "current_phases":     phases,
                "pilot_freqs":        PILOT_FREQS,
                "data_freqs":         DATA_FREQS,
                "redundancy_mode":    self.redundancy_mode,
                "redundancy_groups":  self._redundancy_groups,
                "mod_mode":           self._mod_mode,
                "cpm_h":              self._cpm_h,
                "cpm_pulse":          self._cpm_pulse,
            }

    def set_bg_frame(self, frame, opacity: float = 1.0):
        """Set (or clear) a background video frame (BGR, any size) and signal
        opacity. Pass frame=None to disable the background. Thread-safe."""
        with self._lock:
            self._bg_frame   = frame
            self._bg_opacity = float(max(0.0, min(1.0, opacity)))

    # ----------------------------------------------------------- frame render
    def next_frame(self) -> np.ndarray:
        """Return a BGR uint8 image: [signal (256×256) | annotated FFT (256×256)]."""
        with self._lock:
            total  = self.n_symbols * self.symbol_frames
            fidx   = self._frame_counter % total
            sidx   = fidx // self.symbol_frames
            finsy  = fidx  % self.symbol_frames
            t      = finsy / self.symbol_frames

            cur_sym  = self.symbols[sidx]
            nxt_sym  = self.symbols[(sidx + 1) % self.n_symbols]

            blend = 0.0
            if t > (1.0 - BLEND_FRAC):
                frac  = (t - (1.0 - BLEND_FRAC)) / BLEND_FRAC
                blend = (1.0 - np.cos(frac * np.pi)) / 2.0

            # Build frequency-domain map
            N  = FRAME_SIZE
            F  = np.zeros((N, N), dtype=complex)

            # ---- pilots ----
            for (fy, fx) in PILOT_FREQS:
                amp = PILOT_AMPLITUDE
                ph  = PILOT_PHASE
                F[fy % N, fx % N]        += amp * np.exp( 1j * ph)
                F[(-fy) % N, (-fx) % N]  += amp * np.exp(-1j * ph)  # Hermitian

            # ---- CPM accumulator advance at symbol boundary ----
            if finsy == 0 and self._frame_counter > 0 and self._mod_mode == "cpm":
                prev_sidx  = ((self._frame_counter - 1) //
                              self.symbol_frames) % self.n_symbols
                prev_bytes = np.array(self.symbols[prev_sidx], dtype=np.float64)
                self._advance_cpm(prev_bytes)

            # ---- data carriers ----
            cur_bytes = np.array(cur_sym, dtype=np.float64)
            if self._mod_mode == "cpm":
                # CPM: instantaneous phase from accumulator + pulse integral
                raw_phases = self._cpm_phases(cur_bytes, finsy)
            else:
                # PSK: static phase with complex-plane blend near symbol end
                raw_phases = np.array([_byte_to_phase(b) for b in cur_sym])
                if blend > 0:
                    nxt_phases = np.array([_byte_to_phase(b) for b in nxt_sym])
                    z = ((1.0 - blend) * np.exp(1j * raw_phases)
                         + blend       * np.exp(1j * nxt_phases))
                    raw_phases = np.angle(z)

            active_phases = []
            for i, (fy, fx) in enumerate(DATA_FREQS):
                ph  = float(raw_phases[i])
                active_phases.append(ph)
                amp = DATA_AMPLITUDE
                F[fy % N, fx % N]        += amp * np.exp( 1j * ph)
                F[(-fy) % N, (-fx) % N]  += amp * np.exp(-1j * ph)

            # ---- spatial-domain signal (windowed) ----
            spatial = np.real(ifft2(F)) * N * N
            spatial *= self._window
            # Normalise to [0, 255]
            lo, hi = spatial.min(), spatial.max()
            if hi > lo:
                spatial = (spatial - lo) / (hi - lo) * 255.0
            else:
                spatial = np.full_like(spatial, 128.0)
            sig_u8 = spatial.astype(np.uint8)
            sig_bgr = cv2.cvtColor(sig_u8, cv2.COLOR_GRAY2BGR)

            # ---- blend video background if one is set ----
            if self._bg_frame is not None:
                bg = cv2.resize(self._bg_frame, (N, N))
                sig_bgr = cv2.addWeighted(
                    bg,      1.0 - self._bg_opacity,
                    sig_bgr, self._bg_opacity,
                    0
                )
                # FFT of what actually appears on screen (signal + video mix)
                composite_gray = cv2.cvtColor(sig_bgr, cv2.COLOR_BGR2GRAY)
            else:
                composite_gray = sig_u8

            # ---- ArUco panel (full 256×256, replaces QR overlay) ----
            aruco_panel = _get_aruco_panel().copy()

            # ---- actual FFT of the windowed signal (no QR artefacts) ----
            # Apply window to the clean signal BEFORE QR/ArUco was introduced
            f_disp    = composite_gray.astype(np.float32) / 255.0
            F_act_raw = fft2(f_disp * self._window)
            F_act_sh  = fftshift(F_act_raw)
            _c        = FRAME_SIZE // 2
            act_phases = [
                float(np.angle(F_act_sh[(_c + fy) % FRAME_SIZE,
                                        (_c + fx) % FRAME_SIZE]))
                for (fy, fx) in DATA_FREQS
            ]
            fft_panel = self._make_fft_panel(
                F_act_raw, cur_sym, act_phases, blend
            )

            self._frame_counter += 1

        # 3-panel output: signal | aruco | actual-FFT-with-circles
        combined = np.hstack([sig_bgr, aruco_panel, fft_panel])
        return combined

    # --------------------------------------------------- FFT annotation panel
    def _make_fft_panel(
        self,
        F:             np.ndarray,
        cur_sym:       list[int],
        active_phases: list[float],
        blend:         float,
        label:         str = "",
    ) -> np.ndarray:
        N    = FRAME_SIZE
        zoom = FFT_ZOOM_BINS
        disp = 256   # output panel size in pixels
        half = disp // 2

        # Magnitude spectrum in shifted coordinates
        F_shift = fftshift(F)
        mag     = np.abs(F_shift)
        mag_log = np.log1p(mag)

        # Crop ±zoom bins around centre
        c = N // 2
        crop = mag_log[c - zoom : c + zoom, c - zoom : c + zoom]  # (2*zoom, 2*zoom)

        # Scale to 0-255, resize to disp×disp
        if crop.max() > 0:
            crop = crop / crop.max() * 255.0
        crop_u8   = crop.astype(np.uint8)
        crop_big  = cv2.resize(crop_u8, (disp, disp), interpolation=cv2.INTER_NEAREST)
        panel     = cv2.applyColorMap(crop_big, cv2.COLORMAP_INFERNO)

        # pixel-per-bin scale
        scale = disp / (2.0 * zoom)   # px / bin

        def freq_to_px(fy, fx):
            """Convert (fy, fx) bin offset from DC → pixel coords on panel."""
            px = int(half + fx * scale)
            py = int(half + fy * scale)
            return px, py

        def draw_arrow(img, cx, cy, phase, length, colour, thickness=2):
            ax = cx + int(length * np.cos(phase))
            ay = cy - int(length * np.sin(phase))   # y is inverted on screen
            cv2.arrowedLine(img, (cx, cy), (ax, ay), colour, thickness,
                            tipLength=0.35)

        # ── draw pilots ──
        for (fy, fx) in PILOT_FREQS:
            px, py = freq_to_px(fy, fx)
            if 5 < px < disp-5 and 5 < py < disp-5:
                cv2.circle(panel, (px, py), 9, PILOT_COLOUR, 2)
                draw_arrow(panel, px, py, PILOT_PHASE, ARROW_LENGTH, PILOT_COLOUR)
                cv2.putText(panel, "P", (px+10, py-5), FONT, 0.35,
                            PILOT_COLOUR, 1, cv2.LINE_AA)

        # ── draw data carriers ──
        for i, (fy, fx) in enumerate(DATA_FREQS):
            px, py = freq_to_px(fy, fx)
            colour = CARRIER_COLOURS[i]
            if 5 < px < disp-5 and 5 < py < disp-5:
                # circle for the bin
                cv2.circle(panel, (px, py), 9, colour, 2)
                true_ph = active_phases[i]
                # true-phase arrow (solid)
                draw_arrow(panel, px, py, true_ph, ARROW_LENGTH, colour, 2)
                # label with byte value and phase
                bv        = cur_sym[i]
                carr_lbl  = f"B{i}:{bv:03d}"   # renamed to avoid shadowing outer `label`
                cv2.putText(panel, carr_lbl, (px + 10, py + 4), FONT, 0.28,
                            colour, 1, cv2.LINE_AA)

        # ── transition-blend indicator bar ──
        bar_h = int(blend * (disp - 1))
        cv2.rectangle(panel, (0, disp - bar_h), (4, disp - 1),
                      (200, 200, 200), -1)

        # ── legend ──
        cv2.putText(panel, "GREEN=pilot  CIRCLE=bin",
                    (4, 12), FONT, 0.28, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, "ARROW=phase  BAR=blend",
                    (4, 23), FONT, 0.28, (220, 220, 220), 1, cv2.LINE_AA)
        if label:
            cv2.putText(panel, label,
                        (4, disp - 6), FONT, 0.3, (160, 200, 255), 1, cv2.LINE_AA)

        return panel
