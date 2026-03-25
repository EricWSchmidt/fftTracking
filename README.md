# CPT Multi-Frequency Phase Encoder / Decoder

A screen-to-camera optical data link that encodes a text string as **phase angles
in the 2-D discrete Fourier transform** of a displayed image.  A webcam reads the
signal and a browser-based decoder recovers the original text.

---

## Quick Start

```bash
pip install flask numpy opencv-python scipy Pillow
python server.py          # open http://localhost:5000
```

Point your webcam at the **left half** of the encoder display.

---

## How It Works

### 1 — Encoding (spatial domain → frequency domain)

The encoder builds a **complex frequency-domain map** `F[fy, fx]` of size
`N × N` (`N = 256`).  It places energy at a small set of known frequency-bin
coordinates and sets the **phase** of each bin to carry one byte of data.

#### Byte → Phase mapping

$$\varphi_i = \frac{b_i}{255} \cdot 2\pi \qquad b_i \in [0, 255]$$

The inverse mapping recovers the byte:

$$b_i = \operatorname{round}\!\left(\frac{\varphi_i \bmod 2\pi}{2\pi} \cdot 255\right) \bmod 256$$

A single phase measurement therefore has a **resolution of**

$$\Delta\varphi_\text{bit} = \frac{2\pi}{256} \approx 1.4°/\text{LSB}$$

#### Real-valued constraint (Hermitian symmetry)

To produce a real-valued spatial image the frequency map must satisfy:

$$F[-f_y, -f_x] = F[f_y, f_x]^*$$

Every data carrier `(fy, fx)` therefore also implies a conjugate at
`(-fy, -fx)`.  To avoid two carriers interfering with each other **none of the
8 data-carrier positions is the mirror-image of another one**.

#### 2-D IFFT → spatial signal

$$s[y, x] = \operatorname{Re}\!\left[\mathcal{F}^{-1}\{F\}[y, x]\right] \cdot N^2$$

The spatial image is **multiplied by a 2-D window** (Hann or Blackman, chosen in
encoder settings) to suppress the spectral leakage that would otherwise spread
energy from the strong pilot bins across neighbouring data bins.

#### Symbol transition blending

Every symbol is held for `symbol_frames` frames.  During the last `20 %` of
each symbol the phase of each carrier is **interpolated in the complex plane**
(raised-cosine cross-fade) to avoid a discontinuous phase jump that would
smear energy across the spectrum:

$$z_\text{blend} = (1-\alpha)\,e^{j\varphi_\text{cur}} + \alpha\,e^{j\varphi_\text{next}}, \qquad \alpha = \frac{1-\cos(\pi\,t')}{2}$$

where `t'` is the normalised position within the blend region.

---

### 2 — Pilot carriers

Four pilot bins at symmetric corners `(±9, ±9)` carry a **fixed known
phase of 0 rad** at a fixed amplitude.

$$F_\text{pilot}[9, 9] = A_p e^{j \cdot 0} = A_p$$

Purposes:

| Purpose | Mechanism |
|---------|-----------|
| Global phase anchor | The measured pilot phase ≈ any global phase offset introduced by the channel |
| Geometric de-skew | Four pilot peaks form a square; their displacement from expected positions reveals camera rotation, zoom, and keystone distortion |
| Per-symbol alignment | Because pilots are constant they can be averaged across many frames to sharpen the distortion estimate |

---

### 3 — Decoding pipeline

```
webcam frame
    │
    ▼
resize to 256×256  (bilinear)
    │
    ▼
normalise to [0, 1]
    │
    ▼
multiply 2-D window  W[y,x]     ← suppresses leakage
    │
    ▼
2-D DFT   F = FFT2(frame · W)
    │
    ▼
fftshift → magnitude · phase map
    │
    ├──► temporal average over N frames  (reduces noise)
    │
    ▼
find pilot peaks (±search_r bins around expected position)
    │
    ├──► global phase offset  φ_global = mean(measured pilot phases)
    │
    ├──► (optional) homography correction using 4 pilot positions
    │
    ▼
read phase at each data-carrier bin
    │
    ▼
apply corrections:  φ_corr = φ_meas − φ_global − φ_offset − φ_nudge[i]
    │
    ▼
byte = round(φ_corr / 2π · 255) mod 256
    │
    ▼
multi-symbol ring buffer → scan for 0x00 sentinels → extract message
```

---

## Encoder Settings

### Message
The text string to transmit.  It is UTF-8 encoded, padded with a leading and
trailing `0x00` null byte (sentinel), then padded to a multiple of 8 bytes
(one row per symbol).

### Window (Hann / Blackman)
Applied **in the spatial domain before the FFT** (both encoder and decoder).
A window `w[y,x] = w_1[y] · w_1[x]` trades main-lobe width for sidelobe
suppression.

| Window | Main lobe (bins) | Peak sidelobe (dB) | When to use |
|--------|------------------|--------------------|-------------|
| **Hann** | 4 | −31.5 | Good default; lower inter-bin leakage |
| **Blackman** | 6 | −57.3 | When carriers are close together or SNR is low |

The sidelobe level determines how much energy from one carrier bleeds into a
neighbouring carrier's bin, which sets a noise floor on the phase measurement.
For carriers spaced ≥ 10 bins apart the Hann window is sufficient; Blackman
is advisable at < 8 bin spacing.

### Symbol Duration (frames)
How many frames each symbol (group of 8 bytes) is displayed before advancing.

$$T_\text{sym} = \frac{N_\text{frames}}{f_\text{enc}}$$

A longer symbol gives the temporal averager more frames to accumulate, directly
improving SNR:

$$\text{SNR}_\text{avg} \propto \sqrt{N_\text{avg}} \propto \sqrt{\min(T_\text{avg},\, T_\text{sym})}$$

At 30 fps, 90 frames = 3 s per symbol.  With `temporal_avg = 5` frames of
decoder accumulation the effective integration is ~0.5 s even during transitions.

---

## Decoder Settings

### Window (Hann / Blackman)
Same trade-off as the encoder window.  **Both encoder and decoder should ideally
use the same window type** so the combined spectral response is predictable.
If the encoder window is unknown, Blackman is the safer choice.

### Phase Offset (−180° … +180°)
A **global manual phase correction** added to all decoded carrier phases after
automatic pilot-based correction.

**When to adjust:**  
If all decoded bytes are consistently wrong by the same amount (e.g., every
character is shifted by the same ASCII distance) the total phase offset is
incorrect.  Increase or decrease this slider until bytes align.

The total correction applied to each bin is:

$$\varphi_\text{corr} = \varphi_\text{meas} - \underbrace{\hat{\varphi}_\text{pilot}}_{\text{auto}} - \underbrace{\varphi_\text{offset}}_{\text{this slider}} - \varphi_\text{nudge}[i]$$

### Temporal Averaging (1 … 30 frames)
The decoder accumulates `N` complex FFT frames and averages them **in the
complex plane** before reading phases:

$$\bar{F}[f_y, f_x] = \frac{1}{N}\sum_{k=0}^{N-1} F_k[f_y, f_x]$$

Coherent (complex) averaging improves SNR by:

$$\Delta\text{SNR} = 10\log_{10}(N)\,\text{dB}$$

e.g. 5 frames → +7 dB, 10 frames → +10 dB, 30 frames → +15 dB.

**Trade-off:** a large `N` assumes the phase is constant over those frames.
During a symbol transition blending, the average smears the phase, temporarily
corrupting that byte.  Set `N` to be less than `symbol_frames` (encoder) to
stay within the stable part of the symbol.

### Pilot Search Radius (1 … 15 bins)
When the camera frame is scaled, rotated, or cropped, pilot peaks shift from
their expected bin positions.  The decoder searches within ±`r` bins of each
expected pilot location and takes the **argmax** of the local magnitude patch.

Too small: misses pilots when geometric distortion is large.  
Too large: may lock onto a data carrier instead of a pilot,
corrupting the phase-offset estimate.

A value of `r ≈ 0.5 × (bin-spacing between pilot and nearest data carrier)`
is safe.  The default of 6 keeps clear of the nearest data carrier at bin 14.

### Magnitude Threshold (1 … 50 %)
A carrier bin is declared **undetected** (shown as `·`) if its SNR ratio
(signal vs. local noise annulus) falls below `max(1.5, threshold × 20)`.

$$\text{SNR}_i = \frac{|F[f_{y_i}, f_{x_i}]|}{\mu_\text{noise,annulus}(i)}$$

Lowering the threshold accepts weaker signals but increases the chance of
decoding noise as a valid byte.  Raise it if a carrier with reasonable magnitude
is flickering; lower it if a carrier is consistently showing `·`.

### FFT Zoom (8 … 64 bins)
Controls the zoom level of the annotated FFT display: ±`zoom` bins around DC
are shown.  This is **display only** — it has no effect on decoding.

Wider zoom → more context, smaller circles.  
Narrower zoom → exaggerated positions, easier to see pilot displacement.

### Geometric Correction (ON / OFF)
When ON, the four measured pilot peak positions are used to compute a
**homography** (perspective transform) of the full complex FFT map before
reading data-carrier phases.  This corrects:

- Rotation of the screen in the camera frame
- Zoom / distance variation
- Keystone / trapezoid perspective distortion

The homography `H` maps measured pilot positions → expected pilot positions:

$$\begin{pmatrix}x'\\y'\\1\end{pmatrix} \sim H \begin{pmatrix}x\\y\\1\end{pmatrix}$$

It is solved via Direct Linear Transform (OpenCV `findHomography`) from the
four pilot correspondences.

**When to disable:** if pilots are not detected correctly (high displacement or
anomalous phase errors) the homography may make things worse.  Turn it off and
rely on pilot-phase-only correction until the pilot peaks are clean.

### Per-Carrier Phase Nudge (C0 … C7, −180° … +180°)
Fine-grained per-carrier phase correction applied **after** the global offset:

$$\varphi_\text{corr,i} = \varphi_\text{meas,i} - \hat{\varphi}_\text{pilot} - \varphi_\text{offset} - \underbrace{\varphi_\text{nudge,i}}_{\text{this slider}}$$

Useful when individual carriers experience a different phase bias, e.g. due to
locally different screen brightness, sensor non-uniformity, or being near
a Moiré resonance frequency.

**How to set:** with a known test message, nudge carrier `i` until its decoded
byte matches the expected value.  The required nudge equals the steady-state
phase error of that carrier.

---

## Carrier Layout

```
Frequency plane (fy horizontal, fx vertical, DC at centre)

           fx →
    -20  -14   -9   0   +9  +14  +20
fy
-20         P3            C5  C7
-14         P3        C0  C2
 -9    P2   P3   P3
  0         C3        ·   C0  C1  C4
 +9    P0   P0   P0
+14         P1        C1  C3
+20    C6        P1       C4  C6

  P0–P3  = pilot carriers  (±9, ±9)  phase = 0 rad (fixed)
  C0–C7  = data carriers
```

Actual carrier frequencies:

| Carrier | (fy, fx) | Conjugate at |
|---------|----------|--------------|
| C0 | (+14, 0) | (−14, 0) |
| C1 | (0, +14) | (0, −14) |
| C2 | (+14, +14) | (−14, −14) |
| C3 | (+14, −14) | (−14, +14) |
| C4 | (+20, 0) | (−20, 0) |
| C5 | (0, +20) | (0, −20) |
| C6 | (+20, +10) | (−20, −10) |
| C7 | (+10, +20) | (−10, −20) |

---

## Known Shortcomings of CPT in This Channel

1. **Phase wrapping** — phase is defined mod 2π; byte 0 and byte 255 differ by
   only 1/255 of a full cycle and are easily confused.
2. **No shared clock** — encoder and decoder have independent frame rates and
   timestamps; frame-rate drift causes a slowly rotating phase bias.
3. **Codec compression** — webcam MJPEG/H.264 block-DCT destroys fine phase
   relationships; a single quantisation step can rotate a bin's phase by tens
   of degrees.
4. **Monitor gamma non-linearity** — display gamma (≈2.2) distorts the IFFT
   signal amplitude before it reaches the camera, adding harmonic content.
5. **Moiré / pixel-grid aliasing** — the display pixel grid and camera sensor
   grid beat together, shifting apparent bin positions unpredictably.
6. **Perspective / geometric distortion** — any rotation or keystoning smears
   energy across FFT bins; pilot correction helps but cannot fully recover
   severe distortion.
7. **Ambient light** — room light adds a DC component and spatially-varying
   gain, shifting amplitudes non-uniformly.
8. **Camera noise** — shot/readout noise adds Gaussian phase noise of
   σ_φ ≈ 1/SNR radians at every bin.
9. **Low bitrate** — 8 carriers × 1 byte = 8 bytes per symbol; at 90 frames
   and 30 fps that is ≈ 0.09 bytes/second.
10. **Symbol transition corruption** — frames captured during the blend region
    carry a mixture of two symbols and are decoded incorrectly.
11. **No error correction** — a single noisy frame corrupts the associated byte
    with no way to detect or repair it without FEC.
12. **Flicker** — high-amplitude high-frequency patterns can cause visible
    display flicker at certain carrier frequencies.
