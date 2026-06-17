# classify_multimagnet.py
#
# Live nearest-centroid classifier for the 2x2 magnet array.
# Loads multimagnet_cal.npy produced by calibrate_multimagnet.py.
#
# Noise handling:
#   - Exponential moving average (EMA) smooths the raw signal
#   - Label debounce only emits a new state after CONFIRM consecutive frames agree
#
# Axis weighting (select via DISTANCE_MODE below):
#   'zscore'      — divide each axis by its std across centroids (default)
#                   accounts for the fact that the axis directly over a magnet
#                   dominates the raw Euclidean distance unfairly
#   'mahalanobis' — full inverse-covariance weighting; requires cov_inv in cal file
#   'manual'      — multiply axes by AXIS_WEIGHTS before computing distance
#   'raw'         — unweighted Euclidean (original behaviour)

import serial
import struct
import numpy as np

PORT = '/dev/ttyUSB1'
BAUD = 9600

# ---- Smoothing & debounce ----
ALPHA   = 0.7   # EMA weight on new sample (0.1 = very smooth/slow, 0.7 = responsive)
CONFIRM = 3     # frames the same label must appear before it's accepted

# ---- Distance mode ----
# Choose one of: 'zscore' | 'mahalanobis' | 'manual' | 'raw'
DISTANCE_MODE = 'zscore'

# Used only when DISTANCE_MODE = 'manual'
# Increase weight on axes that carry the most discriminative signal.
# Example: magnet directly above sensor dominates Bz → boost Z.
AXIS_WEIGHTS = np.array([1.0, 1.0, 2.0])

# Confidence threshold: if best distance exceeds this, report "unknown".
# In 'zscore' and 'mahalanobis' modes this is in normalised units (e.g. 3–10).
# In 'raw' or 'manual' mode use raw-scale units (e.g. 50000).
DIST_THRESHOLD = 5.0   # tune after watching live dist_best values

# ---- Load calibration ----
cal       = np.load('multimagnet_cal_sim_data.npy', allow_pickle=True).item()
BASELINE  = cal['baseline']
centroids = cal['centroids']   # dict {label: (3,) corrected array}
labels    = cal['labels']

centroid_matrix = np.array([centroids[l] for l in labels])  # (N, 3)

# Per-axis normalisation weights — computed from centroid spread at calibration time
AXIS_STD = cal.get('axis_std', np.ones(3))   # (3,) fallback to ones if missing

# Inverse covariance — only present if enough samples were collected
COV_INV = cal.get('cov_inv', None)           # (3, 3) or None

# Validate distance mode
if DISTANCE_MODE == 'mahalanobis' and COV_INV is None:
    print("WARNING: 'mahalanobis' requested but cov_inv not in cal file.")
    print("         Falling back to 'zscore'. Re-run calibration to enable Mahalanobis.")
    DISTANCE_MODE = 'zscore'

print("Loaded calibration:")
print(f"  States:        {labels}")
print(f"  Distance mode: {DISTANCE_MODE}")
print(f"  axis_std:      Bx={AXIS_STD[0]:.2f}  By={AXIS_STD[1]:.2f}  Bz={AXIS_STD[2]:.2f}")
print(f"  EMA alpha: {ALPHA}  |  Debounce frames: {CONFIRM}")
print(f"  Dist threshold: {DIST_THRESHOLD}")
print()


# ---- Pre-compute normalised centroid matrix for efficiency ----
if DISTANCE_MODE == 'zscore':
    _norm_centroids = centroid_matrix / AXIS_STD        # (N, 3)
elif DISTANCE_MODE == 'manual':
    _norm_centroids = centroid_matrix * AXIS_WEIGHTS    # (N, 3)
else:
    _norm_centroids = centroid_matrix                   # used as-is


def read_mag_packet(buffer):
    while len(buffer) >= 11:
        if buffer[0] == 0x55 and buffer[1] == 0x54:
            # AFTER:
            # In read_mag_samples (calibrate) and read_mag_packet (classify)
            # replace whichever remap you currently have with:
            rx, ry, rz = struct.unpack('<hhh', bytes(buffer[2:8]))
            mx = rz    # world X
            my = rx    # world Y
            mz = ry    # world Z — confirmed: responds to magnet above sensor
            return np.array([mx, my, mz], dtype=float), buffer[11:]
        elif buffer[0] == 0x55:
            return None, buffer[11:]
        else:
            buffer.pop(0)
    return None, buffer


def classify(corrected):
    """
    Classify a corrected (baseline-subtracted) magnetometer reading.

    Returns
    -------
    label     : str    — best matching state label, or 'unknown'
    best_dist : float  — distance to nearest centroid (in chosen metric)
    margin    : float  — gap between nearest and second-nearest centroid
    """
    if DISTANCE_MODE == 'zscore':
        normed = corrected / AXIS_STD
        dists  = np.linalg.norm(_norm_centroids - normed, axis=1)

    elif DISTANCE_MODE == 'mahalanobis':
        # Mahalanobis: sqrt((x-c)^T @ COV_INV @ (x-c)) for each centroid
        diffs  = centroid_matrix - corrected           # (N, 3)
        dists  = np.sqrt(np.einsum('ni,ij,nj->n', diffs, COV_INV, diffs))

    elif DISTANCE_MODE == 'manual':
        weighted = corrected * AXIS_WEIGHTS
        dists    = np.linalg.norm(_norm_centroids - weighted, axis=1)

    else:   # 'raw'
        dists = np.linalg.norm(centroid_matrix - corrected, axis=1)

    sorted_idx = np.argsort(dists)
    best_dist  = dists[sorted_idx[0]]
    margin     = dists[sorted_idx[1]] - best_dist if len(sorted_idx) > 1 else 0.0

    if best_dist > DIST_THRESHOLD:
        return "unknown", best_dist, margin

    return labels[sorted_idx[0]], best_dist, margin


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("Reading... (Ctrl+C to stop)")
    print()

    header = (
        f"{'Bx':>10}  {'By':>10}  {'Bz':>10}  |  "
        f"{'dist':>8}  {'margin':>8}  | {'confirmed':>14}"
    )
    print(header)
    print("-" * len(header))

    buffer        = bytearray()
    smoothed      = None       # EMA state — initialised on first packet
    pending_state = None
    pending_count = 0
    confirmed     = "unloaded"

    try:
        while True:
            raw = ser.read(11)
            if not raw:
                continue
            buffer.extend(raw)
            reading, buffer = read_mag_packet(buffer)
            if reading is None:
                continue

            corrected = reading - BASELINE

            # EMA smoothing
            if smoothed is None:
                smoothed = corrected.copy()
            else:
                smoothed = ALPHA * corrected + (1 - ALPHA) * smoothed

            # Classify on smoothed signal
            raw_label, dist, margin = classify(smoothed)

            # Debounce
            if raw_label == pending_state:
                pending_count += 1
            else:
                pending_state = raw_label
                pending_count = 1

            if pending_count >= CONFIRM:
                confirmed = pending_state

            print(
                f"{corrected[0]:>10.1f}  {corrected[1]:>10.1f}  {corrected[2]:>10.1f}  |  "
                f"{dist:>8.3f}  {margin:>8.3f}  |  "
                f" {confirmed:>14}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()