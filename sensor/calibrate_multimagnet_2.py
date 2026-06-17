# calibrate_multimagnet.py
#
# 2x2 magnet array calibration for binary (pressed / not pressed) detection.
# Magnets sit at z=10mm above the WT901 sensor at (0,0,0).
#   M1: (0,   0  ) polarity +1
#   M2: (0,   1.5) polarity -1
#   M3: (1.5, 0  ) polarity -1
#   M4: (1.5, 1.5) polarity +1
#
# Calibration states collected:
#   - baseline (all unloaded)
#   - each single magnet pressed
#   - each pair of adjacent magnets pressed (optional)
#
# Output: multimagnet_cal.npy  containing:
#   baseline       : (3,) float array   — raw sensor mean when unloaded
#   centroids      : dict {label: (3,) corrected centroid}
#   labels         : list of label strings in same order as centroids
#   axis_std       : (3,) float array   — per-axis std across all centroids
#                    used to normalise distances in the classifier
#   cov_inv        : (3,3) float array  — inverse covariance across all
#                    per-sample data (for optional Mahalanobis distance);
#                    saved only when enough raw samples are collected
#
# Axis weighting strategy (choose one in classify_multimagnet.py):
#   1. Z-score normalisation  — divide each axis by axis_std  (default)
#   2. Mahalanobis distance   — use cov_inv  (upgrade when data allows)
#   3. Manual weights         — edit AXIS_WEIGHTS in the classifier

import serial
import struct
import numpy as np

PORT = '/dev/ttyUSB0'
BAUD = 9600
N_SAMPLES = 80   # samples averaged per state

# ---------- States to calibrate ----------
SINGLE_STATES = [
    ('M1', 'Press ONLY magnet 1 (top-left,  +1 polarity)'),
    ('M2', 'Press ONLY magnet 2 (top-right, -1 polarity)'),
    ('M3', 'Press ONLY magnet 3 (bot-left,  -1 polarity)'),
    ('M4', 'Press ONLY magnet 4 (bot-right, +1 polarity)'),
]

PAIR_STATES = [
    ('M1+M2', 'Press magnets 1 AND 2 simultaneously (top row)'),
    ('M1+M3', 'Press magnets 1 AND 3 simultaneously (left col)'),
    ('M2+M4', 'Press magnets 2 AND 4 simultaneously (right col)'),
    ('M3+M4', 'Press magnets 3 AND 4 simultaneously (bot row)'),
    ('M1+M4', 'Press magnets 1 AND 4 simultaneously (diagonal +)'),
    ('M2+M3', 'Press magnets 2 AND 3 simultaneously (diagonal -)'),
]
# -----------------------------------------


def read_mag_samples(port, baud, n_samples=N_SAMPLES):
    """Return all raw (mx, my, mz) samples as a (n_samples, 3) float array."""
    ser = serial.Serial(port, baud, timeout=2)
    readings = []
    buffer = bytearray()
    print(f"  Collecting {n_samples} samples...", end='', flush=True)
    while len(readings) < n_samples:
        raw = ser.read(11)
        if not raw:
            continue
        buffer.extend(raw)
        while len(buffer) >= 11:
            if buffer[0] == 0x55 and buffer[1] == 0x54:
                mx, my, mz = struct.unpack('<hhh', bytes(buffer[2:8]))
                readings.append((mx, my, mz))
            if buffer[0] == 0x55:
                buffer = buffer[11:]
            else:
                buffer.pop(0)
    ser.close()
    arr = np.array(readings[:n_samples], dtype=float)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    print(f" done.")
    print(f"    mean: X={mean[0]:+8.1f}  Y={mean[1]:+8.1f}  Z={mean[2]:+8.1f}")
    print(f"    std:  X={std[0]:7.2f}   Y={std[1]:7.2f}   Z={std[2]:7.2f}")
    return arr   # return full sample array, not just mean


def collect_state(label, instruction, baseline):
    print(f"\n--- {label} ---")
    print(f"  {instruction}.")
    input("  Hold steady, press Enter to measure...")
    samples   = read_mag_samples(PORT, BAUD)
    mean      = samples.mean(axis=0)
    corrected = mean - baseline
    print(f"  Corrected: Bx={corrected[0]:+8.1f}  By={corrected[1]:+8.1f}  Bz={corrected[2]:+8.1f}")
    return corrected, samples   # return both centroid and raw samples


def main():
    print("=== Multi-magnet array calibration ===")
    print()
    print("Magnet layout (top view):")
    print("  M1(+1)  M2(-1)")
    print("  M3(-1)  M4(+1)")
    print()

    # ---- Baseline ----
    input("Ensure ALL magnets are fully unloaded. Press Enter to measure baseline...")
    baseline_samples = read_mag_samples(PORT, BAUD)
    baseline = baseline_samples.mean(axis=0)
    print(f"Baseline: Bx={baseline[0]:+8.1f}  By={baseline[1]:+8.1f}  Bz={baseline[2]:+8.1f}")

    centroids    = {}
    all_samples  = []   # accumulate raw samples for covariance estimation

    # ---- Unloaded centroid ----
    print("\n--- unloaded ---")
    print("  Keep everything unloaded (re-measuring for centroid).")
    input("  Press Enter to measure...")
    samples = read_mag_samples(PORT, BAUD)
    centroids['unloaded'] = samples.mean(axis=0) - baseline
    all_samples.append(samples - baseline)

    # ---- Singles ----
    for label, instruction in SINGLE_STATES:
        centroid, samples = collect_state(label, instruction, baseline)
        centroids[label]  = centroid
        all_samples.append(samples - baseline)

    # ---- Pairs (optional) ----
    do_pairs = input("\nCalibrate pair presses too? (y/n): ").strip().lower() == 'y'
    if do_pairs:
        for label, instruction in PAIR_STATES:
            centroid, samples = collect_state(label, instruction, baseline)
            centroids[label]  = centroid
            all_samples.append(samples - baseline)

    labels = list(centroids.keys())

    # ---- Compute axis_std from centroid spread ----
    # This captures how much each axis discriminates across states.
    # A near-zero std on an axis means it carries little information.
    centroid_matrix = np.array([centroids[l] for l in labels])   # (N, 3)
    axis_std = centroid_matrix.std(axis=0)

    # Guard against degenerate axes (std < 1) — set a floor so we never divide by ~0
    axis_std_safe = np.where(axis_std < 1.0, 1.0, axis_std)

    print("\n=== Axis spread across centroids (axis_std) ===")
    print(f"  Bx std: {axis_std[0]:8.2f}")
    print(f"  By std: {axis_std[1]:8.2f}")
    print(f"  Bz std: {axis_std[2]:8.2f}")
    print("  (Larger → that axis varies more across states → more discriminative)")

    # ---- Optional: inverse covariance for Mahalanobis ----
    all_samples_arr = np.vstack(all_samples)   # (N_states * N_SAMPLES, 3)
    cov_inv = None
    try:
        cov     = np.cov(all_samples_arr.T)    # (3, 3)
        cov_inv = np.linalg.inv(cov)
        print("\n=== Per-state sample covariance (full) ===")
        print(np.array2string(cov, precision=1, suppress_small=True))
        print("Inverse covariance computed — Mahalanobis distance available in classifier.")
    except np.linalg.LinAlgError:
        print("\nWARNING: Covariance matrix is singular — Mahalanobis not available.")
        print("         Increase N_SAMPLES or collect more varied states.")

    # ---- Centroid summary ----
    print("\n=== Centroid summary ===")
    print(f"  {'Label':12s}  {'Bx':>10}  {'By':>10}  {'Bz':>10}")
    for lbl, c in centroids.items():
        print(f"  {lbl:12s}  {c[0]:>10.1f}  {c[1]:>10.1f}  {c[2]:>10.1f}")

    # ---- Pairwise separation in normalised space ----
    print("\n=== Pairwise separation (Z-score normalised Euclidean distance) ===")
    bad = False
    WARN_THRESHOLD = 1.5   # in normalised units
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            diff = (centroids[l1] - centroids[l2]) / axis_std_safe
            d    = np.linalg.norm(diff)
            flag = "  <-- WARNING: too close!" if d < WARN_THRESHOLD else ""
            print(f"  {l1:12s} <-> {l2:12s}  norm_dist={d:6.3f}{flag}")
            if flag:
                bad = True
    if bad:
        print("\nWARNING: Some states are poorly separated in normalised space. Consider:")
        print("  - Increasing compression distance")
        print("  - Adjusting magnet positions or strength")
        print("  - Switching to Mahalanobis distance in the classifier")
    else:
        print("\nSeparation looks good.")

    # ---- Save ----
    save_dict = {
        'baseline':  baseline,
        'centroids': centroids,
        'labels':    labels,
        'axis_std':  axis_std_safe,   # (3,) — used for Z-score normalisation
    }
    if cov_inv is not None:
        save_dict['cov_inv'] = cov_inv   # (3,3) — used for Mahalanobis

    np.save('multimagnet_cal_2.npy', save_dict)
    print("\nSaved to multimagnet_cal_2.npy")
    print(f"  axis_std saved:  {axis_std_safe}")
    if cov_inv is not None:
        print("  cov_inv saved:   yes (Mahalanobis available)")


if __name__ == '__main__':
    main()