# calibrate_multimagnet.py
#
# 2x2 magnet array calibration for binary (pressed / not pressed) detection.
#
# Magnet layout (top view, sensor at origin):
#   M1(+1)  M2(-1)
#   M3(-1)  M4(+1)
#
# Output: multimagnet_cal.npy containing everything needed by the simulator:
#   baseline          : (3,)    raw sensor mean when unloaded
#   centroids         : dict {label: (3,) corrected centroid}
#   labels            : list of label strings
#   axis_std          : (3,)    per-axis std across centroids (Z-score normalisation)
#   cov_inv           : (3,3)   inverse covariance (Mahalanobis; saved if invertible)
#   noise_std         : (3,)    per-axis sensor noise std from unloaded samples  ← NEW
#   press_depth_mm    : float   how far each magnet was pressed during calibration ← NEW
#   rest_height_mm    : float   magnet face-to-sensor distance at rest             ← NEW
#   magnet_diameter_mm: float   physical magnet diameter                           ← NEW
#   magnet_thickness_mm: float  physical magnet height                             ← NEW
#   magnet_layout     : list    [(x_mm, y_mm, polarity), ...]                      ← NEW

import serial
import struct
import numpy as np

PORT      = '/dev/ttyUSB1'
BAUD      = 9600
N_SAMPLES = 80   # samples averaged per state

# =============================================================================
# PHYSICAL GEOMETRY — fill in your actual hardware dimensions
# =============================================================================
REST_HEIGHT_MM      = 10.0   # mm — magnet face to sensor face when unloaded
PRESS_DEPTH_MM      = 5.0    # mm — how far you press each magnet down
MAGNET_DIAMETER_MM  = 9.5    # mm — cylindrical magnet diameter
MAGNET_THICKNESS_MM = 3.0    # mm — cylindrical magnet height

# Array layout: (x_mm, y_mm, polarity)
# Polarity +1 = north pole faces sensor, -1 = south pole faces sensor
MAGNET_LAYOUT = [
    (0.0,  0.0,  +1),   # M1
    (0.0,  15.0, -1),   # M2
    (15.0, 0.0,  -1),   # M3
    (15.0, 15.0, +1),   # M4
]
# =============================================================================

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


def read_mag_samples(port, baud, n_samples=N_SAMPLES):
    """Return all raw (mx, my, mz) samples as (n_samples, 3) float array."""
    ser      = serial.Serial(port, baud, timeout=2)
    readings = []
    buffer   = bytearray()
    print(f"  Collecting {n_samples} samples...", end='', flush=True)
    while len(readings) < n_samples:
        raw = ser.read(11)
        if not raw:
            continue
        buffer.extend(raw)
        while len(buffer) >= 11:
            if buffer[0] == 0x55 and buffer[1] == 0x54:
                # In read_mag_samples (calibrate) and read_mag_packet (classify)
                # replace whichever remap you currently have with:
                rx, ry, rz = struct.unpack('<hhh', bytes(buffer[2:8]))
                mx = rz    # world X
                my = rx    # world Y
                mz = ry    # world Z — confirmed: responds to magnet above sensor
                readings.append((mx, my, mz))
            if buffer[0] == 0x55:
                buffer = buffer[11:]
            else:
                buffer.pop(0)
    ser.close()
    arr  = np.array(readings[:n_samples], dtype=float)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    print(f" done.")
    print(f"    mean: X={mean[0]:+8.1f}  Y={mean[1]:+8.1f}  Z={mean[2]:+8.1f}")
    print(f"    std:  X={std[0]:7.2f}   Y={std[1]:7.2f}   Z={std[2]:7.2f}")
    return arr


def collect_state(label, instruction, baseline):
    print(f"\n--- {label} ---")
    print(f"  {instruction}.")
    input("  Hold steady, press Enter to measure...")
    samples   = read_mag_samples(PORT, BAUD)
    mean      = samples.mean(axis=0)
    corrected = mean - baseline
    print(f"  Corrected: Bx={corrected[0]:+8.1f}  By={corrected[1]:+8.1f}  Bz={corrected[2]:+8.1f}")
    return corrected, samples


def main():
    print("=== Multi-magnet array calibration ===")
    print()
    print("Magnet layout (top view):")
    print("  M1(+1)  M2(-1)")
    print("  M3(-1)  M4(+1)")
    print()
    print(f"Physical geometry (edit top of script if different):")
    print(f"  Rest height:       {REST_HEIGHT_MM} mm")
    print(f"  Press depth:       {PRESS_DEPTH_MM} mm")
    print(f"  Magnet diameter:   {MAGNET_DIAMETER_MM} mm")
    print(f"  Magnet thickness:  {MAGNET_THICKNESS_MM} mm")
    print()

    # ---- Baseline ----
    input("Ensure ALL magnets are fully unloaded. Press Enter to measure baseline...")
    baseline_samples = read_mag_samples(PORT, BAUD)
    baseline         = baseline_samples.mean(axis=0)
    # Noise std from unloaded samples — pure sensor noise, no mechanical contribution
    noise_std        = baseline_samples.std(axis=0)
    print(f"Baseline: Bx={baseline[0]:+8.1f}  By={baseline[1]:+8.1f}  Bz={baseline[2]:+8.1f}")
    print(f"Noise std (used by simulator): Bx={noise_std[0]:.2f}  By={noise_std[1]:.2f}  Bz={noise_std[2]:.2f}")

    centroids   = {}
    all_samples = []

    # ---- Unloaded centroid ----
    print("\n--- unloaded ---")
    print("  Keep everything unloaded (re-measuring for centroid).")
    input("  Press Enter to measure...")
    samples              = read_mag_samples(PORT, BAUD)
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

    # ---- Axis std ----
    centroid_matrix = np.array([centroids[l] for l in labels])
    axis_std        = centroid_matrix.std(axis=0)
    axis_std_safe   = np.where(axis_std < 1.0, 1.0, axis_std)

    print("\n=== Axis spread across centroids (axis_std) ===")
    print(f"  Bx std: {axis_std[0]:8.2f}")
    print(f"  By std: {axis_std[1]:8.2f}")
    print(f"  Bz std: {axis_std[2]:8.2f}")

    # ---- Inverse covariance ----
    all_arr = np.vstack(all_samples)
    cov_inv = None
    try:
        cov     = np.cov(all_arr.T)
        cov_inv = np.linalg.inv(cov)
        print("\n=== Sample covariance ===")
        print(np.array2string(cov, precision=1, suppress_small=True))
        print("Inverse covariance computed — Mahalanobis distance available.")
    except np.linalg.LinAlgError:
        print("\nWARNING: Covariance matrix singular — Mahalanobis not available.")

    # ---- Centroid summary ----
    print("\n=== Centroid summary ===")
    print(f"  {'Label':12s}  {'Bx':>10}  {'By':>10}  {'Bz':>10}")
    for lbl, c in centroids.items():
        print(f"  {lbl:12s}  {c[0]:>10.1f}  {c[1]:>10.1f}  {c[2]:>10.1f}")

    # ---- Pairwise separation ----
    print("\n=== Pairwise separation (Z-score normalised) ===")
    bad = False
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            diff = (centroids[l1] - centroids[l2]) / axis_std_safe
            d    = np.linalg.norm(diff)
            flag = "  <-- WARNING: too close!" if d < 1.5 else ""
            print(f"  {l1:12s} <-> {l2:12s}  norm_dist={d:6.3f}{flag}")
            if flag:
                bad = True
    if bad:
        print("\nWARNING: Some states poorly separated. Consider:")
        print("  - Increasing compression distance")
        print("  - Adjusting magnet positions or strength")
        print("  - Switching to Mahalanobis distance in the classifier")
    else:
        print("\nSeparation looks good.")

    # ---- Save ----
    save_dict = {
        # Classifier fields (unchanged)
        'baseline':             baseline,
        'centroids':            centroids,
        'labels':               labels,
        'axis_std':             axis_std_safe,
        # Simulator fields (new)
        'noise_std':            noise_std,
        'press_depth_mm':       PRESS_DEPTH_MM,
        'rest_height_mm':       REST_HEIGHT_MM,
        'magnet_diameter_mm':   MAGNET_DIAMETER_MM,
        'magnet_thickness_mm':  MAGNET_THICKNESS_MM,
        'magnet_layout':        MAGNET_LAYOUT,
    }
    if cov_inv is not None:
        save_dict['cov_inv'] = cov_inv

    np.save('multimagnet_cal_sim_data.npy', save_dict)
    print("\nSaved to multimagnet_cal_sim_data.npy")
    print(f"  noise_std:    Bx={noise_std[0]:.2f}  By={noise_std[1]:.2f}  Bz={noise_std[2]:.2f}")
    print(f"  press_depth:  {PRESS_DEPTH_MM} mm")
    print(f"  rest_height:  {REST_HEIGHT_MM} mm")
    print()
    print("Run extract_sim_params.py to generate sim_params.npy for the simulator.")


if __name__ == '__main__':
    main()