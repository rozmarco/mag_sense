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
#   baseline       : (3,) float array
#   centroids      : dict {label: (3,) corrected centroid}
#   labels         : list of label strings in same order as centroids

import serial
import struct
import numpy as np

PORT = '/dev/ttyUSB0'
BAUD = 9600
N_SAMPLES = 80   # samples averaged per state

# ---------- States to calibrate ----------
# Each entry: (label, human instruction)
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


def read_mag_average(port, baud, n_samples=N_SAMPLES):
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
    arr = np.array(readings, dtype=float)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    print(f" done.")
    print(f"    mean: X={mean[0]:+8.1f}  Y={mean[1]:+8.1f}  Z={mean[2]:+8.1f}")
    print(f"    std:  X={std[0]:7.1f}   Y={std[1]:7.1f}   Z={std[2]:7.1f}")
    return mean


def collect_state(label, instruction, baseline):
    print(f"\n--- {label} ---")
    print(f"  {instruction}.")
    input("  Hold steady, press Enter to measure...")
    raw = read_mag_average(PORT, BAUD)
    corrected = raw - baseline
    print(f"  Corrected: Bx={corrected[0]:+8.1f}  By={corrected[1]:+8.1f}  Bz={corrected[2]:+8.1f}")
    return corrected


def main():
    print("=== Multi-magnet array calibration ===")
    print()
    print("Magnet layout (top view):")
    print("  M1(+1)  M2(-1)")
    print("  M3(-1)  M4(+1)")
    print()

    # Baseline
    input("Ensure ALL magnets are fully unloaded. Press Enter to measure baseline...")
    baseline = read_mag_average(PORT, BAUD)
    print(f"Baseline: Bx={baseline[0]:+8.1f}  By={baseline[1]:+8.1f}  Bz={baseline[2]:+8.1f}")

    centroids = {}

    # Always add an explicit "unloaded" centroid (near zero by definition)
    print("\n--- unloaded ---")
    print("  Keep everything unloaded (re-measuring for centroid).")
    input("  Press Enter to measure...")
    raw_unloaded = read_mag_average(PORT, BAUD)
    centroids['unloaded'] = raw_unloaded - baseline

    # Singles
    for label, instruction in SINGLE_STATES:
        centroids[label] = collect_state(label, instruction, baseline)

    # Pairs (optional)
    do_pairs = input("\nCalibrate pair presses too? (y/n): ").strip().lower() == 'y'
    if do_pairs:
        for label, instruction in PAIR_STATES:
            centroids[label] = collect_state(label, instruction, baseline)

    # Print summary
    print("\n=== Centroid summary ===")
    labels = list(centroids.keys())
    print(f"  {'Label':12s}  {'Bx':>10}  {'By':>10}  {'Bz':>10}")
    for lbl, c in centroids.items():
        print(f"  {lbl:12s}  {c[0]:>10.1f}  {c[1]:>10.1f}  {c[2]:>10.1f}")

    # Separation check — warn if any two centroids are too close
    print("\n=== Pairwise separation (Euclidean distance) ===")
    bad = False
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            d = np.linalg.norm(centroids[l1] - centroids[l2])
            flag = "  <-- WARNING: too close!" if d < 500 else ""
            print(f"  {l1:12s} <-> {l2:12s}  dist={d:8.1f}{flag}")
            if flag:
                bad = True
    if bad:
        print("\nWARNING: Some states are poorly separated. Consider:")
        print("  - Increasing compression distance")
        print("  - Adjusting magnet positions or strength")
    else:
        print("\nSeparation looks good.")

    np.save('multimagnet_cal.npy', {
        'baseline':  baseline,
        'centroids': centroids,
        'labels':    labels,
    })
    print("\nSaved to multimagnet_cal.npy")


if __name__ == '__main__':
    main()
