# classify_multimagnet.py
#
# Live nearest-centroid classifier for the 2x2 magnet array.
# Loads multimagnet_cal.npy produced by calibrate_multimagnet.py.
#
# Noise handling:
#   - Exponential moving average (EMA) smooths the raw signal
#   - Label debounce only emits a new state after CONFIRM consecutive frames agree

import serial
import struct
import numpy as np

PORT = '/dev/ttyUSB0'
BAUD = 9600

# ---- Smoothing & debounce ----
ALPHA   = 0.7   # EMA weight on new sample (0.1 = very smooth/slow, 0.4 = responsive/noisier)
CONFIRM = 3     # frames the same label must appear before it's accepted

# Confidence threshold: if best distance exceeds this, report "unknown"
DIST_THRESHOLD = 50000   # tune after watching live dist_best values

# ---- Load calibration ----
cal       = np.load('multimagnet_cal.npy', allow_pickle=True).item()
BASELINE  = cal['baseline']
centroids = cal['centroids']   # dict {label: (3,) array}
labels    = cal['labels']

centroid_matrix = np.array([centroids[l] for l in labels])  # (N, 3)

print("Loaded calibration:")
print(f"  States: {labels}")
print(f"  EMA alpha: {ALPHA}  |  Debounce frames: {CONFIRM}")
print()


def read_mag_packet(buffer):
    while len(buffer) >= 11:
        if buffer[0] == 0x55 and buffer[1] == 0x54:
            mx, my, mz = struct.unpack('<hhh', bytes(buffer[2:8]))
            return np.array([mx, my, mz], dtype=float), buffer[11:]
        elif buffer[0] == 0x55:
            return None, buffer[11:]
        else:
            buffer.pop(0)
    return None, buffer


def classify(corrected):
    dists      = np.linalg.norm(centroid_matrix - corrected, axis=1)
    sorted_idx = np.argsort(dists)
    best_dist  = dists[sorted_idx[0]]
    margin     = dists[sorted_idx[1]] - best_dist

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
                f"{dist:>8.1f}  {margin:>8.1f}  |  "
                f" {confirmed:>14}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()