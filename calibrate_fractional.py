# calibrate_in_structure.py
import serial
import struct
import numpy as np
from scipy.optimize import curve_fit
import magpylib as magpy

PORT = '/dev/ttyUSB0'
BAUD = 9600

Z_NOMINAL = 0.011       # resting gap magnet-to-sensor in meters (11mm)
FULL_COMPRESSION = 0.005 # maximum compression travel in meters — MEASURE THIS

# Calibration points as fractions of full compression
COMPRESSION_STEPS = {
    'unloaded (0%)':    0.0,
#    'quarter (25%)':    0.25,
    'half (50%)':       0.50,
#    'three-quarter (75%)': 0.75,
    'full (100%)':      1.0,
}

magnet = magpy.magnet.CylinderSegment(
    polarization=(0, 0, 1.0),
    dimension=(0.0015, 0.00475, 0.003, 0, 360),
    position=(0, 0, 0),
)
sensor = magpy.Sensor(position=(0, 0, 0))


def predicted_field(z, scale):
    z = np.atleast_1d(z)
    results = []
    for zi in z:
        sensor.position = (0.0, 0.0, float(-zi))
        B = magpy.getB(magnet, sensor)
        results.append(B[2] * scale)
    return np.array(results) if len(results) > 1 else results[0]


def read_mag_average(port, baud, n_samples=50):
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
    print(f" done. Mean: X={mean[0]:.1f}  Y={mean[1]:.1f}  Z={mean[2]:.1f}")
    return mean


def main():
    print("=== Magpylib Calibration (Ring Magnet in Structure) ===")
    print(f"Nominal resting gap:  {Z_NOMINAL*1000:.1f} mm")
    print(f"Full compression:     {FULL_COMPRESSION*1000:.1f} mm")
    print()
    print("Calibration steps:")
    for label, frac in COMPRESSION_STEPS.items():
        z_abs = Z_NOMINAL - frac * FULL_COMPRESSION
        print(f"  {label:30s} → compress {frac*FULL_COMPRESSION*1000:.1f}mm  (gap = {z_abs*1000:.1f}mm)")
    print()

    input("Remove all load from structure (fully unloaded). Press Enter to measure baseline...")
    baseline = read_mag_average(PORT, BAUD)
    print(f"Baseline (unloaded): {baseline}\n")

    known_z = []
    measured_bz = []

    for label, frac in COMPRESSION_STEPS.items():
        if frac == 0.0:
            # unloaded is already the baseline, but still record it as a data point
            compression_mm = 0.0
            z_abs = Z_NOMINAL
        else:
            compression_mm = frac * FULL_COMPRESSION * 1000
            z_abs = Z_NOMINAL - frac * FULL_COMPRESSION

        print(f"--- {label} ---")
        if frac == 0.0:
            input(f"  Structure unloaded (gap={z_abs*1000:.1f}mm). Press Enter to measure...")
        else:
            print(f"  Press structure down by {compression_mm:.1f}mm (gap should be {z_abs*1000:.1f}mm).")
            input(f"  Hold steady, press Enter to measure...")

        reading = read_mag_average(PORT, BAUD)
        corrected = reading - baseline
        print(f"  Corrected: Bx={corrected[0]:.1f}  By={corrected[1]:.1f}  Bz={corrected[2]:.1f}\n")
        known_z.append(z_abs)
        measured_bz.append(corrected[2])

    known_z = np.array(known_z)
    measured_bz = np.array(measured_bz)

    print("Fitting scale factor...")
    try:
        popt, pcov = curve_fit(predicted_field, known_z, measured_bz, p0=[1.0])
        scale = popt[0]
        scale_err = np.sqrt(pcov[0, 0])
    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return

    print(f"Scale factor = {scale:.6e}  ±  {scale_err:.2e}")

    print("\nFit quality:")
    max_residual = 0
    for label, frac, z, meas in zip(
        COMPRESSION_STEPS.keys(), COMPRESSION_STEPS.values(), known_z, measured_bz
    ):
        pred = predicted_field(z, scale)
        residual = meas - pred
        max_residual = max(max_residual, abs(residual))
        print(f"  {label:30s} z={z*1000:.1f}mm  measured={meas:.1f}  predicted={pred:.1f}  residual={residual:.1f}")

    print(f"\nMax residual: {max_residual:.1f}")
    if max_residual > 5000:
        print("WARNING: Large residuals — magnet may be tilting or FULL_COMPRESSION value needs adjustment.")
    else:
        print("Fit looks good.")

    np.save('calibration.npy', {
        'scale': scale,
        'baseline': baseline,
        'nominal_z': Z_NOMINAL,
        'full_compression': FULL_COMPRESSION,
    })
    print(f"\nSaved to calibration.npy")


if __name__ == '__main__':
    main()
