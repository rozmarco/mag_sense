# calibrate_in_structure.py
import serial
import struct
import numpy as np
from scipy.optimize import curve_fit
import magpylib as magpy

PORT = '/dev/ttyUSB0'
BAUD = 9600

Z_NOMINAL = 0.011  # 11mm resting height in meters

magnet = magpy.magnet.CylinderSegment(
    polarization=(0, 0, 1.0),
    dimension=(0.0015, 0.00475, 0.003, 0, 360),
    position=(0, 0, 0),
)
sensor = magpy.Sensor(position=(0, 0, 0))


def predicted_field(z, scale):
    """
    Predict Bz at sensor (0,0,-z). z is absolute gap in meters.
    Handles scalar or array input from scipy.
    """
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
    print()
    print("IMPORTANT: Magnet must be IN the structure for all measurements.")
    print(f"Nominal resting z: {Z_NOMINAL*1000:.1f}mm")
    print()
    print("For each measurement:")
    print("  1. Press the structure to some position and hold it")
    print("  2. Use calipers to measure the gap from magnet bottom to sensor top")
    print("  3. Enter that distance, then take the reading")
    print()

    input("Step 1: Fully unloaded structure. Press Enter to measure baseline...")
    baseline = read_mag_average(PORT, BAUD)
    print(f"Baseline (unloaded): {baseline}\n")

    known_z = []
    measured_bz = []

    print("Now collect readings at different compression positions.")
    print("Enter the ABSOLUTE gap (magnet-to-sensor) in meters, or 'done'.\n")

    while True:
        z_input = input("Enter absolute gap in meters (e.g. 0.009 for 9mm), or 'done': ").strip()
        if z_input.lower() == 'done':
            break
        try:
            z = float(z_input)
        except ValueError:
            print("  Please enter a number.")
            continue
        if z <= 0 or z > 0.05:
            print("  Please enter a value between 0 and 0.05m (0–50mm).")
            continue

        input(f"  Hold structure at z={z*1000:.1f}mm gap, press Enter to measure...")
        reading = read_mag_average(PORT, BAUD)
        corrected = reading - baseline
        print(f"  Corrected: Bx={corrected[0]:.1f}  By={corrected[1]:.1f}  Bz={corrected[2]:.1f}\n")
        known_z.append(z)
        measured_bz.append(corrected[2])

    if len(known_z) < 2:
        print("Need at least 2 measurements to fit. Exiting.")
        return

    known_z = np.array(known_z)
    measured_bz = np.array(measured_bz)

    # Sort by z for cleaner output
    sort_idx = np.argsort(known_z)
    known_z = known_z[sort_idx]
    measured_bz = measured_bz[sort_idx]

    print("Fitting scale factor...")
    try:
        popt, pcov = curve_fit(predicted_field, known_z, measured_bz, p0=[1.0])
        scale = popt[0]
        scale_err = np.sqrt(pcov[0, 0])
    except RuntimeError as e:
        print(f"Fit failed: {e}")
        return

    print(f"Scale factor = {scale:.6e}  ±  {scale_err:.2e}")

    print("\nFit quality (sorted by z):")
    max_residual = 0
    for z, meas in zip(known_z, measured_bz):
        pred = predicted_field(z, scale)
        residual = meas - pred
        max_residual = max(max_residual, abs(residual))
        print(f"  z={z*1000:.1f}mm  measured Bz={meas:.1f}  predicted Bz={pred:.1f}  residual={residual:.1f}")

    print(f"\nMax residual: {max_residual:.1f}")
    if max_residual > 5000:
        print("WARNING: Large residuals — magnet may be tilting, or more measurements needed.")
    else:
        print("Fit looks good.")

    np.save('calibration.npy', {
        'scale': scale,
        'baseline': baseline,
        'nominal_z': Z_NOMINAL,
    })
    print(f"\nSaved to calibration.npy  (nominal_z={Z_NOMINAL*1000:.1f}mm)")


if __name__ == '__main__':
    main()
