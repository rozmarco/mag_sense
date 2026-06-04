# calibrate.py
import serial
import struct
import numpy as np
from scipy.optimize import curve_fit
import magpylib as magpy

PORT = '/dev/ttyUSB0'
BAUD = 9600

# --- Define your magnet ---
# Adjust dimension and polarization to match your actual magnet.
# dimension=(diameter, height) in meters for a cylinder
# polarization=(0, 0, Br) for vertical orientation — Br in Tesla
# Start with Br=1.0 and let the calibration scale it via the fit
magnet = magpy.magnet.Cylinder(
    polarization=(0, 0, 1.0),
    dimension=(0.0095, 0.003),  # e.g. 10mm diameter, 5mm tall — adjust to your magnet
    position=(0, 0, 0),       # magnet at origin
)

sensor = magpy.Sensor(position=(0, 0, 0))  # sensor position will be set during prediction

def predicted_field(z, scale):
    """
    Predict B field at (0, 0, -z) below the magnet, scaled by `scale`.
    Returns Bz only (for calibration along the axis).
    z may be a scalar or a numpy array (scipy passes arrays during fitting).
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
    print("=== Magpylib Calibration ===")
    print("First, measure baseline (no magnet nearby).")
    input("Remove magnet from area, then press Enter...")
    baseline = read_mag_average(PORT, BAUD)
    print(f"Baseline: {baseline}\n")

    known_z = []
    measured_bz = []

    while True:
        z_input = input("Enter z distance in meters (or 'done'): ").strip()
        if z_input.lower() == 'done':
            break
        try:
            z = float(z_input)
        except ValueError:
            print("  Please enter a number.")
            continue
        input(f"  Place magnet at z={z}m, press Enter to measure...")
        reading = read_mag_average(PORT, BAUD)
        corrected = reading - baseline
        print(f"  Corrected: {corrected}")
        known_z.append(z)
        measured_bz.append(corrected[2])

    known_z = np.array(known_z)
    measured_bz = np.array(measured_bz)

    print("\nFitting scale factor...")
    popt, pcov = curve_fit(predicted_field, known_z, measured_bz, p0=[1.0])
    scale = popt[0]
    scale_err = np.sqrt(pcov[0, 0])
    print(f"Scale factor = {scale:.6e}  ±  {scale_err:.2e}")

    # Show fit quality
    print("\nFit quality:")
    for z, meas in zip(known_z, measured_bz):
        pred = predicted_field(z, scale)
        print(f"  z={z:.3f}m  measured={meas:.2f}  predicted={pred:.2f}  residual={meas-pred:.2f}")

    np.save('calibration.npy', {'scale': scale, 'baseline': baseline})
    print("\nSaved to calibration.npy")

if __name__ == '__main__':
    main()
