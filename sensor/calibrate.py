# calibrate.py
# Place the magnet directly above the sensor at known z distances
# and record the magnetometer readings to fit the dipole moment m

import serial
import struct
import numpy as np
from scipy.optimize import curve_fit

PORT = '/dev/ttyUSB0'
BAUD = 9600

def read_mag_average(port, baud, n_samples=50):
    """Read n_samples magnetometer packets and return the average."""
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

def bz_dipole(z, m):
    """
    Predicted Bz directly above a vertical dipole (x=0, y=0).
    B_z = (mu0 / 4pi) * (2m / z^3)
    We absorb (mu0 / 4pi) into m during fitting, so:
    B_z = 2m / z^3
    """
    return 2 * m / z**3

def main():
    print("=== Dipole Moment Calibration ===")
    print("Place the magnet directly above the sensor (x=0, y=0)")
    print("at each z distance when prompted.\n")

    known_z = []
    measured_bz = []

    while True:
        z_input = input("Enter z distance in meters (or 'done' to fit): ").strip()
        if z_input.lower() == 'done':
            break
        try:
            z = float(z_input)
        except ValueError:
            print("  Please enter a number.")
            continue

        input(f"  Position magnet at z={z}m, then press Enter to measure...")
        reading = read_mag_average(PORT, BAUD)

        # Subtract baseline? For now use raw Bz
        known_z.append(z)
        measured_bz.append(reading[2])  # Bz component

    known_z = np.array(known_z)
    measured_bz = np.array(measured_bz)

    print("\nFitting dipole moment...")
    popt, pcov = curve_fit(bz_dipole, known_z, measured_bz, p0=[1e-4])
    m_fit = popt[0]
    m_err = np.sqrt(pcov[0, 0])

    print(f"\n=== Result ===")
    print(f"Fitted dipole moment m = {m_fit:.6e}  ±  {m_err:.2e}")
    print(f"\nSave this value — you'll need it for the solver.")

    # Show fit quality
    predicted = bz_dipole(known_z, m_fit)
    for z, meas, pred in zip(known_z, measured_bz, predicted):
        print(f"  z={z:.3f}m  measured={meas:.1f}  predicted={pred:.1f}  residual={meas-pred:.1f}")

    # Save to file
    np.save('calibration.npy', {'m': m_fit})
    print("\nSaved to calibration.npy")

if __name__ == '__main__':
    main()
