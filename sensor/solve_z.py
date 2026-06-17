# solve_z.py
import serial
import struct
import numpy as np
from scipy.optimize import minimize_scalar
import magpylib as magpy

PORT = '/dev/ttyUSB0'
BAUD = 9600

# Load calibration
cal = np.load('calibration.npy', allow_pickle=True).item()
SCALE = cal['scale']
BASELINE = cal['baseline']
print(f"Scale: {SCALE:.6e}")
print(f"Baseline: {BASELINE}")

# Fixed magnet x, y offset from sensor (meters)
MAGNET_X = 0.0
MAGNET_Y = 0.0

# Match dimension to your actual magnet AND to calibrate_with_magpy.py
magnet = magpy.magnet.Cylinder(
    polarization=(0, 0, 1.0),
    dimension=(0.0095, 0.003),  # (diameter, height) in meters
    position=(MAGNET_X, MAGNET_Y, 0),
)
sensor = magpy.Sensor()

Z_NOMINAL = 0.011  # set on first reading; or hardcode e.g. Z_NOMINAL = 0.010


def predicted_field_3axis(z):
    """Predict all 3 components at sensor position (0, 0, -z) from magnet."""
    sensor.position = (0.0, 0.0, float(-z))
    B = magpy.getB(magnet, sensor)
    return B * SCALE


def residual(z, measured_corrected):
    pred = predicted_field_3axis(z)
    return np.sum((pred - measured_corrected) ** 2)


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


def main():
    global Z_NOMINAL

    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("\nReading... (Ctrl+C to stop)")
    print("Waiting for first reading to set nominal position...\n")

    header = (
        f"{'Meas Bx':>10} {'Meas By':>10} {'Meas Bz':>10}  |  "
        f"{'z (mm)':>8}  |  {'Displ (mm)':>10}  |  "
        f"{'Pred Bx':>10} {'Pred By':>10} {'Pred Bz':>10}"
    )
    print(header)
    print("-" * len(header))

    buffer = bytearray()

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

            result = minimize_scalar(
                residual,
                bounds=(0.001, 0.5),
                method='bounded',
                args=(corrected,)
            )
            z_solved = result.x

            # Set nominal on first successful solve
            if Z_NOMINAL is None:
                Z_NOMINAL = z_solved
                print(f"Nominal z set: {Z_NOMINAL * 1000:.2f} mm\n")

            displacement_mm = (z_solved - Z_NOMINAL) * 1000
            pred = predicted_field_3axis(z_solved)

            print(
                f"{corrected[0]:>10.1f} {corrected[1]:>10.1f} {corrected[2]:>10.1f}  |  "
                f"{z_solved * 1000:>8.2f}  |  "
                f"{displacement_mm:>+10.3f}  |  "
                f"{pred[0]:>10.1f} {pred[1]:>10.1f} {pred[2]:>10.1f}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
