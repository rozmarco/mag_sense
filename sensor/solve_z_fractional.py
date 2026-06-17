# solve_z.py
import serial
import struct
import numpy as np
from scipy.optimize import minimize_scalar
import magpylib as magpy

PORT = '/dev/ttyUSB0'
BAUD = 9600

cal = np.load('calibration.npy', allow_pickle=True).item()
SCALE           = cal['scale']
BASELINE        = cal['baseline']
Z_NOMINAL       = cal.get('nominal_z', 0.011)
FULL_COMPRESSION = cal.get('full_compression', 0.008)

print(f"Scale:            {SCALE:.6e}")
print(f"Baseline:         {BASELINE}")
print(f"Nominal z:        {Z_NOMINAL*1000:.2f} mm")
print(f"Full compression: {FULL_COMPRESSION*1000:.2f} mm")

magnet = magpy.magnet.CylinderSegment(
    polarization=(0, 0, 1.0),
    dimension=(0.0015, 0.00475, 0.003, 0, 360),
    position=(0, 0, 0),
)
sensor = magpy.Sensor()


def predicted_field_3axis(z):
    sensor.position = (0.0, 0.0, float(-z))
    B = magpy.getB(magnet, sensor)
    return B * SCALE


def residual(z, measured_corrected):
    pred = predicted_field_3axis(z)
    return np.sum((pred - measured_corrected) ** 2)


def z_to_compression_fraction(z):
    """Convert absolute z gap to compression as fraction of full compression."""
    return (Z_NOMINAL - z) / FULL_COMPRESSION


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


def compression_label(frac):
    """Return a human-readable compression label."""
    if frac <= 0.05:
        return "unloaded"
    elif frac <= 0.375:
        return "~quarter"
    elif frac <= 0.625:
        return "~half"
    elif frac <= 0.875:
        return "~three-quarter"
    else:
        return "~full"


def main():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    print("\nReading... (Ctrl+C to stop)\n")

    header = (
        f"{'Meas Bz':>10}  |  "
        f"{'z (mm)':>8}  |  "
        f"{'Displ (mm)':>10}  |  "
        f"{'Compression':>12}  |  "
        f"{'State':>14}"
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
                bounds=(Z_NOMINAL - FULL_COMPRESSION * 1.1, Z_NOMINAL + 0.002),
                method='bounded',
                args=(corrected,)
            )
            z_solved = result.x
            displacement_mm = (z_solved - Z_NOMINAL) * 1000  # negative = compressed
            frac = z_to_compression_fraction(z_solved)
            frac_pct = np.clip(frac * 100, 0, 100)
            label = compression_label(frac)

            print(
                f"{corrected[2]:>10.1f}  |  "
                f"{z_solved*1000:>8.2f}  |  "
                f"{displacement_mm:>+10.3f}  |  "
                f"{frac_pct:>11.1f}%  |  "
                f"{label:>14}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()
