# solve_x.py
import serial
import struct
import numpy as np
from scipy.optimize import minimize_scalar
import magpylib as magpy

PORT = '/dev/ttyUSB0'
BAUD = 9600

cal = np.load('calibration.npy', allow_pickle=True).item()
SCALE             = cal['scale']
BASELINE          = cal['baseline']
Z_OFFSET          = cal.get('z_offset', 0.014)
Y_OFFSET          = cal.get('y_offset', 0.003)
X_NOMINAL         = cal.get('nominal_x', 0.026)
FULL_DISPLACEMENT = cal.get('full_displacement', 0.005)

print(f"Scale:             {SCALE:.6e}")
print(f"Baseline:          {BASELINE}")
print(f"Z offset:          {Z_OFFSET*1000:.2f} mm")
print(f"Y offset:          {Y_OFFSET*1000:.2f} mm")
print(f"Nominal X:         {X_NOMINAL*1000:.2f} mm")
print(f"Full displacement: {FULL_DISPLACEMENT*1000:.2f} mm")

magnet = magpy.magnet.CylinderSegment(
    polarization=(0, 0, 1.0),
    dimension=(0.0015, 0.00475, 0.003, 0, 360),
    position=(0, 0, 0),
)

# Remapped geometry to match physical sensor frame:
#   Sensor Z points toward magnet  → magpylib X axis  (standoff = Z_OFFSET)
#   Magnet slides along sensor X   → magpylib Z axis  (nominal = X_NOMINAL)
#   Y offset is unchanged          → magpylib Y axis
#
# Sensor is fixed; magnet moves along magpylib Z.
# Sensor position in magpylib frame: (Z_OFFSET, Y_OFFSET, 0)
sensor = magpy.Sensor(position=(Z_OFFSET, Y_OFFSET, 0.0))


def predicted_field_bz(x):
    """Return scaled Bz at sensor for magnet at sensor-X position x.
    
    In magpylib frame the magnet sits at (0, 0, x) — i.e. displaced along
    magpylib Z, which corresponds to the physical sensor X axis.
    Returns only the B[2] (magpylib Z) component, which corresponds to
    physical sensor Z — the dominant varying axis confirmed in calibration.
    """
    magnet.position = (0.0, 0.0, float(x))
    B = magpy.getB(magnet, sensor)
    return B[2] * SCALE


def residual(x, measured_bz):
    pred = predicted_field_bz(x)
    return (pred - measured_bz) ** 2


def x_to_displacement_fraction(x):
    """Convert absolute X position to displacement fraction (0 = rest, 1 = full travel)."""
    return (X_NOMINAL - x) / FULL_DISPLACEMENT


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


def displacement_label(frac):
    if frac <= 0.05:
        return "at rest"
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
        f"{'x (mm)':>8}  |  "
        f"{'Displ (mm)':>10}  |  "
        f"{'Travel':>10}  |  "
        f"{'State':>14}"
    )
    print(header)
    print("-" * len(header))

    buffer = bytearray()

    # Magnet travels from X_NOMINAL toward smaller X; allow 10% slop each side
    X_LO = X_NOMINAL - FULL_DISPLACEMENT * 1.1
    X_HI = X_NOMINAL + FULL_DISPLACEMENT * 0.1

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
                bounds=(X_LO, X_HI),
                method='bounded',
                args=(corrected[2],)   # fit against physical Bz only
            )
            x_solved = result.x
            displacement_mm = (x_solved - X_NOMINAL) * 1000
            frac = x_to_displacement_fraction(x_solved)
            frac_pct = np.clip(frac * 100, 0, 100)
            label = displacement_label(frac)

            print(
                f"{corrected[2]:>10.1f}  |  "
                f"{x_solved*1000:>8.2f}  |  "
                f"{displacement_mm:>+10.3f}  |  "
                f"{frac_pct:>9.1f}%  |  "
                f"{label:>14}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        ser.close()


if __name__ == '__main__':
    main()