# calibrate_in_structure.py
import serial
import struct
import numpy as np
from scipy.optimize import curve_fit
import magpylib as magpy

PORT = '/dev/ttyUSB0'
BAUD = 9600

Z_OFFSET = 0.0125
Y_OFFSET = -0.003
X_NOMINAL = 0.026
FULL_DISPLACEMENT = 0.008

DISPLACEMENT_STEPS = {
    'unloaded (0%)':      0.00,
    'quarter (25%)':      0.25,
    'half (50%)':         0.50,
    'three-quarter (75%)':0.75,
    'full (100%)':        1.00,
}

magnet = magpy.magnet.Cylinder(
    polarization=(0, 0, 1.17),   # N35 remanence ~1.17 T
    dimension=(0.008, 0.002),    # (diameter=8mm, height=2mm)
    position=(0, 0, 0),
)

# Remapped geometry (matches solve_x.py):
#   Sensor Z → toward magnet  = magpylib X axis  (standoff = Z_OFFSET)
#   Magnet slides along sensor X = magpylib Z axis (nominal = X_NOMINAL)
#   Y offset unchanged         = magpylib Y axis
sensor = magpy.Sensor(position=(Z_OFFSET, Y_OFFSET, 0.0))


def predicted_field(x, scale):
    """
    Return baseline-corrected predicted Bz.

    The measured values are corrected by subtracting the unloaded baseline,
    so the model predictions must also be referenced to the nominal position.
    """
    x = np.atleast_1d(x)

    # Field at unloaded position
    magnet.position = (0.0, 0.0, X_NOMINAL)
    B0 = magpy.getB(magnet, sensor)[2]

    results = []

    for xi in x:
        magnet.position = (0.0, 0.0, float(xi))
        B = magpy.getB(magnet, sensor)[2]

        corrected_B = (B - B0) * scale
        results.append(corrected_B)

    results = np.array(results)

    if len(results) == 1:
        return results[0]

    return results

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


# def main():
#     print("=== Magpylib Calibration (X-axis magnet displacement) ===")
#     print(f"Fixed Z offset (magnet-to-sensor): {Z_OFFSET*1000:.1f} mm")
#     print(f"Fixed Y offset (magnet-to-sensor): {Y_OFFSET*1000:.1f} mm")
#     print(f"Nominal resting X position:        {X_NOMINAL*1000:.1f} mm")
#     print(f"Full X displacement:               {FULL_DISPLACEMENT*1000:.1f} mm")
#     print()
#     print("Calibration steps:")
#     for label, frac in DISPLACEMENT_STEPS.items():
#         x_abs = X_NOMINAL - frac * FULL_DISPLACEMENT
#         print(f"  {label:30s} → move magnet -{frac*FULL_DISPLACEMENT*1000:.1f}mm along X  (x = {x_abs*1000:.1f}mm)")
#     print()

#     input(
#         f"Return magnet to resting position "
#         f"(x = {X_NOMINAL*1000:.1f} mm). "
#         f"Press Enter to measure baseline..."
#     )
#     baseline = read_mag_average(PORT, BAUD)
#     print(f"Baseline (unloaded): {baseline}\n")

#     known_x = []
#     measured_bz = []

#     for label, frac in DISPLACEMENT_STEPS.items():
#         x_abs = X_NOMINAL - frac * FULL_DISPLACEMENT
#         displacement_mm = frac * FULL_DISPLACEMENT * 1000

#         print(f"--- {label} ---")
#         if frac == 0.0:
#             input(f"  Magnet at rest (x = {x_abs*1000:.1f}mm). Press Enter to measure...")
#         else:
#             print(f"  Move magnet {displacement_mm:.1f}mm along X (x = {x_abs*1000:.1f}mm).")
#             input(f"  Hold steady, press Enter to measure...")

#         reading = read_mag_average(PORT, BAUD)
#         corrected = reading - baseline
#         print(f"  Corrected: Bx={corrected[0]:.1f}  By={corrected[1]:.1f}  Bz={corrected[2]:.1f}\n")
#         known_x.append(x_abs)
#         measured_bz.append(corrected[2])

#     known_x = np.array(known_x)
#     measured_bz = np.array(measured_bz)
#     print("\nUnscaled model comparison:")

#     for label, x, meas in zip(
#         DISPLACEMENT_STEPS.keys(),
#         known_x,
#         measured_bz,
#     ):
#         pred = predicted_field(x, 1.0)

#         print(
#             f"{label:30s}"
#             f" measured={meas:8.1f}"
#             f" predicted={pred:12.3e}"
#         )

#     print("Fitting scale factor...")
#     try:
#         popt, pcov = curve_fit(predicted_field, known_x, measured_bz, p0=[1.0])
#         scale = popt[0]
#         scale_err = np.sqrt(pcov[0, 0])
#     except RuntimeError as e:
#         print(f"Fit failed: {e}")
#         return

#     print(f"Scale factor = {scale:.6e}  ±  {scale_err:.2e}")

#     print("\nFit quality:")
#     max_residual = 0
#     for label, frac, x, meas in zip(
#         DISPLACEMENT_STEPS.keys(), DISPLACEMENT_STEPS.values(), known_x, measured_bz
#     ):
#         pred = predicted_field(x, scale)
#         residual = meas - pred
#         max_residual = max(max_residual, abs(residual))
#         print(f"  {label:30s} x={x*1000:.1f}mm  measured={meas:.1f}  predicted={pred:.1f}  residual={residual:.1f}")

#     print(f"\nMax residual: {max_residual:.1f}")
#     if max_residual > 5000:
#         print("WARNING: Large residuals — magnet may be tilting or FULL_DISPLACEMENT value needs adjustment.")
#     else:
#         print("Fit looks good.")

#     np.save('calibration.npy', {
#         'scale': scale,
#         'baseline': baseline,
#         'z_offset': Z_OFFSET,
#         'y_offset': Y_OFFSET,
#         'nominal_x': X_NOMINAL,
#         'full_displacement': FULL_DISPLACEMENT,
#         'known_x': known_x,
#         'measured_bz': measured_bz,
#     })
#     print(f"\nSaved to calibration.npy")


def main():
    print("=== Magnetic Field Survey ===")
    print(f"Nominal X position: {X_NOMINAL*1000:.1f} mm")
    print("Measure the magnet at the following displacements:")
    print()

    displacements_mm = np.arange(0, 9, 1)

    input(
        f"Return magnet to resting position "
        f"(x = {X_NOMINAL*1000:.1f} mm). "
        f"Press Enter to measure baseline..."
    )

    baseline = read_mag_average(PORT, BAUD)

    print("\nBaseline:")
    print(
        f"  X={baseline[0]:.1f}  "
        f"Y={baseline[1]:.1f}  "
        f"Z={baseline[2]:.1f}"
    )

    results = []

    for displacement_mm in displacements_mm:

        x_abs = X_NOMINAL - displacement_mm / 1000.0

        print()
        print(
            f"=== {displacement_mm} mm displacement "
            f"(x = {x_abs*1000:.1f} mm) ==="
        )

        if displacement_mm == 0:
            input("Magnet at rest. Press Enter to measure...")
        else:
            input(
                f"Move magnet {displacement_mm:.1f} mm "
                f"toward the sensor. Press Enter to measure..."
            )

        reading = read_mag_average(PORT, BAUD)

        corrected = reading - baseline

        print(
            f"Corrected:\n"
            f"  Bx = {corrected[0]:8.1f}\n"
            f"  By = {corrected[1]:8.1f}\n"
            f"  Bz = {corrected[2]:8.1f}"
        )

        results.append([
            displacement_mm,
            corrected[0],
            corrected[1],
            corrected[2],
        ])

    results = np.array(results)

    print("\nSummary:")
    print(
        f"{'Disp (mm)':>10}  "
        f"{'Bx':>10}  "
        f"{'By':>10}  "
        f"{'Bz':>10}"
    )

    print("-" * 48)

    for row in results:
        print(
            f"{row[0]:10.0f}  "
            f"{row[1]:10.1f}  "
            f"{row[2]:10.1f}  "
            f"{row[3]:10.1f}"
        )

    np.save(
        "field_survey.npy",
        {
            "baseline": baseline,
            "displacement_mm": results[:, 0],
            "bx": results[:, 1],
            "by": results[:, 2],
            "bz": results[:, 3],
        },
    )

    print("\nSaved survey data to field_survey.npy")


if __name__ == '__main__':
    main()