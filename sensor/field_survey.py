```python
# field_survey.py
#
# Measure the raw magnetic field response of the sensor as the magnet
# is displaced from its resting position.
#
# Saves:
#   displacement_mm
#   Bx
#   By
#   Bz
#
# relative to a baseline measurement at rest.

import serial
import struct
import numpy as np

PORT = "/dev/ttyUSB0"
BAUD = 9600

X_NOMINAL = 26.0  # mm

def read_mag_average(port, baud, n_samples=50):
    """
    Collect multiple samples and return average field.
    """

    ser = serial.Serial(port, baud, timeout=2)

    readings = []
    buffer = bytearray()

    print(f"Collecting {n_samples} samples...", end="", flush=True)

    while len(readings) < n_samples:

        raw = ser.read(11)

        if not raw:
            continue

        buffer.extend(raw)

        while len(buffer) >= 11:

            if buffer[0] == 0x55 and buffer[1] == 0x54:

                mx, my, mz = struct.unpack(
                    "<hhh",
                    bytes(buffer[2:8])
                )

                readings.append((mx, my, mz))

            if buffer[0] == 0x55:
                buffer = buffer[11:]
            else:
                buffer.pop(0)

    ser.close()

    arr = np.array(readings, dtype=float)
    mean = arr.mean(axis=0)

    print(
        f" done."
        f"  X={mean[0]:.1f}"
        f"  Y={mean[1]:.1f}"
        f"  Z={mean[2]:.1f}"
    )

    return mean


def main():

    print("===================================")
    print("      Magnetic Field Survey")
    print("===================================")
    print()

    displacements_mm = np.arange(0, 9, 1)

    print(
        f"Nominal magnet position: "
        f"{X_NOMINAL:.1f} mm"
    )
    print()

    input(
        "Place the magnet at its resting position.\n"
        "Press Enter to measure baseline..."
    )

    baseline = read_mag_average(PORT, BAUD)

    print("\nBaseline:")
    print(
        f"Bx={baseline[0]:.1f}  "
        f"By={baseline[1]:.1f}  "
        f"Bz={baseline[2]:.1f}"
    )

    results = []

    for disp in displacements_mm:

        print()
        print("-----------------------------------")
        print(f"Displacement = {disp:.0f} mm")
        print("-----------------------------------")

        if disp == 0:
            input(
                "Magnet at rest.\n"
                "Press Enter to measure..."
            )
        else:
            input(
                f"Move magnet {disp:.0f} mm "
                f"toward the sensor.\n"
                f"Press Enter to measure..."
            )

        reading = read_mag_average(PORT, BAUD)

        corrected = reading - baseline

        bx = corrected[0]
        by = corrected[1]
        bz = corrected[2]

        print()
        print("Baseline-corrected field:")
        print(f"Bx = {bx:8.1f}")
        print(f"By = {by:8.1f}")
        print(f"Bz = {bz:8.1f}")

        results.append(
            [disp, bx, by, bz]
        )

    results = np.array(results)

    print("\n===================================")
    print("Summary")
    print("===================================")

    print(
        f"{'Disp(mm)':>8}"
        f"{'Bx':>12}"
        f"{'By':>12}"
        f"{'Bz':>12}"
    )

    print("-" * 44)

    for row in results:

        print(
            f"{row[0]:8.0f}"
            f"{row[1]:12.1f}"
            f"{row[2]:12.1f}"
            f"{row[3]:12.1f}"
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

    print("\nSaved to field_survey.npy")


if __name__ == "__main__":
    main()
