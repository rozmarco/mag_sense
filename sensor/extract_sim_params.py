# extract_sim_params.py
#
# Reads multimagnet_cal.npy produced by calibrate_multimagnet.py and
# extracts / fits all parameters needed by sim_observability.py.
#
# What it does:
#   1. Loads geometry and noise directly from the cal file
#   2. Fits the magnet dipole moment (A·m²) by minimising the error between
#      the simulated and measured field vectors for EVERY single-magnet state
#      simultaneously (not just M1 Bz) — gives a much more robust fit
#   3. Reports fit quality: simulated vs measured centroid for each magnet
#   4. Saves sim_params.npy — the simulator loads this instead of manual values
#
# Usage:
#   python extract_sim_params.py
#   python extract_sim_params.py --cal path/to/other_cal.npy
#
# Output: sim_params.npy

import argparse
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import magpylib as magpy

# WT901 LSB conversion (must match sim_observability.py)
MT_TO_LSB = 1.0 / 0.001529   # LSB per mT


# ---------------------------------------------------------------------------
# Magpylib helpers (duplicated here so this script is self-contained)
# ---------------------------------------------------------------------------

def _build_magnet(x_mm, y_mm, polarity, z_centre_mm,
                  moment_am2, diam_mm, thick_mm):
    vol_m3        = np.pi * (diam_mm / 2 * 1e-3) ** 2 * (thick_mm * 1e-3)
    magnetisation = moment_am2 / vol_m3
    pol           = (0, 0, -polarity * magnetisation)
    return magpy.magnet.Cylinder(
        polarization=pol,
        dimension=(diam_mm, thick_mm),
        position=(x_mm, y_mm, z_centre_mm),
    )


def _get_field_lsb(magnets, sensor_pos):
    sensor = magpy.Sensor(position=sensor_pos)
    B_mT   = magpy.getB(magnets, sensor)
    if B_mT.ndim == 2:
        B_mT = B_mT.sum(axis=0)
    return B_mT * MT_TO_LSB


def _simulate_single_press(layout, diam_mm, thick_mm,
                            rest_height_mm, press_mm, moment_am2,
                            pressed_idx, sensor_pos):
    """
    Return corrected field vector (3,) for one magnet pressed.
    Baseline = all at rest.
    """
    z_rest  = rest_height_mm + thick_mm / 2.0
    z_press = z_rest - press_mm

    baseline_mags = [_build_magnet(x, y, p, z_rest, moment_am2, diam_mm, thick_mm)
                     for (x, y, p) in layout]
    B_base = _get_field_lsb(baseline_mags, sensor_pos)

    pressed_mags = []
    for k, (x, y, p) in enumerate(layout):
        z = z_press if k == pressed_idx else z_rest
        pressed_mags.append(_build_magnet(x, y, p, z, moment_am2, diam_mm, thick_mm))
    B_press = _get_field_lsb(pressed_mags, sensor_pos)

    return B_press - B_base


# ---------------------------------------------------------------------------
# Moment fitting
# ---------------------------------------------------------------------------

def fit_moment(cal, sensor_pos=np.array([0., 0., 0.])):
    """
    Fit a single scalar moment (A·m²) that minimises the total squared error
    between simulated and measured corrected field vectors across all
    single-magnet states found in the cal file.

    Returns (moment_am2, residuals_dict)
    """
    layout   = cal['magnet_layout']
    diam     = cal['magnet_diameter_mm']
    thick    = cal['magnet_thickness_mm']
    rest_h   = cal['rest_height_mm']
    press    = cal['press_depth_mm']
    centroids = cal['centroids']

    # Find which states are single-magnet (M1, M2, M3, M4 etc.)
    # and which index in layout they correspond to
    single_labels = [f'M{i+1}' for i in range(len(layout))]
    measured = {}
    for i, lbl in enumerate(single_labels):
        if lbl in centroids:
            measured[i] = (lbl, centroids[lbl])

    if not measured:
        raise ValueError("No single-magnet states (M1, M2, ...) found in cal file.")

    print(f"  Fitting moment against {len(measured)} single-magnet states: "
          f"{[v[0] for v in measured.values()]}")

    def total_residual(log_moment):
        moment = np.exp(log_moment)
        err    = 0.0
        for idx, (lbl, measured_vec) in measured.items():
            sim_vec = _simulate_single_press(
                layout, diam, thick, rest_h, press, moment, idx, sensor_pos)
            err += np.sum((sim_vec - measured_vec) ** 2)
        return err

    result = minimize_scalar(total_residual, bounds=(-12, 4), method='bounded')
    moment = np.exp(result.x)

    # Per-state residuals for reporting
    residuals = {}
    for idx, (lbl, measured_vec) in measured.items():
        sim_vec         = _simulate_single_press(
            layout, diam, thick, rest_h, press, moment, idx, sensor_pos)
        residuals[lbl]  = {
            'measured': measured_vec,
            'simulated': sim_vec,
            'error_lsb': sim_vec - measured_vec,
            'rmse': float(np.sqrt(np.mean((sim_vec - measured_vec) ** 2))),
        }

    return moment, residuals


# ---------------------------------------------------------------------------
# Sensor position fitting (optional — improves fit if sensor is off-centre)
# ---------------------------------------------------------------------------

def fit_sensor_position(cal, moment_am2):
    """
    Optionally refine the sensor XY position within the array.
    Useful if your sensor isn't exactly at (0,0,0) relative to M1.
    Returns best-fit sensor_pos (3,) array.
    """
    layout    = cal['magnet_layout']
    diam      = cal['magnet_diameter_mm']
    thick     = cal['magnet_thickness_mm']
    rest_h    = cal['rest_height_mm']
    press     = cal['press_depth_mm']
    centroids = cal['centroids']

    single_labels = [f'M{i+1}' for i in range(len(layout))]
    measured = {}
    for i, lbl in enumerate(single_labels):
        if lbl in centroids:
            measured[i] = (lbl, centroids[lbl])

    def total_residual(xy):
        sensor_pos = np.array([xy[0], xy[1], 0.0])
        err = 0.0
        for idx, (lbl, measured_vec) in measured.items():
            sim_vec = _simulate_single_press(
                layout, diam, thick, rest_h, press, moment_am2, idx, sensor_pos)
            err += np.sum((sim_vec - measured_vec) ** 2)
        return err

    result = minimize(total_residual, x0=[0.0, 0.0],
                      method='Nelder-Mead',
                      options={'xatol': 0.01, 'fatol': 1e6, 'maxiter': 2000})
    return np.array([result.x[0], result.x[1], 0.0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cal', default='multimagnet_cal_sim_data.npy',
                        help='Path to calibration file')
    parser.add_argument('--fit-sensor-pos', action='store_true',
                        help='Also fit sensor XY position (slower, more accurate)')
    args = parser.parse_args()

    print("=== Simulation Parameter Extraction ===")
    print()

    cal = np.load(args.cal, allow_pickle=True).item()

    # ---- Check required keys ----
    required = ['noise_std', 'press_depth_mm', 'rest_height_mm',
                'magnet_diameter_mm', 'magnet_thickness_mm', 'magnet_layout']
    missing  = [k for k in required if k not in cal]
    if missing:
        print(f"ERROR: Cal file is missing keys: {missing}")
        print("Re-run calibrate_multimagnet.py to generate an updated cal file.")
        return

    # ---- Print geometry loaded from cal ----
    print("Geometry loaded from cal file:")
    print(f"  rest_height:       {cal['rest_height_mm']} mm")
    print(f"  press_depth:       {cal['press_depth_mm']} mm")
    print(f"  magnet_diameter:   {cal['magnet_diameter_mm']} mm")
    print(f"  magnet_thickness:  {cal['magnet_thickness_mm']} mm")
    print(f"  magnet_layout:     {cal['magnet_layout']}")
    print(f"  noise_std:         {cal['noise_std']}")
    print()

    # ---- Fit moment ----
    print("Fitting magnet moment...")
    sensor_pos     = np.array([0., 0., 0.])
    moment, resids = fit_moment(cal, sensor_pos)
    print(f"  Fitted moment: {moment:.6f} A·m²")
    print()

    # ---- Optional sensor position fit ----
    if args.fit_sensor_pos:
        print("Fitting sensor position...")
        sensor_pos = fit_sensor_position(cal, moment)
        print(f"  Fitted sensor pos: x={sensor_pos[0]:.2f}  y={sensor_pos[1]:.2f} mm")
        # Re-fit moment with corrected sensor position
        print("  Re-fitting moment with corrected sensor position...")
        moment, resids = fit_moment(cal, sensor_pos)
        print(f"  Re-fitted moment: {moment:.6f} A·m²")
        print()

    # ---- Fit quality report ----
    print("Fit quality (simulated vs measured per magnet):")
    print(f"  {'Label':6}  {'meas Bx':>10}  {'sim Bx':>10}  "
          f"{'meas By':>10}  {'sim By':>10}  "
          f"{'meas Bz':>10}  {'sim Bz':>10}  {'RMSE':>10}")

    all_rmse = []
    for lbl, r in resids.items():
        m, s = r['measured'], r['simulated']
        print(f"  {lbl:6}  {m[0]:>10.1f}  {s[0]:>10.1f}  "
              f"{m[1]:>10.1f}  {s[1]:>10.1f}  "
              f"{m[2]:>10.1f}  {s[2]:>10.1f}  {r['rmse']:>10.1f}")
        all_rmse.append(r['rmse'])

    mean_rmse = np.mean(all_rmse)
    print(f"\n  Mean RMSE: {mean_rmse:.1f} LSB", end='  ')

    noise_mean = cal['noise_std'].mean()
    if mean_rmse < noise_mean * 3:
        print("✓  (within 3× noise floor — good fit)")
    elif mean_rmse < noise_mean * 10:
        print("⚠  (moderate fit — consider --fit-sensor-pos)")
    else:
        print("✗  (poor fit — check geometry values at top of calibrate_multimagnet.py)")

    # ---- Axis dominance diagnosis ----
    print()
    print("Axis dominance analysis:")
    axis_names = ['Bx', 'By', 'Bz']
    for lbl, r in resids.items():
        dominant_meas = axis_names[np.argmax(np.abs(r['measured']))]
        dominant_sim  = axis_names[np.argmax(np.abs(r['simulated']))]
        match = "✓" if dominant_meas == dominant_sim else "✗ MISMATCH"
        print(f"  {lbl}: measured dominant={dominant_meas}  "
              f"simulated dominant={dominant_sim}  {match}")

    # ---- Save ----
    sim_params = {
        # From cal file directly
        'noise_std':             cal['noise_std'],
        'press_depth_mm':        cal['press_depth_mm'],
        'rest_height_mm':        cal['rest_height_mm'],
        'magnet_diameter_mm':    cal['magnet_diameter_mm'],
        'magnet_thickness_mm':   cal['magnet_thickness_mm'],
        'magnet_layout':         cal['magnet_layout'],
        # Fitted
        'moment_am2':            moment,
        'sensor_pos':            sensor_pos,
        # Fit quality metadata
        'fit_rmse_mean':         mean_rmse,
        'fit_residuals':         resids,
        # Pass-through for reference
        'measured_centroids':    cal['centroids'],
        'axis_std':              cal['axis_std'],
    }

    np.save('sim_params.npy', sim_params)
    print()
    print("Saved to sim_params.npy")
    print()
    print("Next step: run sim_observability.py")
    print("  It will load sim_params.npy automatically.")
    print("  Edit MAGNET_LAYOUT / REST_HEIGHT_MM / MOMENT_AM2 in sim_observability.py")
    print("  to explore alternative designs — the fitted noise_std is always used.")


if __name__ == '__main__':
    main()