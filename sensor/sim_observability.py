# sim_observability.py
#
# Simulates your 2x2 magnet array over a single WT901 magnetometer,
# evaluates how well each magnet displacement is observable given
# real sensor noise, and produces plots to guide physical design.
#
# What it does:
#   1. Build a magpylib model of your magnet array + sensor geometry
#   2. Fit magnet moment magnitude to match your real calibration data
#      (or use a manual estimate if you don't have cal data)
#   3. Sweep magnet displacement (press depth) and compute field change
#   4. Estimate signal-to-noise ratio (SNR) per axis per magnet
#   5. Compute pairwise centroid separations in normalised space
#   6. Plot: field vs displacement, SNR, pairwise separation heatmap,
#            and axis contribution breakdown
#
# Dependencies:
#   pip install magpylib numpy scipy matplotlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from itertools import combinations
import magpylib as magpy

# =============================================================================
# SYSTEM PARAMETERS — edit these to match / explore your design
# =============================================================================

# ---- Sensor position ----
SENSOR_POS = np.array([0.0, 0.0, 0.0])   # mm — sensor at origin

# ---- Magnet array layout ----
# Each entry: (x_mm, y_mm, polarity)
# Polarity +1 = north pole faces sensor (downward), -1 = south faces sensor
MAGNET_LAYOUT = [
    (0.0,  0.0,  +1),   # M1 top-left
    (0.0,  15.0, -1),   # M2 top-right   (15mm = 1.5cm spacing)
    (15.0, 0.0,  -1),   # M3 bot-left
    (15.0, 15.0, +1),   # M4 bot-right
]
MAGNET_LABELS = ['M1', 'M2', 'M3', 'M4']

# ---- Magnet geometry ----
MAGNET_DIAMETER   = 5.0    # mm — cylindrical magnet diameter
MAGNET_THICKNESS  = 3.0    # mm — cylindrical magnet height

# ---- Rest height (unloaded) ----
# Distance from top face of magnet to sensor face when not pressed, in mm.
# The magnet centre is at rest_height + MAGNET_THICKNESS/2 above sensor.
REST_HEIGHT = 10.0   # mm

# ---- Press displacement sweep ----
# How far the magnet moves DOWN toward the sensor when pressed.
# Script sweeps from 0 (unloaded) to MAX_PRESS in N_PRESS steps.
MAX_PRESS  = 5.0    # mm
N_PRESS    = 50

# ---- Magnet moment magnitude ----
# magpylib uses SI: magnetisation M in A/m, volume in m³, moment = M * V (A·m²)
# You can:
#   a) Set FIT_TO_CAL = True and provide your calibration file path
#      → script will fit moment to match your real Bz reading at rest
#   b) Set FIT_TO_CAL = False and set MOMENT_AM2 manually
FIT_TO_CAL    = False
CAL_FILE      = 'multimagnet_cal_sim_data.npy'   # only used if FIT_TO_CAL = True
MOMENT_AM2    = 0.05   # A·m²  — starting estimate for N35 5mm×3mm disc magnet

# ---- Sensor noise ----
# Measured std from your calibration data (from the printed std lines).
# If you haven't measured it, a conservative estimate for WT901 is ~5-15 LSB.
# Enter your three axis stds here:
SENSOR_NOISE_STD = np.array([10.0, 10.0, 10.0])   # LSB (raw counts)

# WT901 magnetometer scale factor: 1 LSB ≈ 0.01529 µT (for ±800µT range / 2^15)
# We keep everything in sensor LSB for consistency with your real data.
# The field from magpylib comes in mT — convert to LSB:
MT_TO_LSB = 1.0 / 0.001529   # ≈ 653.7 LSB per µT → 653700 LSB per mT
# (If your sensor uses a different range, adjust this)

# ---- Axis normalisation weights ----
# These are used when computing normalised pairwise distances.
# Set to None to auto-compute from centroid spread (recommended),
# or provide a (3,) array to override.
AXIS_STD_OVERRIDE = None

# =============================================================================
# END PARAMETERS
# =============================================================================


def build_magnet(x_mm, y_mm, polarity, z_centre_mm, moment_am2):
    """
    Create a magpylib Cylinder magnet.
    Polarity controls orientation: +1 = magnetised in +z (north up),
    but since magnet is above sensor, +z magnetisation pushes field
    down into sensor. We set orientation accordingly.
    """
    volume_m3    = np.pi * (MAGNET_DIAMETER / 2 * 1e-3) ** 2 * (MAGNET_THICKNESS * 1e-3)
    magnetisation = moment_am2 / volume_m3   # A/m

    # magpylib polarisation vector in local frame (z = along cylinder axis)
    # polarity +1 → north faces down toward sensor → polarisation in -z
    # polarity -1 → south faces down → polarisation in +z
    pol = (0, 0, -polarity * magnetisation)

    mag = magpy.magnet.Cylinder(
        polarization=pol,
        dimension=(MAGNET_DIAMETER, MAGNET_THICKNESS),   # (diameter, height) mm
        position=(x_mm, y_mm, z_centre_mm),
    )
    return mag


def get_field_lsb(magnets, sensor_pos_mm, moment_am2):
    """
    Compute field at sensor_pos from a list of magpylib magnets.
    Returns field in sensor LSB units (matching raw WT901 output).
    getB returns (N_magnets, 3) when multiple sources — sum to get total field.
    """
    sensor = magpy.Sensor(position=sensor_pos_mm)
    B_mT   = magpy.getB(magnets, sensor)   # (N_magnets, 3) or (3,)
    if B_mT.ndim == 2:
        B_mT = B_mT.sum(axis=0)            # superpose all contributions → (3,)
    return B_mT * MT_TO_LSB


def rest_z_centre(press_mm=0.0):
    """Z position of magnet centre at a given press depth."""
    return REST_HEIGHT + MAGNET_THICKNESS / 2.0 - press_mm


def fit_moment_to_cal(cal_file, layout):
    """
    Fit MOMENT_AM2 so that the simulated Bz at rest matches the real
    unloaded→M1 Bz delta from your calibration file.
    Uses scipy scalar minimisation.
    """
    from scipy.optimize import minimize_scalar

    cal       = np.load(cal_file, allow_pickle=True).item()
    centroids = cal['centroids']

    # Target: Bz of M1 centroid (strongest single-magnet signal)
    target_bz_lsb = centroids['M1'][2]   # already baseline-corrected
    print(f"  Fitting moment to match M1 Bz = {target_bz_lsb:.1f} LSB ...")

    def residual(log_moment):
        moment = np.exp(log_moment)
        mags   = []
        for (x, y, pol) in layout:
            z = rest_z_centre(0.0)
            mags.append(build_magnet(x, y, pol, z, moment))
        # Field with M1 only (others not pressed, but include all for realism)
        # Actually compute M1-only contribution:
        m1 = build_magnet(layout[0][0], layout[0][1], layout[0][2],
                          rest_z_centre(0.0), moment)
        B  = get_field_lsb([m1], SENSOR_POS, moment)
        return (B[2] - target_bz_lsb) ** 2

    result  = minimize_scalar(residual, bounds=(-10, 5), method='bounded')
    fitted  = np.exp(result.x)
    print(f"  Fitted moment: {fitted:.6f} A·m²")
    return fitted


def simulate_press_sweep(moment_am2):
    """
    For each magnet, sweep press depth from 0 to MAX_PRESS.
    At each depth, compute the field change (relative to all-unloaded baseline).
    Returns dict: {label: (N_PRESS, 3) array of delta-B in LSB}
    """
    press_depths = np.linspace(0, MAX_PRESS, N_PRESS)

    # Baseline: all magnets at rest height, no press
    baseline_mags = []
    for (x, y, pol) in MAGNET_LAYOUT:
        baseline_mags.append(build_magnet(x, y, pol, rest_z_centre(0.0), moment_am2))
    B_baseline = get_field_lsb(baseline_mags, SENSOR_POS, moment_am2)

    sweep_results = {}

    for i, label in enumerate(MAGNET_LABELS):
        dB_array = np.zeros((N_PRESS, 3))
        for j, press in enumerate(press_depths):
            mags = []
            for k, (x, y, pol) in enumerate(MAGNET_LAYOUT):
                z = rest_z_centre(press if k == i else 0.0)
                mags.append(build_magnet(x, y, pol, z, moment_am2))
            B       = get_field_lsb(mags, SENSOR_POS, moment_am2)
            dB_array[j] = B - B_baseline
        sweep_results[label] = dB_array

    return press_depths, sweep_results, B_baseline


def compute_centroids_at_full_press(moment_am2, press_mm):
    """
    Compute the field vector (corrected) for each single-magnet press
    and all-unloaded state at a given press depth.
    Returns dict {label: (3,) array}
    """
    baseline_mags = [build_magnet(x, y, pol, rest_z_centre(0.0), moment_am2)
                     for (x, y, pol) in MAGNET_LAYOUT]
    B_baseline = get_field_lsb(baseline_mags, SENSOR_POS, moment_am2)

    centroids = {'unloaded': np.zeros(3)}

    for i, label in enumerate(MAGNET_LABELS):
        mags = []
        for k, (x, y, pol) in enumerate(MAGNET_LAYOUT):
            z = rest_z_centre(press_mm if k == i else 0.0)
            mags.append(build_magnet(x, y, pol, z, moment_am2))
        B = get_field_lsb(mags, SENSOR_POS, moment_am2)
        centroids[label] = B - B_baseline

    return centroids


def snr_at_press(sweep_results, press_depths, noise_std):
    """
    For each magnet, compute per-axis SNR at every press depth.
    SNR = |delta_B| / noise_std  per axis.
    Returns dict {label: (N_PRESS, 3) SNR array}
    """
    snr = {}
    for label, dB in sweep_results.items():
        snr[label] = np.abs(dB) / noise_std   # (N_PRESS, 3)
    return snr


def pairwise_separations(centroids, axis_std):
    """
    Compute normalised Euclidean distance between every pair of centroids.
    """
    labels = list(centroids.keys())
    seps   = {}
    for l1, l2 in combinations(labels, 2):
        diff      = (centroids[l1] - centroids[l2]) / axis_std
        seps[(l1, l2)] = np.linalg.norm(diff)
    return seps


def plot_results(press_depths, sweep_results, snr_results,
                 centroids, axis_std, noise_std, moment_am2, out_path):

    colours    = {'Bx': '#e74c3c', 'By': '#2ecc71', 'Bz': '#3498db'}
    axis_names = ['Bx', 'By', 'Bz']
    n_magnets  = len(MAGNET_LABELS)

    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor('#1a1a2e')
    gs  = gridspec.GridSpec(4, n_magnets, figure=fig,
                            hspace=0.45, wspace=0.35)

    title_kw  = dict(color='white', fontsize=11, fontweight='bold', pad=8)
    label_kw  = dict(color='#aaaaaa', fontsize=9)
    tick_kw   = dict(colors='#888888', labelsize=8)

    def style_ax(ax):
        ax.set_facecolor('#16213e')
        ax.tick_params(axis='both', **tick_kw)
        ax.spines[:].set_color('#444466')
        ax.xaxis.label.set(**label_kw)
        ax.yaxis.label.set(**label_kw)

    # ── Row 0: Field change vs press depth ──────────────────────────────────
    for i, label in enumerate(MAGNET_LABELS):
        ax  = fig.add_subplot(gs[0, i])
        dB  = sweep_results[label]
        for j, aname in enumerate(axis_names):
            ax.plot(press_depths, dB[:, j],
                    color=list(colours.values())[j], label=aname, linewidth=1.8)
        ax.axhline(0, color='#555577', linewidth=0.7, linestyle='--')
        ax.set_title(f'{label} — ΔB vs press depth', **title_kw)
        ax.set_xlabel('Press depth (mm)', **label_kw)
        ax.set_ylabel('ΔB (LSB)', **label_kw)
        ax.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='white',
                  edgecolor='#444466')
        style_ax(ax)

    # ── Row 1: SNR vs press depth ────────────────────────────────────────────
    for i, label in enumerate(MAGNET_LABELS):
        ax  = fig.add_subplot(gs[1, i])
        snr = snr_results[label]
        for j, aname in enumerate(axis_names):
            ax.plot(press_depths, snr[:, j],
                    color=list(colours.values())[j], label=aname, linewidth=1.8)
        ax.axhline(1.0, color='#e67e22', linewidth=1.2,
                   linestyle='--', label='SNR=1')
        ax.axhline(3.0, color='#f1c40f', linewidth=1.0,
                   linestyle=':', label='SNR=3')
        ax.set_title(f'{label} — SNR vs press depth', **title_kw)
        ax.set_xlabel('Press depth (mm)', **label_kw)
        ax.set_ylabel('SNR (σ)', **label_kw)
        ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                  edgecolor='#444466')
        style_ax(ax)

    # ── Row 2: Axis contribution breakdown at full press ────────────────────
    full_press_idx = -1
    bar_width = 0.25
    x_pos     = np.arange(3)

    for i, label in enumerate(MAGNET_LABELS):
        ax    = fig.add_subplot(gs[2, i])
        dB_fp = sweep_results[label][full_press_idx]   # (3,)
        snr_fp = snr_results[label][full_press_idx]    # (3,)

        bars = ax.bar(x_pos, np.abs(dB_fp),
                      color=[colours[a] for a in axis_names],
                      alpha=0.85, edgecolor='#333355', linewidth=0.8)

        # Overlay SNR as text
        for b, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(np.abs(dB_fp)) * 0.02,
                    f'SNR={snr_fp[b]:.1f}σ',
                    ha='center', va='bottom', color='white', fontsize=7.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(axis_names, color='#aaaaaa')
        ax.set_title(f'{label} — axis contributions\n@ {MAX_PRESS}mm press',
                     **title_kw)
        ax.set_ylabel('|ΔB| (LSB)', **label_kw)
        # Noise floor line
        noise_line = noise_std.mean()
        ax.axhline(noise_line, color='#e74c3c', linewidth=1.2,
                   linestyle='--', label=f'noise floor ({noise_line:.0f} LSB)')
        ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                  edgecolor='#444466')
        style_ax(ax)

    # ── Row 3: Pairwise separation heatmap ──────────────────────────────────
    all_labels = list(centroids.keys())
    n_states   = len(all_labels)
    sep_matrix = np.zeros((n_states, n_states))

    for i, l1 in enumerate(all_labels):
        for j, l2 in enumerate(all_labels):
            if i != j:
                diff = (centroids[l1] - centroids[l2]) / axis_std
                sep_matrix[i, j] = np.linalg.norm(diff)

    ax_heat = fig.add_subplot(gs[3, :2])
    im      = ax_heat.imshow(sep_matrix, cmap='RdYlGn',
                              vmin=0, vmax=sep_matrix.max())
    ax_heat.set_xticks(range(n_states))
    ax_heat.set_yticks(range(n_states))
    ax_heat.set_xticklabels(all_labels, rotation=45, ha='right',
                             color='#aaaaaa', fontsize=8)
    ax_heat.set_yticklabels(all_labels, color='#aaaaaa', fontsize=8)
    ax_heat.set_title('Pairwise separation (normalised)', **title_kw)
    ax_heat.set_facecolor('#16213e')

    for i in range(n_states):
        for j in range(n_states):
            val  = sep_matrix[i, j]
            col  = 'black' if val > sep_matrix.max() * 0.6 else 'white'
            ax_heat.text(j, i, f'{val:.2f}', ha='center', va='center',
                         fontsize=7, color=col)

    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#888888', labelsize=8)
    cbar.set_label('Normalised distance', color='#aaaaaa', fontsize=9)

    # ── Row 3 right: design parameter summary ───────────────────────────────
    ax_txt = fig.add_subplot(gs[3, 2:])
    ax_txt.set_facecolor('#16213e')
    ax_txt.axis('off')

    seps     = pairwise_separations(centroids, axis_std)
    min_pair = min(seps, key=seps.get)
    min_sep  = seps[min_pair]
    bad_pairs = [(k, v) for k, v in seps.items() if v < 1.5]

    full_snr = {l: snr_results[l][-1] for l in MAGNET_LABELS}
    min_snr_label = min(full_snr, key=lambda l: full_snr[l].max())

    summary = (
        f"DESIGN SUMMARY\n"
        f"{'─'*34}\n"
        f"Moment:          {moment_am2:.4f} A·m²\n"
        f"Rest height:     {REST_HEIGHT} mm\n"
        f"Max press:       {MAX_PRESS} mm\n"
        f"Magnet spacing:  {MAGNET_LAYOUT[1][1]} mm\n"
        f"Noise std:       Bx={noise_std[0]:.1f} By={noise_std[1]:.1f} Bz={noise_std[2]:.1f} LSB\n"
        f"\n"
        f"axis_std (norm): Bx={axis_std[0]:.0f} By={axis_std[1]:.0f} Bz={axis_std[2]:.0f} LSB\n"
        f"\n"
        f"Worst separation:\n"
        f"  {min_pair[0]} <-> {min_pair[1]}: {min_sep:.3f}\n"
        f"\n"
        f"Pairs below threshold (1.5):\n"
    )
    if bad_pairs:
        for (l1, l2), v in bad_pairs:
            summary += f"  {l1} <-> {l2}: {v:.3f}  ← WARNING\n"
    else:
        summary += "  None — all states well separated ✓\n"

    ax_txt.text(0.05, 0.95, summary,
                transform=ax_txt.transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace', color='#e0e0e0',
                bbox=dict(boxstyle='round', facecolor='#0f3460',
                          edgecolor='#444466', alpha=0.8))

    fig.suptitle(
        f'Magnet Array Observability Analysis\n'
        f'moment={moment_am2:.4f} A·m²  |  rest_height={REST_HEIGHT}mm  |  '
        f'spacing={MAGNET_LAYOUT[1][1]}mm  |  noise={noise_std.mean():.1f} LSB',
        color='white', fontsize=13, fontweight='bold', y=0.98
    )

    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print(f"  Plot saved to {out_path}")
    plt.close()


def main():
    print("=== Magnet Array Observability Simulation ===")
    print()

    # ---- Fit or use manual moment ----
    if FIT_TO_CAL:
        print("Fitting magnet moment to calibration data...")
        moment = fit_moment_to_cal(CAL_FILE, MAGNET_LAYOUT)
    else:
        moment = MOMENT_AM2
        print(f"Using manual moment: {moment} A·m²")

    print()
    print(f"Rest height:    {REST_HEIGHT} mm")
    print(f"Max press:      {MAX_PRESS} mm")
    print(f"Magnet spacing: {MAGNET_LAYOUT[1][1]} mm")
    print(f"Noise std:      {SENSOR_NOISE_STD} LSB")
    print()

    # ---- Sweep ----
    print("Simulating press sweep...")
    press_depths, sweep_results, B_baseline = simulate_press_sweep(moment)
    print(f"  Baseline field: Bx={B_baseline[0]:.1f}  By={B_baseline[1]:.1f}  Bz={B_baseline[2]:.1f} LSB")

    # ---- SNR ----
    snr_results = snr_at_press(sweep_results, press_depths, SENSOR_NOISE_STD)

    # ---- Centroids at full press ----
    centroids = compute_centroids_at_full_press(moment, MAX_PRESS)

    print("\nSimulated centroids at full press:")
    print(f"  {'Label':12s}  {'Bx':>10}  {'By':>10}  {'Bz':>10}")
    for lbl, c in centroids.items():
        print(f"  {lbl:12s}  {c[0]:>10.1f}  {c[1]:>10.1f}  {c[2]:>10.1f}")

    # ---- Axis std for normalisation ----
    if AXIS_STD_OVERRIDE is not None:
        axis_std = AXIS_STD_OVERRIDE
    else:
        centroid_matrix = np.array(list(centroids.values()))
        axis_std        = centroid_matrix.std(axis=0)
        axis_std        = np.where(axis_std < 1.0, 1.0, axis_std)

    print(f"\nAxis std (for normalisation): Bx={axis_std[0]:.1f}  By={axis_std[1]:.1f}  Bz={axis_std[2]:.1f}")

    # ---- Pairwise separations ----
    seps = pairwise_separations(centroids, axis_std)
    print("\nPairwise separations (normalised):")
    bad = False
    for (l1, l2), d in sorted(seps.items(), key=lambda x: x[1]):
        flag = "  ← WARNING" if d < 1.5 else ""
        print(f"  {l1:12s} <-> {l2:12s}  {d:.3f}{flag}")
        if flag:
            bad = True

    if bad:
        print("\n  ↑ Some states poorly separated. Try:")
        print("    - Increasing MAX_PRESS")
        print("    - Increasing MOMENT_AM2")
        print("    - Reducing REST_HEIGHT")
        print("    - Changing magnet spacing")
    else:
        print("\n  All states well separated ✓")

    # ---- SNR summary at full press ----
    print("\nSNR at full press (per axis):")
    print(f"  {'Label':6}  {'Bx SNR':>10}  {'By SNR':>10}  {'Bz SNR':>10}  {'dominant':>10}")
    axis_names = ['Bx', 'By', 'Bz']
    for label in MAGNET_LABELS:
        snr_fp   = snr_results[label][-1]
        dominant = axis_names[np.argmax(snr_fp)]
        print(f"  {label:6}  {snr_fp[0]:>10.1f}  {snr_fp[1]:>10.1f}  {snr_fp[2]:>10.1f}  {dominant:>10}")

    # ---- Plot ----
    print("\nGenerating plots...")
    plot_results(press_depths, sweep_results, snr_results,
                 centroids, axis_std, SENSOR_NOISE_STD, moment,
                 out_path='observability_analysis.png')

    print("\nDone.")
    print()
    print("To explore design space, edit these parameters at the top of the script:")
    print("  REST_HEIGHT, MAX_PRESS, MOMENT_AM2, SENSOR_NOISE_STD, MAGNET_LAYOUT")
    print("  Set FIT_TO_CAL = True to anchor the moment to your real calibration data.")


if __name__ == '__main__':
    main()