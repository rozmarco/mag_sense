# debug_model.py - run this standalone to see what's happening
import numpy as np
import magpylib as magpy

cal = np.load('calibration.npy', allow_pickle=True).item()
SCALE = cal['scale']
BASELINE = cal['baseline']

print(f"Scale: {SCALE:.6e}")
print(f"Baseline: {BASELINE}")
print()

# Your measured corrected Bz values from calibration:
measured = {
    0.001: 16234.38,
    0.006: 16277.12,
    0.011: 20855.16,
    0.016: 34397.32,
}

# Current magnet model
magnet = magpy.magnet.Cylinder(
    polarization=(0, 0, 1.0),
    dimension=(0.095, 0.003),
    position=(0, 0, 0),
)
sensor = magpy.Sensor()

print("z(mm)  | Measured Bz | Model*Scale Bz | Ratio")
print("-" * 55)
for z, meas in measured.items():
    sensor.position = (0.0, 0.0, float(-z))
    B = magpy.getB(magnet, sensor)
    pred = B[2] * SCALE
    ratio = meas / pred if pred != 0 else float('inf')
    print(f"{z*1000:5.1f}  | {meas:11.1f} | {pred:14.1f} | {ratio:.3f}")

# Also show what Bz your current live reading implies
live_bz = 48388.5
print(f"\nLive measured Bz: {live_bz:.1f}")
print("Searching for z that matches live reading...")
from scipy.optimize import minimize_scalar

def residual(z):
    sensor.position = (0.0, 0.0, float(-z))
    B = magpy.getB(magnet, sensor)
    pred = B[2] * SCALE
    return (pred - live_bz)**2

result = minimize_scalar(residual, bounds=(0.0001, 0.05), method='bounded')
print(f"Best matching z: {result.x*1000:.3f} mm  (residual: {result.fun:.1f})")
