# FILE: capture_demo.py
"""
Demo: robot picks up a box, drives to another point, places it.
Uses EGL for headless rendering.

Usage:  python capture_demo.py
Output: lekiwi_demo.gif
"""

import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import mujoco
from pathlib import Path
import imageio.v3 as iio

HERE = Path(__file__).parent
OUTPUT_PATH = str(HERE / "lekiwi_demo.gif")

from lekiwi_sim import LeKiwiSim

# ── Create sim ──────────────────────────────────────────────────────────────
sim = LeKiwiSim(dt=0.02)
sim.reset()

# ── Renderer ────────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(sim.engine.model, width=400, height=300)

camera = mujoco.MjvCamera()
camera.distance = 1.5
camera.azimuth = 135
camera.elevation = -20
camera.lookat[:] = [0.0, 0.0, 0.1]

# ── Simulation ──────────────────────────────────────────────────────────────
frames = []
total_steps = 600  # 12 seconds at 50 fps
capture_every = 2

# Box position in world frame
BOX_POS = np.array([0.15, 0.0, 0.025])

for step in range(total_steps):
    t = step / 50.0  # time in seconds

    if t < 1.5:
        # Phase 1: Reach down to box
        # Arm: Pitch down, Elbow out, Wrist down to reach the box
        arm_target = np.array([0.0, -0.8, 1.2, 0.5, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = True

    elif t < 2.5:
        # Phase 2: Close jaw (grip the box)
        arm_target = np.array([0.0, -0.8, 1.2, 0.5, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = False

    elif t < 4.0:
        # Phase 3: Lift arm up with box
        arm_target = np.array([0.0, -0.3, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = False

    elif t < 7.0:
        # Phase 4: Drive forward while holding box
        arm_target = np.array([0.0, -0.3, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.3, 0.0, 0.0])
        jaw_open = False

    elif t < 8.5:
        # Phase 5: Reach down to place
        arm_target = np.array([0.0, -0.8, 1.2, 0.5, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = False

    elif t < 9.5:
        # Phase 6: Open jaw (release box)
        arm_target = np.array([0.0, -0.8, 1.2, 0.5, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = True

    elif t < 11.0:
        # Phase 7: Lift arm and drive away
        arm_target = np.array([0.0, -0.3, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.3, 0.0, 0.0])
        jaw_open = True

    else:
        # Phase 8: Hold
        arm_target = np.array([0.0, -0.3, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = True

    # Set jaw position (0 = open, 0.5 = closed)
    arm_target[5] = 0.0 if jaw_open else 0.5

    sim.arm.step(arm_target)
    sim.base.step(base_vel)
    sim.step()

    # Update camera to follow base
    base_pose = sim.base.state
    camera.lookat[:] = [base_pose[0], base_pose[1], 0.1]

    # Render every Nth frame
    if step % capture_every == 0:
        renderer.update_scene(sim.engine.data, camera)
        pixels = renderer.render()
        frames.append(pixels)

renderer.close()

# ── Save GIF ──────────────────────────────────────────────────────────────
iio.imwrite(
    OUTPUT_PATH, frames,
    fps=50 // capture_every,
    loop=0,
    plugin='pillow',
    optimize=True,
)
print(f"✅ GIF saved: {OUTPUT_PATH}")
print(f"   {len(frames)} frames, 400x300, {50 // capture_every} fps")
file_size = Path(OUTPUT_PATH).stat().st_size
print(f"   File size: {file_size / 1024:.0f} KB")
