# FILE: capture_gif.py
"""
Capture a LeKiwiSim simulation as a GIF — no display needed.
Uses EGL for headless rendering. Optimized for Discord (under 1MB).

Usage:  python capture_gif.py
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

# ── Renderer (smaller for Discord) ──────────────────────────────────────────
renderer = mujoco.Renderer(sim.engine.model, width=400, height=300)

# Camera: follow the robot
camera = mujoco.MjvCamera()
camera.distance = 1.2
camera.azimuth = 135
camera.elevation = -20
camera.lookat[:] = [0.0, 0.0, 0.1]

# ── Simulation ──────────────────────────────────────────────────────────────
frames = []
total_steps = 400  # 8 seconds at 50 fps
# Only capture every 2nd frame to reduce GIF size
capture_every = 2

for step in range(total_steps):
    if step < 80:
        # Phase 1: Arm stretches up
        arm_target = np.array([0.0, -0.8, 1.2, 0.0, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
    elif step < 160:
        # Phase 2: Arm reaches forward
        arm_target = np.array([1.0, 0.5, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
    elif step < 280:
        # Phase 3: Drive base forward while arm holds
        arm_target = np.array([1.0, 0.5, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.3, 0.0, 0.0])
    else:
        # Phase 4: Arm waves + base rotates
        t = (step - 280) / 120.0
        arm_target = np.array([0.5 + 0.5 * np.sin(t * 4 * np.pi), 0.0, 1.0, 0.0, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.5])

    sim.arm.step(arm_target)
    sim.base.step(base_vel)
    sim.step()

    # Update camera to follow base (from kinematic state)
    base_pose = sim.base.state
    camera.lookat[:] = [base_pose[0], base_pose[1], 0.1]

    # ── Render (every Nth frame) ──
    if step % capture_every == 0:
        renderer.update_scene(sim.engine.data, camera)
        pixels = renderer.render()
        frames.append(pixels)

renderer.close()

# ── Save GIF with optimization ──────────────────────────────────────────────
# Use slower-but-smaller settings
iio.imwrite(
    OUTPUT_PATH, frames,
    fps=50 // capture_every,  # effective fps
    loop=0,
    plugin='pillow',
    optimize=True,
)
print(f"✅ GIF saved: {OUTPUT_PATH}")
print(f"   {len(frames)} frames, 400x300, {50 // capture_every} fps")
file_size = Path(OUTPUT_PATH).stat().st_size
print(f"   File size: {file_size / 1024:.0f} KB")
