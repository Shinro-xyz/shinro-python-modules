# FILE: capture_demo.py
"""
Demo: robot picks up a box, drives to another point, places it.
Uses EGL for headless rendering.

Usage:  python capture_demo.py
Output: lekiwi_demo.gif
"""

import os

os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

HERE = Path(__file__).parent
OUTPUT_PATH = str(HERE / "lekiwi_demo.gif")

from lekiwi_sim import LeKiwiSim

sim = LeKiwiSim(dt=0.02)
sim.reset()

renderer = mujoco.Renderer(sim.engine.model, width=400, height=300)

camera = mujoco.MjvCamera()
camera.distance = 1.5
camera.azimuth = 135
camera.elevation = -20
camera.lookat[:] = [0.0, 0.0, 0.1]

frames = []
total_steps = 600
capture_every = 2

for step in range(total_steps):
    t = step / 50.0

    if t < 1.5:
        arm_target = np.array([0.0, -0.8, 1.2, 0.5, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = True
    elif t < 2.5:
        arm_target = np.array([0.0, -0.8, 1.2, 0.5, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = False
    elif t < 4.0:
        arm_target = np.array([0.0, -0.3, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = False
    elif t < 7.0:
        arm_target = np.array([0.0, -0.3, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.3, 0.0, 0.0])
        jaw_open = False
    elif t < 8.5:
        arm_target = np.array([0.0, -0.8, 1.2, 0.5, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = False
    elif t < 9.5:
        arm_target = np.array([0.0, -0.8, 1.2, 0.5, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = True
    elif t < 11.0:
        arm_target = np.array([0.0, -0.3, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.3, 0.0, 0.0])
        jaw_open = True
    else:
        arm_target = np.array([0.0, -0.3, 0.8, 0.0, 0.0, 0.0])
        base_vel = np.array([0.0, 0.0, 0.0])
        jaw_open = True

    arm_target[5] = 0.0 if jaw_open else 0.5

    sim.arm.step(arm_target)
    sim.base.step(base_vel)
    sim.step()

    base_pose = sim.base.get_state()
    camera.lookat[:] = [base_pose[0], base_pose[1], 0.1]

    if step % capture_every == 0:
        renderer.update_scene(sim.engine.data, camera)
        pixels = renderer.render()
        frames.append(pixels)

renderer.close()

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
