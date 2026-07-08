# FILE: demos/demo_pick_and_place.py
"""
Pick-and-place demo: arm reaches, grips, lifts, base drives, places, releases.

Usage:  python demos/demo_pick_and_place.py
Output: lekiwi_demo.gif
"""
import os
os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import mujoco
import imageio.v3 as iio
from pathlib import Path

from lekiwi_sim import RobotSim, MJCF_PATH, HERE as LEKIWI_HOME
from factories import TrajectoryFactory

HERE = Path(__file__).parent.parent
OUTPUT_PATH = str(HERE / "lekiwi_demo.gif")

# ── XML injection ────────────────────────────────────────────────────────
import xml.etree.ElementTree as ET

def inject_free_joint(xml_string):
    root = ET.fromstring(xml_string)
    worldbody = root.find('.//worldbody')
    wheel_base = arm_base = None
    for child in list(worldbody):
        if child.tag == 'body':
            name = child.get('name', '')
            if 'base_plate_layer1' in name:
                wheel_base = child
            elif 'base_plate_layer2' in name:
                arm_base = child
    if wheel_base is not None and arm_base is not None:
        wb_pos = [float(x) for x in wheel_base.get('pos', '0 0 0').split()]
        ab_pos = [float(x) for x in arm_base.get('pos', '0 0 0').split()]
        rel_pos = [ab_pos[i] - wb_pos[i] for i in range(3)]
        arm_base.set('pos', f'{rel_pos[0]} {rel_pos[1]} {rel_pos[2]}')
        worldbody.remove(arm_base)
        fj = ET.Element('freejoint')
        wheel_base.insert(0, fj)
        wheel_base.append(arm_base)
    return ET.tostring(root, encoding='unicode')

# ── Create sim ───────────────────────────────────────────────────────────
with open(MJCF_PATH) as f:
    base_xml = f.read()
mesh_dir = LEKIWI_HOME / 'lekiwi-sim' / 'meshes'
assets = {}
for fname in mesh_dir.iterdir():
    if fname.suffix in ('.stl', '.obj'):
        assets[fname.name] = fname.read_bytes()

xml = inject_free_joint(base_xml)
sim = RobotSim(str(HERE / "robot_config.yaml"), xml_string=xml, assets=assets)
sim.reset()

# ── Load phase schedule ──────────────────────────────────────────────────
sched = TrajectoryFactory(str(HERE / "configs/trajectories/pick_and_place.yaml")).create()
total_steps = len(sched["arm"])

# ── Renderer ─────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(sim.engine.model, width=400, height=300)
camera = mujoco.MjvCamera()
camera.distance = 1.5
camera.azimuth = 135
camera.elevation = -20
camera.lookat[:] = [0.0, 0.0, 0.1]

frames = []
capture_every = 2

for step in range(total_steps):
    arm_target = sched["arm"][step].copy()
    arm_target[5] = sched["jaw"][step]
    base_vel = sched["base"][step]

    sim.arm.step(arm_target)
    sim.base.step(base_vel)
    sim.step()

    base_pose = sim.base.get_state()
    camera.lookat[:] = [base_pose[0], base_pose[1], 0.1]

    if step % capture_every == 0:
        renderer.update_scene(sim.engine.data, camera)
        frames.append(renderer.render())

renderer.close()

iio.imwrite(
    OUTPUT_PATH, frames,
    fps=50 // capture_every, loop=0,
    plugin='pillow', optimize=True,
)
print(f"✅ GIF saved: {OUTPUT_PATH} ({len(frames)} frames, 400x300, {50 // capture_every} fps)")
print(f"   File size: {Path(OUTPUT_PATH).stat().st_size / 1024:.0f} KB")