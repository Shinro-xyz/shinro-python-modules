# FILE: demos/demo_base_tracking.py
"""
Base-only demo: LQR/MPC controller tracks waypoints with state estimation.

Usage:
  python demos/demo_base_tracking.py                          # live viewer + plot (LQR)
  python demos/demo_base_tracking.py --controller mpc         # live viewer + plot (MPC)
  python demos/demo_base_tracking.py --gif                     # render to GIF (headless)
  python demos/demo_base_tracking.py --gif --fast              # fewer frames
  python demos/demo_base_tracking.py --controller mpc --gif    # MPC + GIF
  python demos/demo_base_tracking.py --trajectory triangle     # triangle path
  python demos/demo_base_tracking.py --controller mpc --trajectory triangle --gif
"""
import os
import sys

RENDER_GIF = "--gif" in sys.argv
FAST = "--fast" in sys.argv
CONTROLLER = "lqr"
TRAJECTORY = "straight"
for i, arg in enumerate(sys.argv):
    if arg == "--controller" and i + 1 < len(sys.argv):
        CONTROLLER = sys.argv[i + 1]
    elif arg == "--trajectory" and i + 1 < len(sys.argv):
        TRAJECTORY = sys.argv[i + 1]

if RENDER_GIF:
    os.environ['MUJOCO_GL'] = 'egl'
    import matplotlib
    matplotlib.use('Agg')
else:
    import matplotlib
    matplotlib.use('TkAgg')

import numpy as np
import mujoco
import time
import matplotlib.pyplot as plt
from pathlib import Path

from lekiwi_sim import RobotSim, MJCF_PATH, HERE as LEKIWI_HOME
from factories import ControllerFactory, EstimatorFactory, TrajectoryFactory

HERE = Path(__file__).parent.parent
OUTPUT_PATH = str(HERE / "lekiwi_demo.gif")

# ── XML injection helpers ────────────────────────────────────────────────
def inject_waypoint_markers(xml_string, base_wps, base_steps):
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_string)
    worldbody = root.find('.//worldbody')
    base_markers = []
    seen_base = set()
    step = 0
    for wp, n in zip(base_wps, base_steps):
        key = tuple(wp)
        if key not in seen_base:
            base_markers.append((step, wp))
            seen_base.add(key)
        step += n
    colors = ['1 0.4 0.4 0.6', '0.4 1 0.4 0.6', '0.4 0.4 1 0.6',
              '1 1 0.4 0.6', '1 0.4 1 0.6', '0.4 1 1 0.6', '1 0.7 0.3 0.6']
    for i, (_, pos) in enumerate(base_markers):
        geom = ET.SubElement(worldbody, 'geom')
        geom.set('type', 'cylinder')
        geom.set('size', '0.02 0.02 0.3')
        geom.set('pos', f'{pos[0]} {pos[1]} 0.15')
        geom.set('rgba', colors[i % len(colors)])
        geom.set('contype', '0')
        geom.set('conaffinity', '0')
    return ET.tostring(root, encoding='unicode')

def inject_free_joint(xml_string):
    import xml.etree.ElementTree as ET
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

# ── Build waypoint schedule ──────────────────────────────────────────────
traj_config = f"configs/trajectories/base_{TRAJECTORY}.yaml"
base_schedule = TrajectoryFactory(str(HERE / traj_config)).create()
total_steps = len(base_schedule)

# Extract waypoints for markers
BASE_WAYPOINTS = []
BASE_WAYPOINT_STEPS = []
with open(HERE / traj_config) as f:
    import yaml
    cfg = yaml.safe_load(f)
    for wp in cfg["waypoints"]:
        pos = tuple(wp["position"])
        if not BASE_WAYPOINTS or pos != tuple(BASE_WAYPOINTS[-1]):
            BASE_WAYPOINTS.append(list(pos))
            BASE_WAYPOINT_STEPS.append(int(np.round(wp["duration"] / 0.02)))
        else:
            BASE_WAYPOINT_STEPS[-1] += int(np.round(wp["duration"] / 0.02))

# ── Create sim with XML injection ───────────────────────────────────────
with open(MJCF_PATH) as f:
    base_xml = f.read()
mesh_dir = LEKIWI_HOME / 'lekiwi-sim' / 'meshes'
assets = {}
for fname in mesh_dir.iterdir():
    if fname.suffix in ('.stl', '.obj'):
        assets[fname.name] = fname.read_bytes()

xml = inject_free_joint(inject_waypoint_markers(base_xml, BASE_WAYPOINTS, BASE_WAYPOINT_STEPS))
sim = RobotSim(str(HERE / "robot_config.yaml"), xml_string=xml, assets=assets)
sim.reset()

# ── Controller ───────────────────────────────────────────────────────────
ctrl_config = f"configs/controllers/{CONTROLLER}_base.yaml"
base_ctrl = ControllerFactory(str(HERE / ctrl_config)).create()
CTRL_LABEL = CONTROLLER.upper()

# ── Estimator ────────────────────────────────────────────────────────────
base_observer = EstimatorFactory(str(HERE / "configs/estimators/luenberger_base.yaml")).create()

# ── Noise ────────────────────────────────────────────────────────────────
NOISE_BASE_POS = 0.02
NOISE_BASE_THETA = 0.05

# ── Logging ──────────────────────────────────────────────────────────────
log_time = []
log_base_ref = []
log_base_actual = []
log_base_noisy = []
log_base_estimated = []
log_base_error = []
log_base_effort = []

# ── Plot setup ───────────────────────────────────────────────────────────
GIF_WIDTH = 640
GIF_HEIGHT = 400

if RENDER_GIF:
    renderer = mujoco.Renderer(sim.engine.model, width=GIF_WIDTH, height=GIF_HEIGHT)
    camera = mujoco.MjvCamera()
    camera.distance = 1.8
    camera.azimuth = 135
    camera.elevation = -20
    camera.lookat[:] = [0.0, 0.0, 0.1]
    frames = []
else:
    renderer = None
    camera = None
    frames = None

fig, axes = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
from demos.helpers import setup_dark_plot
setup_dark_plot(fig, axes)

ax_base_track = axes[0]
ax_base_track.set_title('Base — Ref / Noisy Meas / Estimated / True', color='white', fontsize=9, fontweight='bold')
ax_base_track.set_ylabel('Position (m)', color='white', fontsize=8)
ax_base_track.set_ylim(-0.2, 1.5)
line_br_x, = ax_base_track.plot([], [], '#ff6b6b', lw=1.5, ls='--', label='x ref')
line_br_y, = ax_base_track.plot([], [], '#4ecdc4', lw=1.5, ls='--', label='y ref')
line_br_t, = ax_base_track.plot([], [], '#ffe66d', lw=1.5, ls='--', label='θ ref')
line_bn_x, = ax_base_track.plot([], [], '#ff6b6b', lw=0, marker='.', ms=2, alpha=0.4, label='x noisy')
line_bn_y, = ax_base_track.plot([], [], '#4ecdc4', lw=0, marker='.', ms=2, alpha=0.4, label='y noisy')
line_bn_t, = ax_base_track.plot([], [], '#ffe66d', lw=0, marker='.', ms=2, alpha=0.4, label='θ noisy')
line_be_x, = ax_base_track.plot([], [], '#ff6b6b', lw=2.0, label='x est')
line_be_y, = ax_base_track.plot([], [], '#4ecdc4', lw=2.0, label='y est')
line_be_t, = ax_base_track.plot([], [], '#ffe66d', lw=2.0, label='θ est')
line_ba_x, = ax_base_track.plot([], [], '#ff6b6b', lw=0.8, alpha=0.3, label='x true')
line_ba_y, = ax_base_track.plot([], [], '#4ecdc4', lw=0.8, alpha=0.3, label='y true')
line_ba_t, = ax_base_track.plot([], [], '#ffe66d', lw=0.8, alpha=0.3, label='θ true')
ax_base_track.legend(loc='upper left', fontsize=6, labelcolor='white', framealpha=0.3, ncol=2)

ax_base_obs = axes[1]
ax_base_obs.set_title('Observer Innovation (y − Cx̂)', color='white', fontsize=9, fontweight='bold')
ax_base_obs.set_ylabel('Innovation (m, rad)', color='white', fontsize=8)
ax_base_obs.set_ylim(-0.15, 0.15)
ax_base_obs.axhline(0, color='#555', lw=0.5)
line_bo_x, = ax_base_obs.plot([], [], '#ff6b6b', lw=1.0, label='x innov')
line_bo_y, = ax_base_obs.plot([], [], '#4ecdc4', lw=1.0, label='y innov')
line_bo_t, = ax_base_obs.plot([], [], '#ffe66d', lw=1.0, label='θ innov')
ax_base_obs.legend(loc='upper left', fontsize=7, labelcolor='white', framealpha=0.3)

ax_base_ctrl = axes[2]
ax_base_ctrl.set_title(f'Control Effort — {CTRL_LABEL}', color='white', fontsize=9, fontweight='bold')
ax_base_ctrl.set_ylabel('Velocity (m/s, rad/s)', color='white', fontsize=8)
ax_base_ctrl.set_xlabel('Time (s)', color='white', fontsize=8)
ax_base_ctrl.set_ylim(-0.6, 0.6)
ax_base_ctrl.axhline(0, color='#555', lw=0.5)
line_bc_x, = ax_base_ctrl.plot([], [], '#ff6b6b', lw=1.5, label='vx')
line_bc_y, = ax_base_ctrl.plot([], [], '#4ecdc4', lw=1.5, label='vy')
line_bc_t, = ax_base_ctrl.plot([], [], '#ffe66d', lw=1.5, label='ω')
ax_base_ctrl.legend(loc='upper left', fontsize=7, labelcolor='white', framealpha=0.3)

plt.tight_layout()
plt.ion()
plt.show(block=False)

capture_every = 4 if FAST else 2

if not RENDER_GIF:
    viewer = mujoco.viewer.launch_passive(sim.engine.model, sim.engine.data)
    if TRAJECTORY == "triangle":
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30
        viewer.cam.lookat[:] = [0.6, 0.3, 0.1]
    else:
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30
        viewer.cam.lookat[:] = [0.6, 0.0, 0.1]
else:
    viewer = None

estimated_base = sim.base.get_state().copy()
base_vel = np.zeros(3)

for step in range(total_steps):
    t = step * 0.02

    true_base = sim.base.get_state()
    noisy_base = true_base + np.random.normal(
        [0, 0, 0], [NOISE_BASE_POS, NOISE_BASE_POS, NOISE_BASE_THETA]
    )

    estimated_base = base_observer.estimate(
        noisy_base.reshape(-1, 1), base_vel.reshape(-1, 1)
    ).flatten()

    target_pose = base_schedule[step]
    if CONTROLLER == "mpc":
        error = estimated_base - target_pose
        base_vel = base_ctrl.compute(error, u_prev=base_vel)
    else:
        base_vel = base_ctrl.compute(estimated_base, target_pose)
    base_vel = np.clip(base_vel, [-0.5, -0.5, -1.0], [0.5, 0.5, 1.0])
    sim.base.step(base_vel)
    sim.step()

    log_time.append(t)
    log_base_ref.append(target_pose.copy())
    log_base_actual.append(true_base.copy())
    log_base_noisy.append(noisy_base.copy())
    log_base_estimated.append(estimated_base.copy())
    log_base_error.append((target_pose - true_base).copy())
    log_base_effort.append(base_vel.copy())

    if step % 5 == 0 and step > 0:
        time_arr = np.array(log_time)
        br_arr = np.array(log_base_ref)
        ba_arr = np.array(log_base_actual)
        bn_arr = np.array(log_base_noisy)
        be_arr = np.array(log_base_estimated)
        bo_arr = be_arr - ba_arr
        bc_arr = np.array(log_base_effort)

        line_br_x.set_data(time_arr, br_arr[:, 0])
        line_br_y.set_data(time_arr, br_arr[:, 1])
        line_br_t.set_data(time_arr, br_arr[:, 2])
        line_bn_x.set_data(time_arr, bn_arr[:, 0])
        line_bn_y.set_data(time_arr, bn_arr[:, 1])
        line_bn_t.set_data(time_arr, bn_arr[:, 2])
        line_be_x.set_data(time_arr, be_arr[:, 0])
        line_be_y.set_data(time_arr, be_arr[:, 1])
        line_be_t.set_data(time_arr, be_arr[:, 2])
        line_ba_x.set_data(time_arr, ba_arr[:, 0])
        line_ba_y.set_data(time_arr, ba_arr[:, 1])
        line_ba_t.set_data(time_arr, ba_arr[:, 2])
        ax_base_track.relim(); ax_base_track.autoscale_view()

        line_bo_x.set_data(time_arr, bo_arr[:, 0])
        line_bo_y.set_data(time_arr, bo_arr[:, 1])
        line_bo_t.set_data(time_arr, bo_arr[:, 2])
        ax_base_obs.relim(); ax_base_obs.autoscale_view()

        line_bc_x.set_data(time_arr, bc_arr[:, 0])
        line_bc_y.set_data(time_arr, bc_arr[:, 1])
        line_bc_t.set_data(time_arr, bc_arr[:, 2])
        ax_base_ctrl.relim(); ax_base_ctrl.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

    if RENDER_GIF and step % capture_every == 0:
        renderer.update_scene(sim.engine.data, camera)
        mujoco_frame = renderer.render()
        from demos.helpers import make_composite_frame
        frames.append(make_composite_frame(mujoco_frame, fig))

    if viewer is not None:
        viewer.sync()
        time.sleep(sim.engine.dt / 4)
        if not viewer.is_running():
            break

if RENDER_GIF and frames:
    from demos.helpers import save_gif
    fps = 12 if not FAST else 6
    save_gif(frames, OUTPUT_PATH, fps=fps)

print("✅ Demo complete")