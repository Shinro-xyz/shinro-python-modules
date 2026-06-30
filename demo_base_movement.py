# FILE: capture_gif.py
"""
LeKiwi base-only demo — live viewer + growing plot + optional GIF capture.

Architecture:
  Base: LQR or MPC controller tracks waypoints in world frame [x, y, theta]
  Arm:  Frozen at home position (no control)

Usage:
  python capture_gif.py                          # live viewer + plot (LQR)
  python capture_gif.py --controller mpc         # live viewer + plot (MPC)
  python capture_gif.py --gif                     # render to GIF (headless)
  python capture_gif.py --gif --fast              # render to GIF, fewer frames
  python capture_gif.py --controller mpc --gif    # MPC + GIF
"""
import os
import sys

# ── Parse args ──────────────────────────────────────────────────────────────
RENDER_GIF = "--gif" in sys.argv
FAST = "--fast" in sys.argv
CONTROLLER = "lqr"  # default
TRAJECTORY = "straight"  # default
for i, arg in enumerate(sys.argv):
    if arg == "--controller" and i + 1 < len(sys.argv):
        CONTROLLER = sys.argv[i + 1]
    elif arg == "--trajectory" and i + 1 < len(sys.argv):
        TRAJECTORY = sys.argv[i + 1]

# Set MuJoCo GL backend BEFORE importing mujoco
if RENDER_GIF:
    os.environ['MUJOCO_GL'] = 'egl'
    import matplotlib
    matplotlib.use('Agg')
else:
    import matplotlib
    matplotlib.use('TkAgg')

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import time
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
OUTPUT_PATH = str(HERE / "lekiwi_demo.gif")

from lekiwi_sim import LeKiwiSim
from lqr import LQR


# ── Trajectory presets ──────────────────────────────────────────────────────
if TRAJECTORY == "triangle":
    BASE_WAYPOINTS = [
        (0.0, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (0.8, 0.0, 0.0),
        (1.2, 0.6, 0.0),
        (1.2, 0.6, 0.0),
        (1.2, 0.6, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
    BASE_WAYPOINT_STEPS = [50, 50, 100, 100, 100, 100, 100, 100, 100, 100]
else:
    # Straight line (default)
    BASE_WAYPOINTS = [
        (0.0, 0.0, 0.0),
        (0.4, 0.0, 0.0),
        (0.4, 0.0, 0.0),
        (0.4, 0.0, 0.0),
        (1.2, 0.0, 0.0),
        (1.2, 0.0, 0.0),
        (1.2, 0.0, 0.0),
    ]
    BASE_WAYPOINT_STEPS = [50, 50, 150, 100, 200, 150, 100]

# ── Build base schedule ────────────────────────────────────────────────────
total_steps = sum(BASE_WAYPOINT_STEPS)

base_schedule = []
for wp, n in zip(BASE_WAYPOINTS, BASE_WAYPOINT_STEPS):
    base_schedule.extend([np.array(wp)] * n)
base_schedule = np.array(base_schedule)


# ── Waypoint markers (colored poles in the 3D scene) ────────────────────────
def inject_waypoint_markers(xml_string, base_wps, base_steps):
    """Inject colored cylindrical markers into the MJCF XML before loading."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_string)
    worldbody = root.find('.//worldbody')

    # Collect unique waypoints with their step indices
    base_markers = []
    seen_base = set()
    step = 0
    for wp, n in zip(base_wps, base_steps):
        key = tuple(wp)
        if key not in seen_base:
            base_markers.append((step, wp))
            seen_base.add(key)
        step += n

    base_colors = [
        '1 0.4 0.4 0.6', '0.4 1 0.4 0.6', '0.4 0.4 1 0.6',
        '1 1 0.4 0.6', '1 0.4 1 0.6', '0.4 1 1 0.6', '1 0.7 0.3 0.6',
    ]

    for i, (_, pos) in enumerate(base_markers):
        geom = ET.SubElement(worldbody, 'geom')
        geom.set('type', 'cylinder')
        geom.set('size', '0.02 0.02 0.3')
        geom.set('pos', f'{pos[0]} {pos[1]} 0.15')
        geom.set('rgba', base_colors[i % len(base_colors)])
        geom.set('contype', '0')
        geom.set('conaffinity', '0')

    return ET.tostring(root, encoding='unicode')


def inject_free_joint(xml_string):
    """Restructure: make arm body a child of wheel base, add free joint to wheel base."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml_string)
    worldbody = root.find('.//worldbody')

    # Find the two sibling bodies
    wheel_base = None
    arm_base = None
    for child in list(worldbody):
        if child.tag == 'body':
            name = child.get('name', '')
            if 'base_plate_layer1' in name:
                wheel_base = child
            elif 'base_plate_layer2' in name:
                arm_base = child

    if wheel_base is not None and arm_base is not None:
        # Get wheel base world position
        wb_pos = [float(x) for x in wheel_base.get('pos', '0 0 0').split()]
        # Get arm world position
        ab_pos = [float(x) for x in arm_base.get('pos', '0 0 0').split()]
        # Make arm position relative to wheel base
        rel_pos = [ab_pos[i] - wb_pos[i] for i in range(3)]
        arm_base.set('pos', f'{rel_pos[0]} {rel_pos[1]} {rel_pos[2]}')

        # Remove arm from worldbody first
        worldbody.remove(arm_base)
        # Add free joint to wheel base
        fj = ET.Element('freejoint')
        wheel_base.insert(0, fj)
        # Move arm base inside wheel base
        wheel_base.append(arm_base)

    return ET.tostring(root, encoding='unicode')


# ── Create sim ──────────────────────────────────────────────────────────────
from lekiwi_sim import MJCF_PATH, HERE as LEKIWI_HOME
import xml.etree.ElementTree as ET

with open(MJCF_PATH) as f:
    base_xml = f.read()

# Collect mesh files as assets
mesh_dir = LEKIWI_HOME / 'lekiwi-sim' / 'meshes'
assets = {}
for fname in mesh_dir.iterdir():
    if fname.suffix in ('.stl', '.obj'):
        assets[fname.name] = fname.read_bytes()

# Inject waypoint markers and free joint into XML
xml_with_markers = inject_waypoint_markers(base_xml, BASE_WAYPOINTS, BASE_WAYPOINT_STEPS)
xml_with_freejoint = inject_free_joint(xml_with_markers)

sim = LeKiwiSim(dt=0.02, xml_string=xml_with_freejoint, assets=assets)
sim.reset()

# ── Base controller: LQR or MPC ────────────────────────────────────────────
A_base = np.eye(3)
B_base = 0.02 * np.eye(3)
Q_base = np.diag([100.0, 100.0, 50.0])
R_base = np.diag([0.1, 0.1, 0.1])

if CONTROLLER == "mpc":
    from mpc_lti import MPC_LTI_DeltaU
    # Δu penalty: S = diag(1.0, 1.0, 2.0) — penalizes rapid changes in vx, vy, ω
    # Higher ω penalty because yaw chatter is most visible
    S_delta = np.diag([1.0, 1.0, 2.0])
    base_ctrl = MPC_LTI_DeltaU(
        delta_u_penalty=S_delta,
        horizon=15,
        control_cost_matrix=R_base,
        state_cost_matrix=Q_base,
        A_dynamics=A_base,
        B_dynamics=B_base,
        terminal_cost=Q_base,
    )
    # Constraints: Δvx, Δvy ∈ [-0.5, 0.5], Δω ∈ [-1.0, 1.0]
    base_ctrl.constraints(
        np.vstack([np.eye(3), -np.eye(3)]),
        np.array([0.5, 0.5, 1.0, 0.5, 0.5, 1.0]),
        np.array([-0.5, -0.5, -1.0, -0.5, -0.5, -1.0]),
    )
    CTRL_LABEL = "MPC"
else:
    from lqr import LQR
    base_ctrl = LQR(Q_base, R_base, A_base, B_base)
    CTRL_LABEL = "LQR"

# ── Sensor noise ────────────────────────────────────────────────────────────
NOISE_BASE_POS = 0.02    # m std for x, y
NOISE_BASE_THETA = 0.05  # rad std for θ

# ── Luenberger observer for base ────────────────────────────────────────────
from luenberger_observer import LuenbergerObserver

# Observer gain: poles at ~0.1 (faster than LQR closed-loop ~0.37)
L_obs = np.diag([0.8, 0.8, 0.8])
base_observer = LuenbergerObserver(
    A=np.eye(3), B=0.02 * np.eye(3),
    observer_gain=L_obs,
    C=np.eye(3), D=np.zeros((3, 3)),
    x0=np.zeros((3, 1)),
)

# ── Data logging ───────────────────────────────────────────────────────────
log_time = []
log_base_ref = []       # reference [x, y, θ]
log_base_actual = []    # true state [x, y, θ]
log_base_noisy = []     # noisy measurement [x, y, θ]
log_base_estimated = [] # estimated state [x, y, θ]
log_base_error = []     # tracking error = ref - true
log_base_effort = []    # LQR output [vx, vy, ω]

# ── Plot setup ─────────────────────────────────────────────────────────────
if RENDER_GIF:
    renderer = mujoco.Renderer(sim.engine.model, width=400, height=300)
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
fig.patch.set_facecolor('#1a1a2e')
for ax in axes:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white', labelsize=7)
    ax.spines['bottom'].set_color('#555')
    ax.spines['top'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['right'].set_color('#555')

# ── Top: Base tracking ──
ax_base_track = axes[0]
ax_base_track.set_title('Base — Ref / Noisy Meas / Estimated / True', color='white', fontsize=9, fontweight='bold')
ax_base_track.set_ylabel('Position (m)', color='white', fontsize=8)
if TRAJECTORY == "triangle":
    ax_base_track.set_ylim(-0.2, 1.5)
else:
    ax_base_track.set_ylim(-0.1, 1.5)
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

# ── Middle: Observer innovation ──
ax_base_obs = axes[1]
ax_base_obs.set_title('Base — Observer Innovation (y − Cx̂)', color='white', fontsize=9, fontweight='bold')
ax_base_obs.set_ylabel('Innovation (m, rad)', color='white', fontsize=8)
ax_base_obs.set_ylim(-0.15, 0.15)
ax_base_obs.axhline(0, color='#555', lw=0.5)
line_bo_x, = ax_base_obs.plot([], [], '#ff6b6b', lw=1.0, label='x innov')
line_bo_y, = ax_base_obs.plot([], [], '#4ecdc4', lw=1.0, label='y innov')
line_bo_t, = ax_base_obs.plot([], [], '#ffe66d', lw=1.0, label='θ innov')
ax_base_obs.legend(loc='upper left', fontsize=7, labelcolor='white', framealpha=0.3)

# ── Bottom: Control effort ──
ax_base_ctrl = axes[2]
ax_base_ctrl.set_title(f'Base — {CTRL_LABEL} Control Effort (u = −Kx̂)', color='white', fontsize=9, fontweight='bold')
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

# ── Simulation loop ─────────────────────────────────────────────────────────
capture_every = 4 if FAST else 2

# Open MuJoCo viewer (only in live mode)
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

# Initialize observer state from true state
estimated_base = sim.base.get_state().copy()
base_vel = np.zeros(3)  # initial control input for observer

for step in range(total_steps):
    t = step * 0.02

    # ── Get true state ──
    true_base = sim.base.get_state()

    # ── Add sensor noise ──
    noisy_base = true_base + np.random.normal(
        [0, 0, 0], [NOISE_BASE_POS, NOISE_BASE_POS, NOISE_BASE_THETA]
    )

    # ── Luenberger observer (uses previous control input) ──
    estimated_base = base_observer.estimate(
        noisy_base.reshape(-1, 1), base_vel.reshape(-1, 1)
    ).flatten()

    # ── Controller using ESTIMATED state ──
    target_pose = base_schedule[step]
    if CONTROLLER == "mpc":
        error = estimated_base - target_pose
        base_vel = base_ctrl.compute(error, u_prev=base_vel)
    else:
        base_vel = base_ctrl.compute(estimated_base, target_pose)
    base_vel = np.clip(base_vel, [-0.5, -0.5, -1.0], [0.5, 0.5, 1.0])
    sim.base.step(base_vel)

    # ── Step physics ──
    sim.step()

    # ── Log data ──
    log_time.append(t)
    log_base_ref.append(target_pose.copy())
    log_base_actual.append(true_base.copy())
    log_base_noisy.append(noisy_base.copy())
    log_base_estimated.append(estimated_base.copy())
    log_base_error.append((target_pose - true_base).copy())
    log_base_effort.append(base_vel.copy())

    # ── Update plot every 5 steps ──
    if step % 5 == 0 and step > 0:
        time_arr = np.array(log_time)
        br_arr = np.array(log_base_ref)
        ba_arr = np.array(log_base_actual)
        bn_arr = np.array(log_base_noisy)
        be_arr = np.array(log_base_estimated)
        bo_arr = be_arr - ba_arr  # innovation = estimated - true
        bc_arr = np.array(log_base_effort)

        # Base tracking
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
        ax_base_track.relim()
        ax_base_track.autoscale_view()

        # Observer innovation
        line_bo_x.set_data(time_arr, bo_arr[:, 0])
        line_bo_y.set_data(time_arr, bo_arr[:, 1])
        line_bo_t.set_data(time_arr, bo_arr[:, 2])
        ax_base_obs.relim()
        ax_base_obs.autoscale_view()

        # Control effort
        line_bc_x.set_data(time_arr, bc_arr[:, 0])
        line_bc_y.set_data(time_arr, bc_arr[:, 1])
        line_bc_t.set_data(time_arr, bc_arr[:, 2])
        ax_base_ctrl.relim()
        ax_base_ctrl.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

    # ── Capture frame for GIF ──
    if RENDER_GIF and step % capture_every == 0:
        renderer.update_scene(sim.engine.data, camera)
        frames.append(renderer.render())

    # ── Sync viewer ──
    if viewer is not None:
        viewer.sync()
        time.sleep(sim.engine.dt / 4)
        if not viewer.is_running():
            break

# ── Save GIF ────────────────────────────────────────────────────────────────
if RENDER_GIF and frames:
    import imageio
    fps = 12 if not FAST else 6
    imageio.mimsave(OUTPUT_PATH, frames, fps=fps, loop=0)
    print(f"✅ GIF saved to {OUTPUT_PATH} ({len(frames)} frames, {fps} fps)")

print("✅ Demo complete")
