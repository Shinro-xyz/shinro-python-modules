# FILE: capture_gif.py
"""
LeKiwi pick-and-place demo — live viewer + growing plot + optional GIF capture.

Architecture:
  Base: LQR controller tracks waypoints in world frame [x, y, theta]
  Arm:  P-controller in EE space → fallback FK/IK → mirror joints to MuJoCo
  Gripper: Direct jaw position command

  The arm uses the fallback FK/IK path (not MuJoCo physics) because MuJoCo's
  position servos (kp=50) can't track small Jacobian steps under gravity.
  Joint positions are mirrored to MuJoCo for rendering.

Usage:
  python capture_gif.py              # live viewer + plot
  python capture_gif.py --gif        # render to GIF (headless)
  python capture_gif.py --gif --fast # render to GIF, fewer frames
"""
import os
import sys

# ── Parse args ──────────────────────────────────────────────────────────────
RENDER_GIF = "--gif" in sys.argv
FAST = "--fast" in sys.argv

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
from armrobot import ArmRobot


# ── EE Pose Trajectory ──────────────────────────────────────────────────────
EE_KEYFRAMES = [
    (50,  np.array([0.015,  0.101,  0.031, 0.0, 0.0, 0.0])),
    (100, np.array([0.015,  0.101, -0.050, 0.0, 0.0, 0.0])),
    (50,  np.array([0.015,  0.101, -0.050, 0.0, 0.0, 0.0])),
    (100, np.array([0.015,  0.101,  0.100, 0.0, 0.0, 0.0])),
    (200, np.array([0.015,  0.101,  0.100, 0.0, 0.0, 0.0])),
    (100, np.array([0.015,  0.101, -0.050, 0.0, 0.0, 0.0])),
    (50,  np.array([0.015,  0.101, -0.050, 0.0, 0.0, 0.0])),
    (100, np.array([0.015,  0.101,  0.031, 0.0, 0.0, 0.0])),
]

GRIP_SCHEDULE = [
    (150, 0.5),
    (450, 0.0),
]

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

# ── Build schedules ─────────────────────────────────────────────────────────
total_steps = sum(k[0] for k in EE_KEYFRAMES)

base_schedule = []
for wp, n in zip(BASE_WAYPOINTS, BASE_WAYPOINT_STEPS):
    base_schedule.extend([np.array(wp)] * n)
base_schedule = np.array(base_schedule[:total_steps])

ee_schedule = []
for n, pose in EE_KEYFRAMES:
    ee_schedule.extend([pose.copy() for _ in range(n)])
ee_schedule = np.array(ee_schedule)


# ── Waypoint markers (colored poles in the 3D scene) ────────────────────────
def inject_waypoint_markers(xml_string, base_wps, ee_keyframes, base_steps):
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

    ee_markers = []
    seen_ee = set()
    step = 0
    for n, pose in ee_keyframes:
        key = tuple(pose[:3])
        if key not in seen_ee:
            ee_markers.append((step, tuple(pose[:3])))
            seen_ee.add(key)
        step += n

    base_colors = [
        '1 0.4 0.4 0.6', '0.4 1 0.4 0.6', '0.4 0.4 1 0.6',
        '1 1 0.4 0.6', '1 0.4 1 0.6', '0.4 1 1 0.6', '1 0.7 0.3 0.6',
    ]
    ee_colors = [
        '1 0.6 0.6 0.5', '0.6 1 0.6 0.5', '0.6 0.6 1 0.5',
        '1 1 0.6 0.5', '1 0.6 1 0.5', '0.6 1 1 0.5',
        '1 0.8 0.5 0.5', '0.8 0.6 1 0.5',
    ]

    for i, (_, pos) in enumerate(base_markers):
        geom = ET.SubElement(worldbody, 'geom')
        geom.set('type', 'cylinder')
        geom.set('size', '0.02 0.02 0.3')
        geom.set('pos', f'{pos[0]} {pos[1]} 0.15')
        geom.set('rgba', base_colors[i % len(base_colors)])
        geom.set('contype', '0')
        geom.set('conaffinity', '0')

    for i, (_, pos) in enumerate(ee_markers):
        geom = ET.SubElement(worldbody, 'geom')
        geom.set('type', 'cylinder')
        geom.set('size', '0.015 0.015 0.2')
        geom.set('pos', f'{pos[0]} {pos[1]} 0.1')
        geom.set('rgba', ee_colors[i % len(ee_colors)])
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
# Read MJCF, inject waypoint markers, then load with mesh assets
from lekiwi_sim import MJCF_PATH, HERE as LEKIWI_HOME
import xml.etree.ElementTree as ET

with open(MJCF_PATH) as f:
    base_xml = f.read()

# Inject waypoint markers and free joint into XML
xml_with_markers = inject_waypoint_markers(base_xml, BASE_WAYPOINTS, EE_KEYFRAMES, BASE_WAYPOINT_STEPS)
xml_with_freejoint = inject_free_joint(xml_with_markers)

# Collect mesh files as assets for from_xml_string
mesh_dir = LEKIWI_HOME / 'lekiwi-sim' / 'meshes'
assets = {}
for fname in mesh_dir.iterdir():
    if fname.suffix in ('.stl', '.obj'):
        assets[fname.name] = fname.read_bytes()

sim = LeKiwiSim(dt=0.02, xml_string=xml_with_freejoint, assets=assets)
sim.reset()

# ── Fallback arm (FK/IK, no MuJoCo physics) ─────────────────────────────────
NUM_DOF = 6
DT = 0.02
JOINT_LIMITS = np.array([
    [-3.0,   3.0],
    [-3.1416, 3.14],
    [-3.14,  3.1416],
    [-3.0,   3.14],
    [-3.1416, 3.1416],
    [-3.14,  3.0],
])
LINK_OFFSETS = np.array([
    [0.018300,  0.030600,  0.052200],
    [-0.001500, -0.114582,  0.018082],
    [-0.001500,  0.132932,  0.028720],
    [-0.020100,  0.025822, -0.055375],
    [0.019800,  0.026631, -0.013098],
    [0.0,        0.0,       0.0],
])
ROT_AXES = ["y", "z", "z", "x", "z", "z"]

arm_fallback = ArmRobot(NUM_DOF, DT, JOINT_LIMITS, LINK_OFFSETS, ROT_AXES)
T_home, _, _ = arm_fallback.forward_kinematics(np.zeros(6))
arm_fallback.state = np.array([T_home[0,3], T_home[1,3], T_home[2,3], 0.0, 0.0, 0.0])

# ── LQR for base ────────────────────────────────────────────────────────────
A_base = np.eye(3)
B_base = 0.02 * np.eye(3)
Q_base = np.diag([100.0, 100.0, 50.0])
R_base = np.diag([0.1, 0.1, 0.1])
base_ctrl = LQR(Q_base, R_base, A_base, B_base)

# ── Sensor noise ────────────────────────────────────────────────────────────
NOISE_BASE_POS = 0.02    # m std for x, y
NOISE_BASE_THETA = 0.05  # rad std for θ
NOISE_ARM_POS = 0.01     # m std for x, y, z

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

# ── Arm EE estimator (exponential moving average) ───────────────────────────
arm_estimated_ee = [np.zeros(3)]  # list wrapper to avoid Python scoping issues
ALPHA_ARM = 0.3  # EMA factor (lower = smoother)

# ── Data logging ───────────────────────────────────────────────────────────
log_time = []
log_base_ref = []       # reference [x, y, θ]
log_base_actual = []    # true state [x, y, θ]
log_base_noisy = []     # noisy measurement [x, y, θ]
log_base_estimated = [] # estimated state [x, y, θ]
log_base_error = []     # tracking error = ref - true
log_base_effort = []    # LQR output [vx, vy, ω]
log_arm_ref = []        # reference EE [x, y, z]
log_arm_actual = []     # true EE [x, y, z]
log_arm_noisy = []      # noisy EE measurement [x, y, z]
log_arm_estimated = []  # estimated EE [x, y, z]
log_arm_error = []      # tracking error = ref - true
log_arm_effort = []     # P-controller output [vx, vy, vz]

# ── Plot setup ─────────────────────────────────────────────────────────────
if RENDER_GIF:
    renderer = mujoco.Renderer(sim.engine.model, width=400, height=300)
    camera = mujoco.MjvCamera()
    camera.distance = 1.8
    camera.azimuth = 135
    camera.elevation = -20
    camera.lookat[:] = [0.0, 0.0, 0.1]
    frames = []
    viewer_ctx = None  # no viewer in headless mode
else:
    renderer = None
    camera = None
    frames = None
    viewer_ctx = "launch"  # will open viewer below

fig, axes = plt.subplots(3, 2, figsize=(10, 6.5), sharex=True)
fig.patch.set_facecolor('#1a1a2e')
for ax in axes.flat:
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='white', labelsize=7)
    ax.spines['bottom'].set_color('#555')
    ax.spines['top'].set_color('#555')
    ax.spines['left'].set_color('#555')
    ax.spines['right'].set_color('#555')

# ── Row 0: Base — reference, noisy measurement, estimated, true ──
ax_base_track, ax_base_obs = axes[0]
ax_base_track.set_title('Base — Ref / Noisy Meas / Estimated / True', color='white', fontsize=9, fontweight='bold')
ax_base_track.set_ylabel('Position (m)', color='white', fontsize=8)
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

ax_base_obs.set_title('Base — Observer Innovation (y − Cx̂)', color='white', fontsize=9, fontweight='bold')
ax_base_obs.set_ylabel('Innovation (m, rad)', color='white', fontsize=8)
ax_base_obs.set_ylim(-0.15, 0.15)
ax_base_obs.axhline(0, color='#555', lw=0.5)
line_bo_x, = ax_base_obs.plot([], [], '#ff6b6b', lw=1.0, label='x innov')
line_bo_y, = ax_base_obs.plot([], [], '#4ecdc4', lw=1.0, label='y innov')
line_bo_t, = ax_base_obs.plot([], [], '#ffe66d', lw=1.0, label='θ innov')
ax_base_obs.legend(loc='upper left', fontsize=7, labelcolor='white', framealpha=0.3)

# ── Row 1: Arm — reference, noisy measurement, estimated, true ──
ax_arm_track, ax_arm_obs = axes[1]
ax_arm_track.set_title('Arm EE — Ref / Noisy Meas / Estimated / True', color='white', fontsize=9, fontweight='bold')
ax_arm_track.set_ylabel('Position (m)', color='white', fontsize=8)
ax_arm_track.set_ylim(-0.15, 0.25)
line_ar_x, = ax_arm_track.plot([], [], '#ff6b6b', lw=1.5, ls='--', label='x ref')
line_ar_y, = ax_arm_track.plot([], [], '#4ecdc4', lw=1.5, ls='--', label='y ref')
line_ar_z, = ax_arm_track.plot([], [], '#45b7d1', lw=1.5, ls='--', label='z ref')
line_an_x, = ax_arm_track.plot([], [], '#ff6b6b', lw=0, marker='.', ms=2, alpha=0.4, label='x noisy')
line_an_y, = ax_arm_track.plot([], [], '#4ecdc4', lw=0, marker='.', ms=2, alpha=0.4, label='y noisy')
line_an_z, = ax_arm_track.plot([], [], '#45b7d1', lw=0, marker='.', ms=2, alpha=0.4, label='z noisy')
line_ae_x, = ax_arm_track.plot([], [], '#ff6b6b', lw=2.0, label='x est')
line_ae_y, = ax_arm_track.plot([], [], '#4ecdc4', lw=2.0, label='y est')
line_ae_z, = ax_arm_track.plot([], [], '#45b7d1', lw=2.0, label='z est')
line_aa_x, = ax_arm_track.plot([], [], '#ff6b6b', lw=0.8, alpha=0.3, label='x true')
line_aa_y, = ax_arm_track.plot([], [], '#4ecdc4', lw=0.8, alpha=0.3, label='y true')
line_aa_z, = ax_arm_track.plot([], [], '#45b7d1', lw=0.8, alpha=0.3, label='z true')
ax_arm_track.legend(loc='upper left', fontsize=6, labelcolor='white', framealpha=0.3, ncol=2)

ax_arm_obs.set_title('Arm EE — Estimator Innovation (meas − est)', color='white', fontsize=9, fontweight='bold')
ax_arm_obs.set_ylabel('Innovation (m)', color='white', fontsize=8)
ax_arm_obs.set_ylim(-0.08, 0.08)
ax_arm_obs.axhline(0, color='#555', lw=0.5)
line_ao_x, = ax_arm_obs.plot([], [], '#ff6b6b', lw=1.0, label='x innov')
line_ao_y, = ax_arm_obs.plot([], [], '#4ecdc4', lw=1.0, label='y innov')
line_ao_z, = ax_arm_obs.plot([], [], '#45b7d1', lw=1.0, label='z innov')
ax_arm_obs.legend(loc='upper left', fontsize=7, labelcolor='white', framealpha=0.3)

# ── Row 2: Control effort ──
ax_base_ctrl, ax_arm_ctrl = axes[2]
ax_base_ctrl.set_title('Base — LQR Control Effort (u = −Kx̂)', color='white', fontsize=9, fontweight='bold')
ax_base_ctrl.set_ylabel('Velocity (m/s, rad/s)', color='white', fontsize=8)
ax_base_ctrl.set_xlabel('Time (s)', color='white', fontsize=8)
ax_base_ctrl.set_ylim(-0.6, 0.6)
ax_base_ctrl.axhline(0, color='#555', lw=0.5)
line_bc_x, = ax_base_ctrl.plot([], [], '#ff6b6b', lw=1.5, label='vx')
line_bc_y, = ax_base_ctrl.plot([], [], '#4ecdc4', lw=1.5, label='vy')
line_bc_t, = ax_base_ctrl.plot([], [], '#ffe66d', lw=1.5, label='ω')
ax_base_ctrl.legend(loc='upper left', fontsize=7, labelcolor='white', framealpha=0.3)

ax_arm_ctrl.set_title('Arm — P-Control Effort (u = Kp·ê)', color='white', fontsize=9, fontweight='bold')
ax_arm_ctrl.set_ylabel('EE Velocity (m/s)', color='white', fontsize=8)
ax_arm_ctrl.set_xlabel('Time (s)', color='white', fontsize=8)
ax_arm_ctrl.set_ylim(-0.4, 0.4)
ax_arm_ctrl.axhline(0, color='#555', lw=0.5)
line_ac_x, = ax_arm_ctrl.plot([], [], '#ff6b6b', lw=1.5, label='vx')
line_ac_y, = ax_arm_ctrl.plot([], [], '#4ecdc4', lw=1.5, label='vy')
line_ac_z, = ax_arm_ctrl.plot([], [], '#45b7d1', lw=1.5, label='vz')
ax_arm_ctrl.legend(loc='upper left', fontsize=7, labelcolor='white', framealpha=0.3)

plt.tight_layout()
plt.ion()
plt.show(block=False)

# ── Simulation loop ─────────────────────────────────────────────────────────
capture_every = 4 if FAST else 2
grip_idx = 0

# Open MuJoCo viewer (only in live mode)
if not RENDER_GIF:
    viewer = mujoco.viewer.launch_passive(sim.engine.model, sim.engine.data)
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

    # ── Get true states ──
    true_base = sim.base.get_state()
    true_ee = arm_fallback.get_state()[:3]

    # ── Add sensor noise ──
    noisy_base = true_base + np.random.normal(
        [0, 0, 0], [NOISE_BASE_POS, NOISE_BASE_POS, NOISE_BASE_THETA]
    )
    noisy_ee = true_ee + np.random.normal(
        [0, 0, 0], [NOISE_ARM_POS, NOISE_ARM_POS, NOISE_ARM_POS]
    )

    # ── Base: Luenberger observer (uses previous control input) ──
    estimated_base = base_observer.estimate(
        noisy_base.reshape(-1, 1), base_vel.reshape(-1, 1)
    ).flatten()

    # ── Arm: EMA estimator ──
    arm_estimated_ee[0] = ALPHA_ARM * noisy_ee + (1 - ALPHA_ARM) * arm_estimated_ee[0]

    # ── Arm: P-controller using ESTIMATED state ──
    target_ee = ee_schedule[step]
    vel = 2.0 * (target_ee[:3] - arm_estimated_ee[0])
    vel = np.clip(vel, [-0.3, -0.3, -0.3], [0.3, 0.3, 0.3])
    joints = arm_fallback.step(np.concatenate([vel, [0, 0, 0]]))

    # Mirror to MuJoCo
    sim.engine.set_arm_ctrl(joints)
    if grip_idx < len(GRIP_SCHEDULE) and step >= GRIP_SCHEDULE[grip_idx][0]:
        jaw_pos = GRIP_SCHEDULE[grip_idx][1]
        ctrl = sim.engine.data.ctrl.copy()
        ctrl[8] = jaw_pos
        sim.engine.set_full_ctrl(ctrl)
        grip_idx += 1

    # ── Base: LQR using ESTIMATED state ──
    target_pose = base_schedule[step]
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
    log_arm_ref.append(target_ee[:3].copy())
    log_arm_actual.append(true_ee.copy())
    log_arm_noisy.append(noisy_ee.copy())
    log_arm_estimated.append(arm_estimated_ee[0].copy())
    log_arm_error.append((target_ee[:3] - true_ee).copy())
    log_arm_effort.append(vel[:3].copy())

    # ── Update plot every 5 steps ──
    if step % 5 == 0 and step > 0:
        time_arr = np.array(log_time)
        br_arr = np.array(log_base_ref)
        ba_arr = np.array(log_base_actual)
        bn_arr = np.array(log_base_noisy)
        be_arr = np.array(log_base_estimated)
        bo_arr = be_arr - ba_arr  # innovation = estimated - true (observer drives to zero)
        bc_arr = np.array(log_base_effort)
        ar_arr = np.array(log_arm_ref)
        aa_arr = np.array(log_arm_actual)
        an_arr = np.array(log_arm_noisy)
        ae_arr = np.array(log_arm_estimated)
        ao_arr = ae_arr - aa_arr  # innovation = estimated - true
        ac_arr = np.array(log_arm_effort)

        # Base tracking: ref, noisy, estimated, true
        line_br_x.set_data(time_arr, br_arr[:, 0])
        line_br_y.set_data(time_arr, br_arr[:, 1])
        line_br_t.set_data(time_arr, br_arr[:, 2])
        ax_base_track.relim()
        ax_base_track.autoscale_view()

        line_bn_x.set_data(time_arr, bn_arr[:, 0])
        line_bn_y.set_data(time_arr, bn_arr[:, 1])
        line_bn_t.set_data(time_arr, bn_arr[:, 2])
        ax_base_track.relim()
        ax_base_track.autoscale_view()

        line_be_x.set_data(time_arr, be_arr[:, 0])
        line_be_y.set_data(time_arr, be_arr[:, 1])
        line_be_t.set_data(time_arr, be_arr[:, 2])
        ax_base_track.relim()
        ax_base_track.autoscale_view()

        line_ba_x.set_data(time_arr, ba_arr[:, 0])
        line_ba_y.set_data(time_arr, ba_arr[:, 1])
        line_ba_t.set_data(time_arr, ba_arr[:, 2])
        ax_base_track.relim()
        ax_base_track.autoscale_view()

        # Base observer innovation
        line_bo_x.set_data(time_arr, bo_arr[:, 0])
        line_bo_y.set_data(time_arr, bo_arr[:, 1])
        line_bo_t.set_data(time_arr, bo_arr[:, 2])
        ax_base_obs.relim()
        ax_base_obs.autoscale_view()

        # Arm tracking: ref, noisy, estimated, true
        line_ar_x.set_data(time_arr, ar_arr[:, 0])
        line_ar_y.set_data(time_arr, ar_arr[:, 1])
        line_ar_z.set_data(time_arr, ar_arr[:, 2])
        ax_arm_track.relim()
        ax_arm_track.autoscale_view()

        line_an_x.set_data(time_arr, an_arr[:, 0])
        line_an_y.set_data(time_arr, an_arr[:, 1])
        line_an_z.set_data(time_arr, an_arr[:, 2])
        ax_arm_track.relim()
        ax_arm_track.autoscale_view()

        line_ae_x.set_data(time_arr, ae_arr[:, 0])
        line_ae_y.set_data(time_arr, ae_arr[:, 1])
        line_ae_z.set_data(time_arr, ae_arr[:, 2])
        ax_arm_track.relim()
        ax_arm_track.autoscale_view()

        line_aa_x.set_data(time_arr, aa_arr[:, 0])
        line_aa_y.set_data(time_arr, aa_arr[:, 1])
        line_aa_z.set_data(time_arr, aa_arr[:, 2])
        ax_arm_track.relim()
        ax_arm_track.autoscale_view()

        # Arm estimator innovation
        line_ao_x.set_data(time_arr, ao_arr[:, 0])
        line_ao_y.set_data(time_arr, ao_arr[:, 1])
        line_ao_z.set_data(time_arr, ao_arr[:, 2])
        ax_arm_obs.relim()
        ax_arm_obs.autoscale_view()

        # Base control effort
        line_bc_x.set_data(time_arr, bc_arr[:, 0])
        line_bc_y.set_data(time_arr, bc_arr[:, 1])
        line_bc_t.set_data(time_arr, bc_arr[:, 2])
        ax_base_ctrl.relim()
        ax_base_ctrl.autoscale_view()

        # Arm control effort
        line_ac_x.set_data(time_arr, ac_arr[:, 0])
        line_ac_y.set_data(time_arr, ac_arr[:, 1])
        line_ac_z.set_data(time_arr, ac_arr[:, 2])
        ax_arm_ctrl.relim()
        ax_arm_ctrl.autoscale_view()

        fig.canvas.draw()
        if not RENDER_GIF:
            fig.canvas.flush_events()

    # ── Sync viewer ──
    if viewer is not None:
        viewer.sync()
        time.sleep(sim.engine.dt / 4)

        if not viewer.is_running():
            break

    # ── Capture frame for GIF ──
    if RENDER_GIF and step % capture_every == 0:
        renderer.update_scene(sim.engine.data, camera)
        frame = renderer.render()

        fig.canvas.draw()
        plot_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        plot_img = plot_img[:, :, :3]

        plot_h = frame.shape[0]
        plot_w = int(plot_img.shape[1] * plot_h / plot_img.shape[0])
        y_ratio = plot_img.shape[0] / plot_h
        x_ratio = plot_img.shape[1] / plot_w
        y_idx = (np.arange(plot_h) * y_ratio).astype(int)
        x_idx = (np.arange(plot_w) * x_ratio).astype(int)
        plot_resized = plot_img[y_idx[:, None], x_idx]

        combined = np.hstack([frame, plot_resized])
        frames.append(combined)

if viewer is not None:
    viewer.close()

if RENDER_GIF and frames:
    import imageio.v3 as iio
    iio.imwrite(
        OUTPUT_PATH, frames,
        fps=50 // capture_every,
        loop=0,
        plugin='pillow',
        optimize=True,
    )
    print(f"✅ GIF saved: {OUTPUT_PATH}")
    print(f"   {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}, {50 // capture_every} fps")
    file_size = Path(OUTPUT_PATH).stat().st_size
    print(f"   File size: {file_size / 1024:.0f} KB")

if renderer:
    renderer.close()
plt.ioff()
plt.close(fig)
print("✅ Demo complete")
