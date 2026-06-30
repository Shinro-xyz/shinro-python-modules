# FILE: capture_gif.py
"""
LeKiwi arm extension demo — EE-space P-controller + smooth trajectory.

Architecture:
  Base:  Fixed at origin (no control, no free joint)
  Arm:   P-controller in EE space → Jacobian → joint velocity → position servos
         Reference is a smooth cubic trajectory between waypoints

Usage:
  python capture_gif.py                          # live viewer + plot
  python capture_gif.py --gif                     # render to GIF (headless)
  python capture_gif.py --gif --fast              # render to GIF, fewer frames
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
OUTPUT_PATH = str(HERE / "lekiwi_arm_demo.gif")

from lekiwi_sim import MJCF_PATH, HERE as LEKIWI_HOME, MuJoCoEngine
from pid import PIDController


# ── Smooth trajectory (cubic interpolation between waypoints) ──────────────
# Waypoints as OFFSETS from home (dx, dy, dz)
WAYPOINT_OFFSETS = [
    (1.0,   np.array([0.0,   0.0,   0.0])),    # home — 1s
    (1.0,   np.array([0.0,   0.08,  0.0])),    # extend in y — 1s
    (1.0,   np.array([0.0,   0.16,  0.0])),    # extend further — 1s
    (1.0,   np.array([0.0,   0.24,  0.0])),    # max extension — 1s
    (1.0,   np.array([0.0,   0.16,  0.0])),    # retract — 1s
    (1.0,   np.array([0.0,   0.08,  0.0])),    # retract — 1s
    (1.0,   np.array([0.0,   0.0,   0.0])),    # back to home — 1s
]

# Build cubic trajectory
def build_cubic_trajectory(waypoints, dt=0.02):
    """
    Generate smooth position + velocity profile via cubic interpolation.

    Each segment between p_i and p_{i+1} over duration T:
        p(t) = a0 + a2*t^2 + a3*t^3   (zero velocity at both ends)
    """
    schedule_pos = []
    schedule_vel = []

    for i in range(len(waypoints) - 1):
        duration, p_start = waypoints[i]
        _, p_end = waypoints[i + 1]
        T = duration
        n_steps = int(np.round(T / dt))

        delta = p_end - p_start
        a0 = p_start
        a2 = 3.0 * delta / (T * T)
        a3 = -2.0 * delta / (T * T * T)

        for k in range(n_steps):
            t_local = k * dt
            pos = a0 + a2 * t_local**2 + a3 * t_local**3
            vel = 2.0 * a2 * t_local + 3.0 * a3 * t_local**2
            schedule_pos.append(pos.copy())
            schedule_vel.append(vel.copy())

    return np.array(schedule_pos), np.array(schedule_vel)


# ── Create sim ──────────────────────────────────────────────────────────────
with open(MJCF_PATH) as f:
    base_xml = f.read()

mesh_dir = LEKIWI_HOME / 'lekiwi-sim' / 'meshes'
assets = {}
for fname in mesh_dir.iterdir():
    if fname.suffix in ('.stl', '.obj'):
        assets[fname.name] = fname.read_bytes()

# Quick FK on a temp model to get EE home position
temp_model = mujoco.MjModel.from_xml_string(base_xml, assets)
temp_data = mujoco.MjData(temp_model)
temp_ee_id = mujoco.mj_name2id(temp_model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw_08d-v1")
if temp_ee_id == -1:
    temp_ee_id = temp_model.nbody - 1
mujoco.mj_forward(temp_model, temp_data)
ee_home = temp_data.xpos[temp_ee_id].copy()
print(f"EE home position (from MuJoCo): x={ee_home[0]:.3f}, y={ee_home[1]:.3f}, z={ee_home[2]:.3f}")

# Convert offset waypoints to absolute world-frame
waypoints_abs = [(d, ee_home + offset) for d, offset in WAYPOINT_OFFSETS]

# Build smooth trajectory
ee_ref_pos, ee_ref_vel = build_cubic_trajectory(waypoints_abs)
total_steps = len(ee_ref_pos)
print(f"Trajectory: {total_steps} steps ({total_steps * 0.02:.1f}s)")

# Create engine directly
engine = MuJoCoEngine(dt=0.02, xml_string=base_xml, assets=assets)
engine.reset()

# Cache arm indices
arm_qpos_slice = engine.arm_qpos_slice
arm_limits = engine.arm_limits
ee_body_id = mujoco.mj_name2id(engine.model, mujoco.mjtObj.mjOBJ_BODY, "Moving_Jaw_08d-v1")
if ee_body_id == -1:
    ee_body_id = engine.model.nbody - 1
arm_jac_start = 9 if engine.has_free_joint else 3

# ── PID controller for EE space ────────────────────────────────────────────
KP_EE = np.array([5.0, 5.0, 5.0])   # proportional gain
KI_EE = np.array([0.5, 0.5, 0.5])   # integral gain (eliminates steady-state error)
KD_EE = np.array([0.1, 0.1, 0.1])   # derivative gain (dampens overshoot)
ee_ctrl = PIDController(
    kp=KP_EE, ki=KI_EE, kd=KD_EE, dt=0.02,
    output_limits=(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
)
LAM_IK = 0.1   # Damping for Jacobian pseudoinverse
MAX_DQ = 0.5   # Max joint velocity per step (rad/step)


# ── Data logging ───────────────────────────────────────────────────────────
log_time = []
log_ee_ref = []       # reference EE [x, y, z]
log_ee_actual = []    # true EE [x, y, z]
log_ee_error = []     # tracking error = ref - true
log_ee_ctrl = []      # controller output (EE velocity command)
log_joints = []       # joint positions [q1..q6]


# ── Plot setup ─────────────────────────────────────────────────────────────
if RENDER_GIF:
    renderer = mujoco.Renderer(engine.model, width=400, height=300)
    camera = mujoco.MjvCamera()
    camera.distance = 1.5
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

# ── Top: EE tracking ──
ax_ee_track = axes[0]
ax_ee_track.set_title('Arm EE — Ref / True (P-controller: u = Kp · e)', color='white', fontsize=9, fontweight='bold')
ax_ee_track.set_ylabel('Position (m)', color='white', fontsize=8)
ax_ee_track.set_ylim(-0.1, 0.4)
line_er_x, = ax_ee_track.plot([], [], '#ff6b6b', lw=1.5, ls='--', label='x ref')
line_er_y, = ax_ee_track.plot([], [], '#4ecdc4', lw=1.5, ls='--', label='y ref')
line_er_z, = ax_ee_track.plot([], [], '#45b7d1', lw=1.5, ls='--', label='z ref')
line_ea_x, = ax_ee_track.plot([], [], '#ff6b6b', lw=1.0, alpha=0.6, label='x true')
line_ea_y, = ax_ee_track.plot([], [], '#4ecdc4', lw=1.0, alpha=0.6, label='y true')
line_ea_z, = ax_ee_track.plot([], [], '#45b7d1', lw=1.0, alpha=0.6, label='z true')
ax_ee_track.legend(loc='upper left', fontsize=6, labelcolor='white', framealpha=0.3, ncol=2)

# ── Middle: Tracking error + control effort ──
ax_error = axes[1]
ax_error.set_title('Arm EE — Tracking Error (ref − true)', color='white', fontsize=9, fontweight='bold')
ax_error.set_ylabel('Error (m)', color='white', fontsize=8)
ax_error.set_ylim(-0.1, 0.1)
ax_error.axhline(0, color='#555', lw=0.5)
line_ex_x, = ax_error.plot([], [], '#ff6b6b', lw=1.0, label='x err')
line_ex_y, = ax_error.plot([], [], '#4ecdc4', lw=1.0, label='y err')
line_ex_z, = ax_error.plot([], [], '#45b7d1', lw=1.0, label='z err')
ax_error.legend(loc='upper left', fontsize=7, labelcolor='white', framealpha=0.3)

# ── Bottom: Joint positions ──
ax_joints = axes[2]
ax_joints.set_title('Arm — Joint Positions', color='white', fontsize=9, fontweight='bold')
ax_joints.set_ylabel('Joint angle (rad)', color='white', fontsize=8)
ax_joints.set_xlabel('Time (s)', color='white', fontsize=8)
ax_joints.set_ylim(-1.5, 1.5)
ax_joints.axhline(0, color='#555', lw=0.5)
joint_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffe66d', '#c792ea', '#f78c6c']
joint_names = ['Rotation', 'Pitch', 'Elbow', 'Wrist_Pitch', 'Wrist_Roll', 'Jaw']
joint_lines = []
for i in range(6):
    line, = ax_joints.plot([], [], joint_colors[i], lw=1.0, label=joint_names[i])
    joint_lines.append(line)
ax_joints.legend(loc='upper left', fontsize=6, labelcolor='white', framealpha=0.3, ncol=2)

plt.tight_layout()
plt.ion()
plt.show(block=False)

# ── Simulation loop ─────────────────────────────────────────────────────────
capture_every = 4 if FAST else 2

# Open MuJoCo viewer (only in live mode)
if not RENDER_GIF:
    viewer = mujoco.viewer.launch_passive(engine.model, engine.data)
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = [0.0, 0.0, 0.1]
else:
    viewer = None

# Initialize joint state
current_joints = engine.get_arm_qpos().copy()

for step in range(total_steps):
    t = step * 0.02

    # ── Get true EE position ──
    true_ee = engine.data.xpos[ee_body_id].copy()

    # ── Reference from smooth trajectory ──
    target_ee = ee_ref_pos[step]

    # ── PID controller in EE space ──
    ee_error = target_ee - true_ee
    ee_vel_cmd = ee_ctrl.compute(true_ee, target_ee)  # PID output (m/s)

    # ── Map EE velocity to joint velocity via Jacobian ──
    jacp = np.zeros((3, engine.model.nv))
    jacr = np.zeros((3, engine.model.nv))
    mujoco.mj_jac(engine.model, engine.data, jacp, jacr,
                  engine.data.xpos[ee_body_id], ee_body_id)
    J = jacp[:, arm_jac_start:arm_jac_start + 6]

    # Damped least-squares: dq = J^T (J J^T + λ²I)⁻¹ ee_vel
    JJT = J @ J.T
    dq = J.T @ np.linalg.solve(JJT + LAM_IK**2 * np.eye(3), ee_vel_cmd)
    dq = np.clip(dq, -MAX_DQ, MAX_DQ)

    # Integrate to joint targets
    current_joints = current_joints + dq
    current_joints = np.clip(current_joints, arm_limits[:, 0], arm_limits[:, 1])

    # Send to MuJoCo position servos
    engine.set_arm_ctrl(current_joints)

    # ── Step physics ──
    engine.step()

    # ── Log data ──
    log_time.append(t)
    log_ee_ref.append(target_ee.copy())
    log_ee_actual.append(true_ee.copy())
    log_ee_error.append(ee_error.copy())
    log_ee_ctrl.append(ee_vel_cmd.copy())
    log_joints.append(engine.get_arm_qpos().copy())

    # ── Update plot every 5 steps ──
    if step % 5 == 0 and step > 0:
        time_arr = np.array(log_time)
        ref_arr = np.array(log_ee_ref)
        act_arr = np.array(log_ee_actual)
        err_arr = np.array(log_ee_error)

        # EE tracking
        line_er_x.set_data(time_arr, ref_arr[:, 0])
        line_er_y.set_data(time_arr, ref_arr[:, 1])
        line_er_z.set_data(time_arr, ref_arr[:, 2])
        line_ea_x.set_data(time_arr, act_arr[:, 0])
        line_ea_y.set_data(time_arr, act_arr[:, 1])
        line_ea_z.set_data(time_arr, act_arr[:, 2])
        ax_ee_track.relim()
        ax_ee_track.autoscale_view()

        # Tracking error
        line_ex_x.set_data(time_arr, err_arr[:, 0])
        line_ex_y.set_data(time_arr, err_arr[:, 1])
        line_ex_z.set_data(time_arr, err_arr[:, 2])
        ax_error.relim()
        ax_error.autoscale_view()

        # Joint positions
        jnt_arr = np.array(log_joints)
        for i in range(6):
            joint_lines[i].set_data(time_arr, jnt_arr[:, i])
        ax_joints.relim()
        ax_joints.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

    # ── Capture frame for GIF ──
    if RENDER_GIF and step % capture_every == 0:
        renderer.update_scene(engine.data, camera)
        frames.append(renderer.render())

    # ── Sync viewer ──
    if viewer is not None:
        viewer.sync()
        time.sleep(engine.dt / 4)
        if not viewer.is_running():
            break

# ── Save GIF ────────────────────────────────────────────────────────────────
if RENDER_GIF and frames:
    import imageio
    fps = 12 if not FAST else 6
    imageio.mimsave(OUTPUT_PATH, frames, fps=fps, loop=0)
    print(f"✅ GIF saved to {OUTPUT_PATH} ({len(frames)} frames, {fps} fps)")

print("✅ Arm extension demo complete")
