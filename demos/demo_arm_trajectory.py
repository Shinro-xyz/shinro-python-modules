# FILE: demos/demo_arm_trajectory.py
"""
Arm-only demo: smooth EE trajectory via MuJoCo Jacobian IK.

Usage:
  python demos/demo_arm_trajectory.py              # live viewer + plot
  python demos/demo_arm_trajectory.py --gif         # render to GIF (headless)
  python demos/demo_arm_trajectory.py --gif --fast  # faster GIF
"""
import os
import sys

RENDER_GIF = "--gif" in sys.argv
FAST = "--fast" in sys.argv

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

from lekiwi_sim import RobotSim
from factories import TrajectoryFactory

HERE = Path(__file__).parent.parent
OUTPUT_PATH = str(HERE / "lekiwi_arm_demo.gif")
CONFIG_PATH = str(HERE / "robot_config.yaml")

sim = RobotSim(CONFIG_PATH)
sim.reset()

arm_joint_names = sim.config["joint_groups"]["arm_joints"]
ee_home = sim.arm.get_state()[:3].copy()
print(f"EE home position: x={ee_home[0]:.3f}, y={ee_home[1]:.3f}, z={ee_home[2]:.3f}")

offset_schedule = TrajectoryFactory(str(HERE / "configs/trajectories/arm_extension.yaml")).create()
ee_ref_pos = ee_home + offset_schedule
total_steps = len(ee_ref_pos)
print(f"Trajectory: {total_steps} steps ({total_steps * 0.02:.1f}s)")

log_time = []
log_ee_ref = []
log_ee_actual = []
log_ee_error = []
log_joints = []

GIF_WIDTH = 640
GIF_HEIGHT = 400

if RENDER_GIF:
    renderer = mujoco.Renderer(sim.engine.model, width=GIF_WIDTH, height=GIF_HEIGHT)
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
from demos.helpers import setup_dark_plot
setup_dark_plot(fig, axes)

ax_ee_track = axes[0]
ax_ee_track.set_title('Arm EE — Ref / True', color='white', fontsize=9, fontweight='bold')
ax_ee_track.set_ylabel('Position (m)', color='white', fontsize=8)
ax_ee_track.set_ylim(-0.1, 0.4)
line_er_x, = ax_ee_track.plot([], [], '#ff6b6b', lw=1.5, ls='--', label='x ref')
line_er_y, = ax_ee_track.plot([], [], '#4ecdc4', lw=1.5, ls='--', label='y ref')
line_er_z, = ax_ee_track.plot([], [], '#45b7d1', lw=1.5, ls='--', label='z ref')
line_ea_x, = ax_ee_track.plot([], [], '#ff6b6b', lw=1.0, alpha=0.6, label='x true')
line_ea_y, = ax_ee_track.plot([], [], '#4ecdc4', lw=1.0, alpha=0.6, label='y true')
line_ea_z, = ax_ee_track.plot([], [], '#45b7d1', lw=1.0, alpha=0.6, label='z true')
ax_ee_track.legend(loc='upper left', fontsize=6, labelcolor='white', framealpha=0.3, ncol=2)

ax_error = axes[1]
ax_error.set_title('Tracking Error (ref − true)', color='white', fontsize=9, fontweight='bold')
ax_error.set_ylabel('Error (m)', color='white', fontsize=8)
ax_error.set_ylim(-0.1, 0.1)
ax_error.axhline(0, color='#555', lw=0.5)
line_ex_x, = ax_error.plot([], [], '#ff6b6b', lw=1.0, label='x err')
line_ex_y, = ax_error.plot([], [], '#4ecdc4', lw=1.0, label='y err')
line_ex_z, = ax_error.plot([], [], '#45b7d1', lw=1.0, label='z err')
ax_error.legend(loc='upper left', fontsize=7, labelcolor='white', framealpha=0.3)

ax_joints = axes[2]
ax_joints.set_title('Joint Positions', color='white', fontsize=9, fontweight='bold')
ax_joints.set_ylabel('Angle (rad)', color='white', fontsize=8)
ax_joints.set_xlabel('Time (s)', color='white', fontsize=8)
ax_joints.set_ylim(-1.5, 1.5)
ax_joints.axhline(0, color='#555', lw=0.5)
joint_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#ffe66d', '#c792ea', '#f78c6c']
joint_lines = []
for i in range(6):
    line, = ax_joints.plot([], [], joint_colors[i], lw=1.0, label=arm_joint_names[i])
    joint_lines.append(line)
ax_joints.legend(loc='upper left', fontsize=6, labelcolor='white', framealpha=0.3, ncol=2)

plt.tight_layout()
plt.ion()
plt.show(block=False)

capture_every = 4 if FAST else 2

if not RENDER_GIF:
    viewer = mujoco.viewer.launch_passive(sim.engine.model, sim.engine.data)
    viewer.cam.distance = 1.5
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -20
    viewer.cam.lookat[:] = [0.0, 0.0, 0.1]
else:
    viewer = None

for step in range(total_steps):
    t = step * 0.02

    true_ee = sim.arm.get_state()[:3]
    target_ee = ee_ref_pos[step]

    joint_targets = sim.arm.engine_ik(target_ee)
    for name, val in zip(arm_joint_names, joint_targets):
        sim.engine.set_joint_ctrl(name, val)

    sim.engine.step()

    log_time.append(t)
    log_ee_ref.append(target_ee.copy())
    log_ee_actual.append(true_ee.copy())
    log_ee_error.append((target_ee - true_ee).copy())
    log_joints.append(np.array([sim.engine.get_joint_qpos(n) for n in arm_joint_names]).copy())

    if step % 5 == 0 and step > 0:
        time_arr = np.array(log_time)
        ref_arr = np.array(log_ee_ref)
        act_arr = np.array(log_ee_actual)
        err_arr = np.array(log_ee_error)

        line_er_x.set_data(time_arr, ref_arr[:, 0])
        line_er_y.set_data(time_arr, ref_arr[:, 1])
        line_er_z.set_data(time_arr, ref_arr[:, 2])
        line_ea_x.set_data(time_arr, act_arr[:, 0])
        line_ea_y.set_data(time_arr, act_arr[:, 1])
        line_ea_z.set_data(time_arr, act_arr[:, 2])
        ax_ee_track.relim()
        ax_ee_track.autoscale_view()

        line_ex_x.set_data(time_arr, err_arr[:, 0])
        line_ex_y.set_data(time_arr, err_arr[:, 1])
        line_ex_z.set_data(time_arr, err_arr[:, 2])
        ax_error.relim()
        ax_error.autoscale_view()

        jnt_arr = np.array(log_joints)
        for i in range(6):
            joint_lines[i].set_data(time_arr, jnt_arr[:, i])
        ax_joints.relim()
        ax_joints.autoscale_view()

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

print("✅ Arm extension demo complete")