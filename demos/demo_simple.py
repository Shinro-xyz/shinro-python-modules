# FILE: demos/demo_simple.py
"""
Simple demo: terminal output, no graphs, no live viewer.

Usage:
  python -m demos.demo_simple              # print state to terminal
  python -m demos.demo_simple --gif         # + capture GIF
"""
import os
import sys

RENDER_GIF = "--gif" in sys.argv
if RENDER_GIF:
    os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import mujoco
import numpy as np

from factories import ControllerFactory, EstimatorFactory, TrajectoryFactory
from simulation import RobotSim

HERE = Path(__file__).parent.parent


def phase_arm_trajectory(sim):
    print("=== Phase 1: Arm trajectory ===")
    arm_schedule = TrajectoryFactory(str(HERE / "configs/trajectories/arm_extension.toml")).create()
    arm_joints = sim.config["joint_groups"]["arm_joints"]
    ee_home = sim.arm.get_state()[:3].copy()
    for step, offset in enumerate(arm_schedule):
        target_ee = ee_home + offset
        joints = sim.arm.engine_ik(target_ee)
        for name, val in zip(arm_joints, joints):
            sim.engine.set_joint_ctrl(name, val)
        sim.engine.step()
        if step % 50 == 0:
            ee = sim.arm.get_state()[:3]
            print(f"  step {step:3d}: EE = ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")


def phase_base_tracking(sim):
    print("=== Phase 2: Base tracking with LQR + observer ===")
    ctrl = ControllerFactory(str(HERE / "configs/controllers/lqr_base.toml")).create()
    observer = EstimatorFactory(str(HERE / "configs/estimators/luenberger_base.toml")).create()
    schedule = TrajectoryFactory(str(HERE / "configs/trajectories/base_straight.toml")).create()

    base_vel = np.zeros(3)
    for step in range(len(schedule)):
        true_base = sim.base.get_state()
        noisy = true_base + np.random.normal(0, 0.02, 3)
        estimated = observer.estimate(noisy.reshape(-1, 1), base_vel.reshape(-1, 1)).flatten()
        base_vel = ctrl.compute(estimated, schedule[step])
        base_vel = np.clip(base_vel, [-0.5, -0.5, -1.0], [0.5, 0.5, 1.0])
        sim.base.step(base_vel)
        sim.step()
        if step % 50 == 0:
            print(f"  step {step:3d}: base = ({true_base[0]:.3f}, {true_base[1]:.3f}, {true_base[2]:.3f})")


def capture_gif(sim):
    print("=== Capturing GIF ===")
    renderer = mujoco.Renderer(sim.engine.model, width=320, height=240)
    camera = mujoco.MjvCamera()
    camera.distance = 2.0
    camera.azimuth = 135
    camera.elevation = -20
    frames = []

    sim.reset()
    arm_joints = sim.config["joint_groups"]["arm_joints"]
    for i in range(100):
        theta = np.sin(i * 0.05)
        arm_target = np.array([theta * 0.3, -0.2 * abs(theta), 0.3 * theta, 0.0, 0.0, 0.0])
        sim.arm.step(arm_target)
        sim.base.step(np.array([0.2, 0.0, 0.0]))
        sim.step()
        if i % 2 == 0:
            renderer.update_scene(sim.engine.data, camera)
            frames.append(renderer.render())

    renderer.close()
    import imageio
    path = str(HERE / "simple_demo.gif")
    imageio.mimsave(path, frames, fps=15, loop=0)
    print(f"  GIF saved: simple_demo.gif ({len(frames)} frames)")


if __name__ == "__main__":
    sim = RobotSim(str(HERE / "robot_config.toml"))
    sim.reset()

    phase_arm_trajectory(sim)
    phase_base_tracking(sim)

    if RENDER_GIF:
        capture_gif(sim)

    print("Done.")
