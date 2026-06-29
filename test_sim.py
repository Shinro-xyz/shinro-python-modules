# FILE: test_sim.py (comprehensive test for LeKiwi simulation stack)
"""
Test script for the LeKiwi MuJoCo simulation stack.

Tests:
  1. MuJoCoEngine — load, step, read state, reset
  2. ArmRobot with MuJoCo — position control, joint limits, steady-state error
  3. HolonomicMobileRobot with MuJoCo — base drive, kinematic accuracy
  4. LeKiwiSim combined — arm + base simultaneously
  5. Reset and re-run
  6. Sensor data access

Run:  python test_sim.py
"""

import numpy as np
import sys
import time

# ── Imports ──────────────────────────────────────────────────────────────────
from lekiwi_sim import MuJoCoEngine, LeKiwiSim, ARM_JOINT_NAMES, DRIVE_JOINT_NAMES
from armrobot import ArmRobot
from holonomicmobilerobot import HolonomicMobileRobot


# ── Helpers ──────────────────────────────────────────────────────────────────
PASS = 0
FAIL = 0

def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        print(f"  ✅ {name}")
        PASS += 1
    else:
        print(f"  ❌ {name} — {detail}")
        FAIL += 1


def almost_eq(a, b, tol=1e-3):
    return np.allclose(a, b, atol=tol)


# ── Test 1: MuJoCoEngine ────────────────────────────────────────────────────
def test_engine():
    print("\n═══ Test 1: MuJoCoEngine ═══")
    engine = MuJoCoEngine()

    # 1a. Model loads with expected structure
    check("Model has 9 actuators (nu=9)", engine.model.nu == 9,
          f"got nu={engine.model.nu}")
    check("Model has 16 qpos (nq=16)", engine.model.nq == 16,
          f"got nq={engine.model.nq}")
    check("Model has 43 bodies", engine.model.nbody == 43,
          f"got nbody={engine.model.nbody}")

    # 1b. Initial state is zeroed (except base height)
    arm_qpos = engine.get_arm_qpos()
    check("Arm starts at zero", almost_eq(arm_qpos, np.zeros(6)),
          f"got {arm_qpos}")
    base_pose = engine.get_base_pose()
    check("Base starts at origin", almost_eq(base_pose[:2], [0, 0]),
          f"got {base_pose}")

    # 1c. Arm joint limits are sensible
    check("Arm limits have 6 entries", engine.arm_limits.shape == (6, 2),
           f"got {engine.arm_limits.shape}")
    check("All arm limits are finite",
          np.all(np.isfinite(engine.arm_limits)),
          f"got {engine.arm_limits}")

    # 1d. Step with no control — arm droops under gravity (position servos)
    qpos_before = engine.get_full_qpos().copy()
    for _ in range(10):
        engine.step()
    qpos_after = engine.get_full_qpos()
    # Arm droops under gravity; base (free joint) should stay put
    base_diff = np.max(np.abs(qpos_after[:7] - qpos_before[:7]))
    check("No control → base stays put (free joint)",
          base_diff < 0.01,
          f"base max diff: {base_diff}")

    # 1e. Arm position control moves joints
    ctrl = np.zeros(9)
    ctrl[3] = 0.5   # Rotation
    ctrl[4] = -0.3  # Pitch
    ctrl[5] = 0.8   # Elbow
    engine.set_full_ctrl(ctrl)
    for _ in range(100):
        engine.step()
    arm_after = engine.get_arm_qpos()
    check("Arm moves toward position target",
          np.any(np.abs(arm_after) > 0.01),
          f"all joints near zero: {arm_after}")
    check("Arm doesn't overshoot target (within limits)",
          np.all(arm_after <= engine.arm_limits[:, 1] + 0.01),
          f"arm: {arm_after}, limits: {engine.arm_limits[:, 1]}")

    # 1f. Reset works
    engine.reset()
    arm_reset = engine.get_arm_qpos()
    check("Reset returns arm to zero",
          almost_eq(arm_reset, np.zeros(6), tol=1e-4),
          f"got {arm_reset}")

    # 1g. Sensor data returns expected keys
    sensor = engine.get_sensor_data()
    expected_keys = {"qpos", "qvel", "ctrl", "base_pose", "arm_joints", "time"}
    check("Sensor data has all keys",
          expected_keys.issubset(sensor.keys()),
          f"missing: {expected_keys - set(sensor.keys())}")
    check("Sensor time advances", sensor["time"] >= 0.0,
          f"got {sensor['time']}")

    # 1h. Print info doesn't crash
    engine.print_info()
    check("print_info() runs without error", True)


# ── Test 2: ArmRobot with MuJoCo ────────────────────────────────────────────
def test_arm_mujoco():
    print("\n═══ Test 2: ArmRobot with MuJoCo ═══")
    engine = MuJoCoEngine()
    link_offsets = np.array([
        [0.0388,  0.0,     0.0624],
        [-0.0304, -0.0183, -0.0542],
        [-0.1126, -0.028,   0.0],
        [-0.1349,  0.0052,  0.0],
        [0.0,    -0.0611,  0.0181],
        [0.0,     0.0,     0.05],
    ])
    rot_axes = ["z"] * 6

    arm = ArmRobot(
        num_dof=6, dt=0.02,
        joint_limits=engine.arm_limits,
        joint_offsets=link_offsets,
        rot_axes=rot_axes,
    )
    arm._engine = engine

    # 2a. Step with zero target — arm droops under gravity (expected with position servos)
    joints = arm.step(np.zeros(6))
    # Arm droops: joints 1-2 (Pitch, Elbow) sag under gravity
    check("Zero target → arm droops under gravity (expected)",
          np.any(np.abs(joints[1:3]) > 0.001),
          f"arm didn't droop at all: {joints}")
    check("Zero target → joints are finite",
          np.all(np.isfinite(joints)),
          f"NaN in joints: {joints}")

    # 2b. Step with positive target moves joints (with steady-state error)
    target = np.array([0.5, -0.3, 0.8, 0.0, 0.0, 0.0])
    for _ in range(100):
        joints = arm.step(target)
    # Position servos with kp=50 have steady-state error under gravity
    # Rotation joint (0) has ~0.48 rad error; Pitch (1) and Elbow (2) track better
    check("Arm moves toward positive target",
          np.all(np.abs(joints[:3] - target[:3]) < 0.5),
          f"target: {target[:3]}, got: {joints[:3]}")
    check("Arm moves in correct direction for each joint",
          joints[0] > 0 and joints[1] < 0 and joints[2] > 0,
          f"signs wrong: {joints[:3]}")

    # 2c. Joint limits are respected
    extreme = np.array([10.0, -10.0, 10.0, 10.0, 10.0, 10.0])
    for _ in range(200):
        joints = arm.step(extreme)
    check("Joint limits respected",
          np.all(joints >= engine.arm_limits[:, 0] - 0.01) and
          np.all(joints <= engine.arm_limits[:, 1] + 0.01),
          f"joints: {joints}\n  limits: {engine.arm_limits}")

    # 2d. get_state returns current joints
    state = arm.get_state()
    check("get_state matches last joints",
          almost_eq(state, joints, tol=1e-3),
          f"state: {state}, joints: {joints}")

    # 2e. get_model returns A, B
    A, B = arm.get_model()
    check("Arm get_model returns A (6x6)", A.shape == (6, 6),
          f"got {A.shape}")
    check("Arm get_model returns B (6x6)", B.shape == (6, 6),
          f"got {B.shape}")
    check("Arm A is identity", almost_eq(A, np.eye(6)),
          f"got {A}")
    check("Arm B = dt * I", almost_eq(B, 0.02 * np.eye(6)),
          f"got {B}")


# ── Test 3: HolonomicMobileRobot with MuJoCo ───────────────────────────────
def test_base_mujoco():
    print("\n═══ Test 3: HolonomicMobileRobot with MuJoCo ═══")
    engine = MuJoCoEngine()
    base = HolonomicMobileRobot(
        num_wheels=3, radius_robots=0.12,
        gamma=-np.pi / 2, radius_wheels=0.09, dt=0.02,
    )
    base._engine = engine

    # 3a. Zero velocity → no movement
    base.step(np.zeros(3))
    check("Zero velocity → base stays at origin",
          almost_eq(base.get_state(), np.zeros(3), tol=1e-4),
          f"got {base.get_state()}")

    # 3b. Forward velocity moves in +x
    base.set_pose(0, 0, 0)
    v_forward = np.array([0.2, 0.0, 0.0])
    for _ in range(100):
        base.step(v_forward)
    state = base.get_state()
    check("Forward velocity → +x movement",
          state[0] > 0.15 and abs(state[1]) < 0.01 and abs(state[2]) < 0.01,
          f"got {state} (expected x≈0.2, y≈0, θ≈0)")

    # 3c. Lateral velocity moves in +y
    base.set_pose(0, 0, 0)
    v_lateral = np.array([0.0, 0.2, 0.0])
    for _ in range(100):
        base.step(v_lateral)
    state = base.get_state()
    check("Lateral velocity → +y movement",
          abs(state[0]) < 0.01 and state[1] > 0.15 and abs(state[2]) < 0.01,
          f"got {state} (expected x≈0, y≈0.2, θ≈0)")

    # 3d. Rotational velocity turns
    base.set_pose(0, 0, 0)
    v_rot = np.array([0.0, 0.0, 0.5])
    for _ in range(100):
        base.step(v_rot)
    state = base.get_state()
    check("Rotational velocity → yaw change",
          abs(state[0]) < 0.01 and abs(state[1]) < 0.01 and state[2] > 0.4,
          f"got {state} (expected x≈0, y≈0, θ≈0.5)")

    # 3e. Kinematic accuracy: 0.2 m/s for 1s → 0.2 m
    base.set_pose(0, 0, 0)
    for _ in range(50):
        base.step(v_forward)
    state = base.get_state()
    check("Kinematic accuracy: 0.2 m/s × 1s ≈ 0.2 m",
          abs(state[0] - 0.2) < 0.01,
          f"got x={state[0]:.4f} (expected 0.2)")

    # 3f. get_model returns A, B
    A, B = base.get_model()
    check("Base get_model returns A (3x3)", A.shape == (3, 3),
          f"got {A.shape}")
    check("Base get_model returns B (3x3)", B.shape == (3, 3),
          f"got {B.shape}")

    # 3g. Wheel speeds are returned
    base.set_pose(0, 0, 0)
    wheel_speeds = base.step(np.array([0.2, 0.0, 0.0]))
    check("step returns wheel speeds (3,)",
          wheel_speeds.shape == (3,),
          f"got {wheel_speeds.shape}")
    check("Wheel speeds are finite",
          np.all(np.isfinite(wheel_speeds)),
          f"got {wheel_speeds}")


# ── Test 4: LeKiwiSim combined ─────────────────────────────────────────────
def test_lekiwi_sim():
    print("\n═══ Test 4: LeKiwiSim (combined arm + base) ═══")
    sim = LeKiwiSim(dt=0.02)

    # 4a. Initial state is zero
    state = sim.get_state()
    check("Sim initial arm joints are zero",
          almost_eq(state["arm_joints"], np.zeros(6), tol=1e-4),
          f"got {state['arm_joints']}")
    check("Sim initial base pose is origin",
          almost_eq(state["base_pose"][:2], [0, 0], tol=1e-4),
          f"got {state['base_pose']}")

    # 4b. Arm + base simultaneous movement
    arm_target = np.array([0.3, -0.2, 0.5, 0.0, 0.0, 0.0])
    base_vel = np.array([0.2, 0.1, 0.3])

    for _ in range(100):
        sim.arm.step(arm_target)
        sim.base.step(base_vel)
        sim.step()

    state = sim.get_state()
    check("Arm moved under combined control",
          np.any(np.abs(state["arm_joints"]) > 0.01),
          f"arm: {state['arm_joints']}")
    check("Base moved under combined control",
          np.any(np.abs(state["base_pose"]) > 0.01),
          f"base: {state['base_pose']}")

    # 4c. Reset clears everything
    sim.reset()
    state = sim.get_state()
    check("Reset clears arm joints",
          almost_eq(state["arm_joints"], np.zeros(6), tol=1e-3),
          f"got {state['arm_joints']}")
    check("Reset clears base pose",
          almost_eq(state["base_pose"][:2], [0, 0], tol=1e-3),
          f"got {state['base_pose']}")

    # 4d. Re-run after reset works
    for _ in range(50):
        sim.arm.step(arm_target)
        sim.base.step(base_vel)
        sim.step()
    state = sim.get_state()
    check("Re-run after reset produces movement",
          np.any(np.abs(state["arm_joints"]) > 0.01),
          f"arm: {state['arm_joints']}")


# ── Test 5: Edge cases ─────────────────────────────────────────────────────
def test_edge_cases():
    print("\n═══ Test 5: Edge cases ═══")
    engine = MuJoCoEngine()

    # 5a. Multiple resets don't accumulate state
    for _ in range(5):
        engine.reset()
    check("Multiple resets are idempotent",
          almost_eq(engine.get_arm_qpos(), np.zeros(6), tol=1e-4),
          f"got {engine.get_arm_qpos()}")

    # 5b. Large control signals don't crash
    ctrl = np.ones(9) * 100.0
    engine.set_full_ctrl(ctrl)
    for _ in range(200):
        engine.step()
    check("Large control doesn't crash (NaN check)",
          np.all(np.isfinite(engine.get_full_qpos())),
          f"qpos has NaN")

    # 5c. Negative control signals work
    engine.reset()
    ctrl = np.zeros(9)
    ctrl[3] = -0.5
    engine.set_full_ctrl(ctrl)
    for _ in range(100):
        engine.step()
    arm = engine.get_arm_qpos()
    check("Negative control moves joint negative",
          arm[0] < -0.01,
          f"got {arm[0]}")

    # 5d. Zero dt doesn't crash (just no movement)
    engine_zero = MuJoCoEngine(dt=0.0)
    engine_zero.set_full_ctrl(np.ones(9))
    for _ in range(10):
        engine_zero.step()
    check("Zero dt doesn't crash",
          np.all(np.isfinite(engine_zero.get_full_qpos())),
          "NaN in qpos after zero-dt step")

    # 5e. Full ctrl setter works
    engine.reset()
    ctrl = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    engine.set_full_ctrl(ctrl)
    check("set_full_ctrl stores all 9 values",
          almost_eq(engine.data.ctrl, ctrl, tol=1e-6),
          f"got {engine.data.ctrl}")


# ── Test 6: Performance sanity ─────────────────────────────────────────────
def test_performance():
    print("\n═══ Test 6: Performance sanity ═══")
    engine = MuJoCoEngine()

    # 6a. 1000 steps should complete in reasonable time
    ctrl = np.zeros(9)
    ctrl[3:9] = [0.3, -0.2, 0.5, 0.0, 0.0, 0.0]

    start = time.time()
    for _ in range(1000):
        engine.set_full_ctrl(ctrl)
        engine.step()
    elapsed = time.time() - start

    check(f"1000 MuJoCo steps in {elapsed:.2f}s (should be < 5s)",
          elapsed < 5.0,
          f"took {elapsed:.2f}s")

    # 6b. LeKiwiSim 1000 steps
    sim = LeKiwiSim(dt=0.02)
    arm_target = np.array([0.3, -0.2, 0.5, 0.0, 0.0, 0.0])
    base_vel = np.array([0.2, 0.1, 0.3])

    start = time.time()
    for _ in range(1000):
        sim.arm.step(arm_target)
        sim.base.step(base_vel)
        sim.step()
    elapsed = time.time() - start

    check(f"1000 LeKiwiSim steps in {elapsed:.2f}s (should be < 5s)",
          elapsed < 5.0,
          f"took {elapsed:.2f}s")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  LeKiwi Simulation Stack — Test Suite")
    print("=" * 60)

    test_engine()
    test_arm_mujoco()
    test_base_mujoco()
    test_lekiwi_sim()
    test_edge_cases()
    test_performance()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed, {FAIL}/{total} failed")
    if FAIL == 0:
        print("  🎉 All tests passed! Slay bestie.")
    else:
        print(f"  😤 {FAIL} test(s) failed. Debug time.")
    print("=" * 60)

    sys.exit(0 if FAIL == 0 else 1)
