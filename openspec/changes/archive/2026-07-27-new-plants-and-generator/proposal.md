## Why

The control suite currently has only two plants (ArmRobot, HolonomicMobileRobot), limiting the range of systems we can test and demonstrate. Adding benchmark plants (InvertedPendulum, CartPole) enables testing nonlinear controllers (NMPC, MPPI, sliding mode) and estimators (EKF, UKF) on well-known nonlinear systems. A generic XML-to-config generator makes it trivial to add new robots by dropping an MJCF/URDF file into the models directory.

## What Changes

- **New plants**: InvertedPendulum (2D state, 1D torque) and CartPole (4D state, 1D force) — both with standalone analytical dynamics and optional MuJoCo engine mode
- **Quadrotor placeholder**: Stub file marking the pattern for future implementation
- **Generic XML config generator**: Refactor `scripts/generate_robot_config.py` to auto-detect plant types from MJCF/URDF files via a detector registry, supporting single and combined robots
- **MuJoCo model files**: Minimal XML models for pendulum and cartpole in `models/`
- **Plant configs**: TOML configs for the new plants in `configs/plants/`
- **Tests**: Unit tests for standalone dynamics, engine mode, linearized models, state bounds, and config generation

## Capabilities

### New Capabilities
- `inverted-pendulum-plant`: 2D inverted pendulum with standalone analytical dynamics, MuJoCo engine mode, linearized model, and configurable damping/state bounds
- `cartpole-plant`: 4D cart-pole with coupled dynamics, MuJoCo engine mode, linearized model, configurable track limits
- `xml-config-generator`: Generic MJCF/URDF-to-TOML config generator with plant type auto-detection via registered detectors, supporting single and combined robots

### Modified Capabilities
- (none — existing plants unchanged)

## Impact

- **New files**: `plants/inverted_pendulum.py`, `plants/cartpole.py`, `plants/quadrotor.py`, `models/pendulum.xml`, `models/cartpole.xml`, `configs/plants/inverted_pendulum.toml`, `configs/plants/cartpole.toml`
- **Modified files**: `plants/__init__.py`, `scripts/generate_robot_config.py`, `tests/test_plants.py`
- **Dependencies**: None new — uses existing numpy, scipy, mujoco (optional)
- **Breaking**: None — all existing plants and APIs unchanged
