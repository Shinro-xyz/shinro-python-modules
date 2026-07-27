## 1. MuJoCo XML Models

- [x] 1.1 Create `models/pendulum.xml` — hinge joint, capsule rod + sphere mass, torque actuator
- [x] 1.2 Create `models/cartpole.xml` — slider + hinge, box cart + capsule pole, force actuator

## 2. InvertedPendulum Plant

- [x] 2.1 Create `plants/inverted_pendulum.py` with `@register_plant("InvertedPendulum")`
- [x] 2.2 Implement `__init__` with configurable mass, length, damping, gravity, dt, state_bounds
- [x] 2.3 Implement standalone `step(u)` with semi-implicit Euler integration
- [x] 2.4 Implement `dynamics(x, u)` returning continuous-time dx/dt as flat (2,) array
- [x] 2.5 Implement `get_model()` returning linearized A (2,2) and B (2,1) around upright
- [x] 2.6 Implement `physics_engine(engine)` for MuJoCo attachment
- [x] 2.7 Implement engine-mode `step(u)` delegating to MuJoCo
- [x] 2.8 Implement `get_state()` reading from engine or internal state
- [x] 2.9 Implement `from_config(config, backend)` classmethod
- [x] 2.10 Add import to `plants/__init__.py`

## 3. CartPole Plant

- [x] 3.1 Create `plants/cartpole.py` with `@register_plant("CartPole")`
- [x] 3.2 Implement `__init__` with configurable cart_mass, pole_mass, pole_length, damping, gravity, dt, track_limits
- [x] 3.3 Implement standalone `step(u)` with semi-implicit Euler integration of coupled dynamics
- [x] 3.4 Implement `dynamics(x, u)` returning continuous-time dx/dt as flat (4,) array
- [x] 3.5 Implement `get_model()` returning linearized A (4,4) and B (4,1) around upright
- [x] 3.6 Implement `physics_engine(engine)` for MuJoCo attachment
- [x] 3.7 Implement engine-mode `step(u)` delegating to MuJoCo
- [x] 3.8 Implement `get_state()` reading from engine or internal state
- [x] 3.9 Implement `from_config(config, backend)` classmethod
- [x] 3.10 Add import to `plants/__init__.py`

## 4. Quadrotor Placeholder

- [x] 4.1 Create `plants/quadrotor.py` with stub class and comment noting HolonomicMobileRobot pattern

## 5. Plant Configs

- [x] 5.1 Create `configs/plants/inverted_pendulum.toml` with default parameters
- [x] 5.2 Create `configs/plants/cartpole.toml` with default parameters

## 6. Generic XML Config Generator

- [x] 6.1 Add `_PLANT_DETECTOR_REGISTRY` and `register_plant_detector` decorator to `factories/registry.py`
- [x] 6.2 Add detector registration to `plants/inverted_pendulum.py` (1 hinge + 1 motor → InvertedPendulum)
- [x] 6.3 Add detector registration to `plants/cartpole.py` (1 slider + 1 hinge + 1 motor → CartPole)
- [x] 6.4 Add detector registration to `plants/armrobot.py` (position actuators on body chain → ArmRobot)
- [x] 6.5 Add detector registration to `plants/holonomicmobilerobot.py` (motor actuators on wheels → HolonomicMobileRobot)
- [x] 6.6 Refactor `scripts/generate_robot_config.py` to use detector registry instead of hardcoded logic
- [x] 6.7 Implement three-tier fallback: heuristic → XML annotation → CLI `--type` flag
- [x] 6.8 Implement batch directory mode with `--output-dir` flag
- [x] 6.9 Implement physical parameter extraction (masses, inertias, joint limits, damping, actuator ranges, gravity)
- [x] 6.10 Preserve existing LeKiwi output format (two `[[plants]]` entries)

## 7. Tests

- [x] 7.1 Add `TestInvertedPendulum` to `tests/test_plants.py` (standalone dynamics, linearized model, state bounds, from_config, engine attachment)
- [x] 7.2 Add `TestCartPole` to `tests/test_plants.py` (standalone dynamics, linearized model, track limits, from_config, engine attachment)
- [x] 7.3 Add tests for config generator (pendulum detection, cartpole detection, LeKiwi detection, batch mode, unknown XML fallback)
