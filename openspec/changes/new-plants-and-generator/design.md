## Context

The control suite has two plants (ArmRobot, HolonomicMobileRobot) that follow a consistent pattern: the controller sends task-space commands, the plant maps them to low-level actions, and optionally delegates to a MuJoCo physics engine. Both plants return simplified double-integrator models from `get_model()` and do not implement `dynamics()` (returns None).

To support nonlinear controllers and estimators, new plants need to implement `dynamics(x, u)` returning continuous-time dx/dt. The XML config generator needs to be generic enough to handle any robot model without manual wiring.

## Goals / Non-Goals

**Goals:**
- InvertedPendulum plant with standalone analytical dynamics and optional MuJoCo engine mode
- CartPole plant with standalone analytical dynamics and optional MuJoCo engine mode
- Both plants implement `dynamics()`, `get_model()` (linearized), state bounds, and configurable damping
- Generic XML-to-TOML config generator with plant type auto-detection via registered detectors
- Detector registry lives alongside plant classes — each plant registers a detector at import time
- Generator handles single plants, combined robots (multiple detectors fire), and unknown XMLs gracefully
- Quadrotor placeholder stub for future implementation
- Minimal MuJoCo XML models for pendulum and cartpole
- TOML configs for the new plants
- Unit tests for standalone dynamics, engine mode, linearized models, state bounds, and config generation

**Non-Goals:**
- Quadrotor plant implementation (placeholder only)
- Nonlinear controllers or estimators (these consume the new plants but are separate changes)
- URDF support in the config generator (MJCF only for now, URDF can be added later)
- Full RobotSim integration for the new plants (they work standalone or via `from_config`)

## Decisions

### 1. Plant dynamics integration: semi-implicit Euler
Both plants use semi-implicit Euler (velocity updated first, then position) for better energy conservation than forward Euler. This matches the lab notes' finding that the simulation is already 85x faster than real-time, so the slight extra cost is negligible.

### 2. `dynamics()` returns flat (n_x,) arrays
The existing `utils/linearization.py` expects flat arrays. The estimators use column vectors (n_x, 1), but the conversion is trivial. Flat arrays are the convention for `dynamics()` since it's consumed by linearization and nonlinear controllers.

### 3. Detector registry pattern
Each plant registers a detector function alongside its class:
```python
@register_plant_detector("InvertedPendulum")
def detect_pendulum(xml_tree) -> bool:
    ...
```
Detectors are non-exclusive — multiple can fire for the same XML (e.g., LeKiwi produces ArmRobot + HolonomicMobileRobot). The generator collects all matches and produces one `[[plants]]` entry per match.

### 4. Detector matching: three-tier fallback
1. Heuristic match (detector registry) — auto-detect from XML structure
2. XML annotation (`<plant type="..."/>`) — explicit override
3. CLI `--type` flag — manual override
4. Fallback: extract whatever is available and warn

### 5. MuJoCo XML models: minimal inline
Pendulum and cartpole XMLs are ~20-30 lines each — simple enough to write directly. No external downloads needed. Stored in `models/` at repo root.

### 6. Configs in `configs/plants/`
Following the existing pattern (`configs/controllers/`, `configs/estimators/`, `configs/trajectories/`). Each plant gets a TOML config with its physical parameters.

### 7. State bounds enforced in `step()`
Both plants clip state to configurable bounds after integration. CartPole enforces track limits on the cart position. This prevents the standalone dynamics from diverging to infinity.

### 8. Damping as configurable parameter
Included in both plants, defaulting to zero. The linearized model includes the damping term in the A matrix.

## Risks / Trade-offs

- **Detector fragility**: Heuristics may misclassify unusual XML structures. Mitigation: three-tier fallback (heuristic → annotation → CLI) ensures a path exists for any XML.
- **Standalone vs engine divergence**: The analytical dynamics may not perfectly match MuJoCo's physics (e.g., joint friction, contact dynamics). Mitigation: engine mode is optional; standalone mode is the primary path for controller/estimator testing.
- **Semi-implicit Euler stability**: For stiff systems or large dt, semi-implicit Euler can still diverge. Mitigation: state bounds provide a safety net; users can reduce dt.
- **Generator script complexity**: Adding detector registration adds indirection. Mitigation: the pattern is simple (5-line detector functions) and the existing LeKiwi logic is preserved as a fallback.
