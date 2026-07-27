## ADDED Requirements

### Requirement: InvertedPendulum state and control dimensions
The InvertedPendulum plant SHALL have a 2D state [θ, θ̇] (angle from upright in radians, angular velocity in rad/s) and a 1D control input [τ] (torque at pivot in Nm).

#### Scenario: State and control dimensions
- **WHEN** an InvertedPendulum is created with default config
- **THEN** `get_state()` returns a 2-element array and `step(u)` accepts a 1-element array

### Requirement: Standalone analytical dynamics
The InvertedPendulum SHALL implement standalone dynamics using semi-implicit Euler integration: θ̈ = (g/l)·sin(θ) + τ/(ml²) − (b/ml²)·θ̇, with velocity updated before position.

#### Scenario: Pendulum falls under gravity
- **WHEN** the pendulum starts at θ=0.1 with τ=0
- **THEN** after one step, θ increases (gravity pulls it down)

#### Scenario: Pendulum stays at upright
- **WHEN** the pendulum starts at θ=0 with τ=0
- **THEN** after one step, θ remains at 0

#### Scenario: Balancing torque holds angle
- **WHEN** τ = mgl·sin(θ) is applied at a non-zero angle
- **THEN** the pendulum maintains its angle (within tolerance)

### Requirement: dynamics() method
The InvertedPendulum SHALL implement `dynamics(x, u)` returning continuous-time dx/dt as a flat (2,) array.

#### Scenario: dynamics returns correct shape
- **WHEN** `dynamics(x, u)` is called
- **THEN** it returns a (2,) array

### Requirement: Linearized model
The InvertedPendulum SHALL provide a linearized state-space model via `get_model()` around the upright equilibrium (θ=0, θ̇=0).

#### Scenario: get_model returns correct shapes
- **WHEN** `get_model()` is called
- **THEN** A has shape (2,2) and B has shape (2,1)

#### Scenario: Linearized model has unstable eigenvalue
- **WHEN** eigenvalues of A are computed
- **THEN** at least one eigenvalue has positive real part (inverted pendulum is unstable)

### Requirement: Configurable damping
The InvertedPendulum SHALL accept a configurable damping coefficient b, defaulting to 0.

#### Scenario: Damping affects dynamics
- **WHEN** damping is set to a positive value
- **THEN** the pendulum's angular velocity decays faster than without damping

### Requirement: State bounds
The InvertedPendulum SHALL clip state to configurable bounds after each step.

#### Scenario: State is clipped to bounds
- **WHEN** the state exceeds configured bounds
- **THEN** the state is clipped to the bounds

### Requirement: MuJoCo engine mode
The InvertedPendulum SHALL support optional MuJoCo engine attachment via `physics_engine(engine)`. When attached, `step(u)` delegates to the engine.

#### Scenario: Engine attachment
- **WHEN** a MuJoCo engine is attached
- **THEN** `step(u)` reads/writes state through the engine

### Requirement: from_config factory method
The InvertedPendulum SHALL implement `from_config(config, backend)` for TOML-driven instantiation.

#### Scenario: from_config creates valid plant
- **WHEN** `from_config()` is called with a valid config dict
- **THEN** it returns an InvertedPendulum instance with correct parameters
