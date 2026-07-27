## ADDED Requirements

### Requirement: CartPole state and control dimensions
The CartPole plant SHALL have a 4D state [x, ẋ, θ, θ̇] (cart position in m, cart velocity in m/s, pole angle from upright in rad, pole angular velocity in rad/s) and a 1D control input [F] (horizontal force on cart in N).

#### Scenario: State and control dimensions
- **WHEN** a CartPole is created with default config
- **THEN** `get_state()` returns a 4-element array and `step(u)` accepts a 1-element array

### Requirement: Standalone analytical dynamics
The CartPole SHALL implement standalone coupled dynamics using semi-implicit Euler integration, solving the 2x2 system for ẍ and θ̈ from the equations of motion.

#### Scenario: Pole falls under gravity
- **WHEN** the pole starts at θ=0.1 with F=0
- **THEN** after one step, θ increases and the cart moves

#### Scenario: CartPole stays at upright
- **WHEN** the pole starts at θ=0 with F=0
- **THEN** after one step, θ remains at 0

### Requirement: dynamics() method
The CartPole SHALL implement `dynamics(x, u)` returning continuous-time dx/dt as a flat (4,) array.

#### Scenario: dynamics returns correct shape
- **WHEN** `dynamics(x, u)` is called
- **THEN** it returns a (4,) array

### Requirement: Linearized model
The CartPole SHALL provide a linearized state-space model via `get_model()` around the upright equilibrium (θ=0, θ̇=0, x=0, ẋ=0).

#### Scenario: get_model returns correct shapes
- **WHEN** `get_model()` is called
- **THEN** A has shape (4,4) and B has shape (4,1)

#### Scenario: Linearized model has unstable eigenvalue
- **WHEN** eigenvalues of A are computed
- **THEN** at least one eigenvalue has positive real part (cartpole is unstable)

### Requirement: Configurable damping
The CartPole SHALL accept a configurable damping coefficient for the pole joint, defaulting to 0.

#### Scenario: Damping affects dynamics
- **WHEN** damping is set to a positive value
- **THEN** the pole's angular velocity decays faster than without damping

### Requirement: Track limits
The CartPole SHALL enforce configurable track limits on the cart position x, clipping after each step.

#### Scenario: Cart position clipped to track limits
- **WHEN** the cart position exceeds configured track limits
- **THEN** the cart position is clipped to the limits

### Requirement: MuJoCo engine mode
The CartPole SHALL support optional MuJoCo engine attachment via `physics_engine(engine)`. When attached, `step(u)` delegates to the engine.

#### Scenario: Engine attachment
- **WHEN** a MuJoCo engine is attached
- **THEN** `step(u)` reads/writes state through the engine

### Requirement: from_config factory method
The CartPole SHALL implement `from_config(config, backend)` for TOML-driven instantiation.

#### Scenario: from_config creates valid plant
- **WHEN** `from_config()` is called with a valid config dict
- **THEN** it returns a CartPole instance with correct parameters
