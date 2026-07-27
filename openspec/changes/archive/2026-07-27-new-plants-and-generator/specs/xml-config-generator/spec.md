## ADDED Requirements

### Requirement: Plant type auto-detection from XML
The config generator SHALL auto-detect plant types from MJCF XML files using a registry of detector functions. Each plant registers a detector that inspects the XML structure (joint types, actuator types, body tree) and returns True if the XML matches.

#### Scenario: Pendulum XML detected as InvertedPendulum
- **WHEN** a pendulum XML (1 hinge joint + 1 motor actuator) is passed to the generator
- **THEN** it generates an InvertedPendulum config

#### Scenario: CartPole XML detected as CartPole
- **WHEN** a cartpole XML (1 slider + 1 hinge + 1 motor actuator) is passed to the generator
- **THEN** it generates a CartPole config

#### Scenario: LeKiwi XML detected as ArmRobot + HolonomicMobileRobot
- **WHEN** the LeKiwi XML (position actuators on arm chain + motor actuators on wheels) is passed to the generator
- **THEN** it generates both ArmRobot and HolonomicMobileRobot configs

### Requirement: Detector registry
Detector functions SHALL be registered alongside plant classes via a `@register_plant_detector` decorator. The registry is auto-populated at import time.

#### Scenario: Detector registered with plant
- **WHEN** a plant module is imported
- **THEN** its detector is available in the detector registry

### Requirement: Three-tier detection fallback
The generator SHALL use a three-tier detection strategy: (1) heuristic match from detector registry, (2) XML annotation via `<plant type="..."/>` element, (3) CLI `--type` flag. If none match, it SHALL extract available information and print a warning.

#### Scenario: XML annotation overrides heuristic
- **WHEN** an XML contains `<plant type="Quadrotor"/>` but the heuristic would match something else
- **THEN** the annotated type is used

#### Scenario: CLI flag overrides all
- **WHEN** `--type InvertedPendulum` is passed on the CLI
- **THEN** the specified type is used regardless of XML content

#### Scenario: Unknown XML produces warning
- **WHEN** an XML does not match any detector and has no annotation
- **THEN** the generator prints a warning and extracts whatever information is available

### Requirement: Combined robot support
The generator SHALL support XMLs containing multiple plant types (e.g., arm + base), producing multiple `[[plants]]` entries in the output config.

#### Scenario: LeKiwi produces two plant entries
- **WHEN** the LeKiwi XML is processed
- **THEN** the output TOML contains two `[[plants]]` entries (ArmRobot and HolonomicMobileRobot)

### Requirement: Batch directory mode
The generator SHALL support scanning a directory of XML files and generating configs for all of them.

#### Scenario: Batch mode processes all XMLs
- **WHEN** `--output-dir` is specified with a directory path
- **THEN** all XML files in the input directory are processed and configs written to the output directory

### Requirement: Physical parameter extraction
The generator SHALL extract physical parameters from the XML (masses, inertias, joint limits, damping, actuator ranges, gravity) and include them in the generated config.

#### Scenario: Mass extracted from body element
- **WHEN** a body element has a `mass` attribute
- **THEN** the mass is included in the generated config

#### Scenario: Joint limits extracted
- **WHEN** a joint element has a `range` attribute
- **THEN** the range is included in the generated config
