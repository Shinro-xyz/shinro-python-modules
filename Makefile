.PHONY: test test-quick test-functional test-all test-integration lint \
	test-controllers test-estimators test-plants test-trajectories test-armrobot \
	test-components test-array-backend test-controllability test-factories \
	test-linearization test-adversarial test-mcp-server test-mcp-functional

# Run the full test suite (skips the very large horizon MPC timeout test)
test:
	python3 -m pytest tests/ -v --tb=short -k "not test_very_large_horizon_mpc_times_out"

# Run only unit tests (fast)
test-quick:
	python3 -m pytest tests/test_mcp_server.py tests/test_controllers.py tests/test_estimators.py tests/test_trajectories.py tests/test_plants.py tests/test_factories.py tests/test_components.py tests/test_array_backend.py tests/test_controllability_checker.py -v --tb=short

# Run only functional tests (spawns real server)
test-functional:
	python3 -m pytest tests/test_mcp_server_functional.py -v --tb=short

# Run all tests including the slow horizon test
test-all:
	python3 -m pytest tests/ -v --tb=short

# Run the full-loop integration suite (MuJoCo required, opt-in only — not in CI)
test-integration:
	python3 -m pytest tests/integration/ -v --tb=short --override-ini="addopts="

# Run linter and type checker
lint:
	ruff check .
	pyright utils/ components.py controllers/ estimators/ trajectories/ plants/

# Run an individual test group by short name, e.g. `make test-controllers`
test-controllers:
	python3 -m pytest tests/test_controllers.py -v --tb=short

test-estimators:
	python3 -m pytest tests/test_estimators.py -v --tb=short

test-plants:
	python3 -m pytest tests/test_plants.py -v --tb=short

test-trajectories:
	python3 -m pytest tests/test_trajectories.py -v --tb=short

test-armrobot:
	python3 -m pytest tests/test_armrobot.py -v --tb=short

test-components:
	python3 -m pytest tests/test_components.py -v --tb=short

test-array-backend:
	python3 -m pytest tests/test_array_backend.py -v --tb=short

test-controllability:
	python3 -m pytest tests/test_controllability_checker.py -v --tb=short

test-factories:
	python3 -m pytest tests/test_factories.py -v --tb=short

test-linearization:
	python3 -m pytest tests/test_linearization.py -v --tb=short

test-adversarial:
	python3 -m pytest tests/test_adversarial.py -v --tb=short

test-mcp-server:
	python3 -m pytest tests/test_mcp_server.py -v --tb=short

test-mcp-functional:
	python3 -m pytest tests/test_mcp_server_functional.py -v --tb=short
