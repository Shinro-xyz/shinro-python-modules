.PHONY: test test-quick test-functional test-all lint

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

# Run type checker
lint:
	pyright utils/ components.py controllers/ estimators/ trajectories/ plants/
