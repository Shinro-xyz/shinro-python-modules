.PHONY: test test-quick test-functional test-all test-integration lint install build \
	release-patch release-minor release-major changelog \
	test-controllers test-estimators test-plants test-trajectories test-armrobot \
	test-components test-array-backend test-batched-adapter test-controllability test-factories \
	test-linearization test-adversarial test-mcp-server test-mcp-functional

# Install the package in editable mode
install:
	pip install -e .

# Build source + wheel distributions
build:
	python -m build

# ---------------------------------------------------------------------------
# Release helpers. Versioning is setuptools-scm: the released version IS the
# git tag (v<X>.<Y>.<Z>). These targets bump the tag, regenerate CHANGELOG.md
# via git-cliff (from Conventional Commits, see cliff.toml), stage it, and
# create the tag. Push the changelog commit on main and the tag to origin to
# trigger .github/workflows/release.yml (wheel build + GitHub Release).
#
# Semver rules:
#   patch  — backwards-compatible bug fix          (0.1.0 -> 0.1.1)
#   minor  — backwards-compatible new feature      (0.1.1 -> 0.2.0)
#   major  — incompatible API change               (0.2.0 -> 1.0.0)
# ---------------------------------------------------------------------------
release-patch:
	@$(MAKE) _release BUMP=patch
release-minor:
	@$(MAKE) _release BUMP=minor
release-major:
	@$(MAKE) _release BUMP=major

# Preview/regenerate the changelog without cutting a release.
changelog:
	git cliff --unreleased --output CHANGELOG.md

_release:
	@current=$$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "0.0.0"); \
	maj=$${current%%.*}; \
	rest=$${current#*.}; \
	min=$${rest%%.*}; \
	pat=$${rest#*.}; \
	case "$(BUMP)" in \
	  patch) new="$$maj.$$min.$$((pat+1))" ;; \
	  minor) new="$$maj.$$((min+1)).0" ;; \
	  major) new="$$((maj+1)).0.0" ;; \
	esac; \
	echo ">>> Bumping $$current -> v$$new"; \
	git cliff --tag "v$$new" --output CHANGELOG.md; \
	git add CHANGELOG.md; \
	echo ">>> Regenerated CHANGELOG.md and staged it."; \
	echo "    Commit it on main, then push the tag to trigger the release:"; \
	echo "      git commit -m \"chore: release v$$new\""; \
	echo "      git push origin main"; \
	echo "      git tag -a \"v$$new\" -m \"Release v$$new\""; \
	echo "      git push origin \"v$$new\""

# Run the full test suite (skips the very large horizon MPC timeout test)
test:
	python3 -m pytest tests/ -v --tb=short -k "not test_very_large_horizon_mpc_times_out"

# Run only unit tests (fast)
test-quick:
	python3 -m pytest tests/test_mcp_server.py tests/test_controllers.py tests/test_estimators.py tests/test_trajectories.py tests/test_plants.py tests/test_factories.py tests/test_components.py tests/test_array_backend.py tests/test_batched_adapter.py tests/test_controllability_checker.py -v --tb=short

# Run only functional tests (spawns real server)
test-functional:
	python3 -m pytest tests/test_mcp_server_functional.py -v --tb=short

# Run all tests including the slow horizon test
test-all:
	python3 -m pytest tests/ -v --tb=short

# Run the full-loop integration suite (MuJoCo required, opt-in only — not in CI)
test-integration:
	python3 -m pytest tests/integration/ -v --tb=short --override-ini="addopts="

# ───────────────────────────────────────────────────────────────────────────
# Zig lowering (slice b): serialize the base_tracking composed graph to
# runtime/graph_data.zig, compile the comptime VM into build/base.so, then
# cross-check the .so against the Python interpreter (ctypes oracle).
# Requires `zig` on PATH (see runtime/README.md). build/ is gitignored.
# ───────────────────────────────────────────────────────────────────────────
zig-gen:
	mkdir -p build
	python3 -m shinro.codegen.gen_base

zig-build: zig-gen
	zig build-lib runtime/lower.zig -dynamic -lc -femit-bin=build/base.so -I runtime

test-zig: zig-build
	python3 -m pytest tests/test_zig_lowering.py -v --tb=short

# Run linter and type checker
lint:
	ruff check .
	pyright src/shinro/utils/ src/shinro/components.py src/shinro/controllers/ src/shinro/estimators/ src/shinro/trajectories/ src/shinro/plants/

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

test-batched-adapter:
	python3 -m pytest tests/test_batched_adapter.py -v --tb=short

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
