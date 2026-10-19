"""Lifecycle tools — enumerating and evicting stored instances and graphs.

The stores are session-scoped and previously grew without bound: a name could be
overwritten, but nothing could be listed or removed. These pin the enumeration and
the eviction rules, including the one that matters for correctness — a plant that
components are attached to cannot vanish underneath them.
"""

import json

import pytest

from shinro.mcp._common import _attached, _graph_store, _plant_store, _store
from shinro.mcp.codegen_server import delete_graph, generate_scenario, list_graphs
from shinro.mcp.server import (
    attach_plant,
    create_controller,
    create_estimator,
    create_plant,
    create_trajectory,
    delete_instance,
    list_instances,
)

PENDULUM = "samples/plants/inverted_pendulum.toml"
SCENARIO = "tests/integration/scenarios/base_tracking.toml"


@pytest.fixture(autouse=True)
def clear_stores():
    """Every test starts and ends with all four stores empty."""
    for store in (_store, _plant_store, _attached, _graph_store):
        store.clear()
    yield
    for store in (_store, _plant_store, _attached, _graph_store):
        store.clear()


class TestListInstances:
    """One call should show the whole session's live state."""

    def test_reports_every_kind(self):
        """Controllers, estimators, trajectories, and plants are all listed with a kind."""
        create_plant("pend", config_path=PENDULUM)
        create_controller("smc", config_path="samples/controllers/smc.toml")
        create_estimator("kf", config_path="samples/estimators/kalman_base.toml")
        create_trajectory("traj", type="cubic_segments", params={"dt": 0.02})

        data = json.loads(list_instances())
        assert data["pend"]["kind"] == "plant"
        assert data["smc"]["kind"] == "controller"
        assert data["kf"]["kind"] == "estimator"
        assert data["traj"]["kind"] == "trajectory"

    def test_reports_attachments_from_both_sides(self):
        """A component names its plant; the plant names its dependents."""
        create_plant("pend", config_path=PENDULUM)
        create_controller("smc", config_path="samples/controllers/smc.toml")
        attach_plant("smc", "pend")

        data = json.loads(list_instances())
        assert data["smc"]["plant"] == "pend"
        assert data["pend"]["attached_to"] == ["smc"]

    def test_empty_store_is_an_empty_object(self):
        """No instances is `{}`, not an error — an agent can call it unconditionally."""
        assert json.loads(list_instances()) == {}


class TestDeleteInstance:
    """Eviction, and the one guard that must not be skipped."""

    def test_deletes_a_component(self):
        """A controller is removed from the store and from the attachment map."""
        create_plant("pend", config_path=PENDULUM)
        create_controller("smc", config_path="samples/controllers/smc.toml")
        attach_plant("smc", "pend")

        data = json.loads(delete_instance("smc"))
        assert data == {"deleted": "smc", "kind": "controller"}
        assert "smc" not in _store
        assert "smc" not in _attached
        assert json.loads(list_instances())["pend"]["attached_to"] == []

    def test_deletes_a_free_plant(self):
        """A plant with nothing attached is removable."""
        create_plant("pend", config_path=PENDULUM)
        assert json.loads(delete_instance("pend"))["kind"] == "plant"
        assert "pend" not in _plant_store

    def test_refuses_to_delete_an_attached_plant(self):
        """Deleting a plant in use would silently strip its dependents' model."""
        create_plant("pend", config_path=PENDULUM)
        create_controller("smc", config_path="samples/controllers/smc.toml")
        attach_plant("smc", "pend")

        data = json.loads(delete_instance("pend"))
        assert "error" in data
        assert data["attached_to"] == ["smc"]
        assert "pend" in _plant_store  # still there

        # Once the dependent is gone, the plant can be deleted.
        delete_instance("smc")
        assert json.loads(delete_instance("pend"))["deleted"] == "pend"

    def test_unknown_name_returns_error(self):
        """Removing something that was never created reports the miss."""
        data = json.loads(delete_instance("nope"))
        assert "no controller, estimator, trajectory, or plant" in data["error"].lower()


class TestGraphLifecycle:
    """The codegen server's graph registry has the same problem and the same fix."""

    def test_list_graphs_reports_stored_graphs(self, tmp_path):
        """A generated graph appears with its trace size and ports."""
        assert "error" not in json.loads(generate_scenario("base", SCENARIO, str(tmp_path / "g")))
        data = json.loads(list_graphs())
        assert "base" in data
        assert data["base"]["nodes_traced"] > 10
        assert "u" in data["base"]["outputs"]

    def test_delete_graph_removes_it(self, tmp_path):
        """Deleting a graph drops it from the registry (artifacts on disk are untouched)."""
        generate_scenario("base", SCENARIO, str(tmp_path / "g"))
        assert json.loads(delete_graph("base")) == {"deleted": "base"}
        assert "base" not in _graph_store
        assert json.loads(list_graphs()) == {}
        assert (tmp_path / "g" / "graph_data.zig").exists()

    def test_delete_unknown_graph_returns_error(self):
        """An unknown graph name reports the miss."""
        assert "error" in json.loads(delete_graph("nope"))
