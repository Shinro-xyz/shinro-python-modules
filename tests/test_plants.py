import numpy as np


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestHolonomicMobileRobot:
    """Verify holonomic mobile robot: state-space model, integration, copy semantics, and wheel speeds."""

    def test_A_is_identity(self, bk):
        """The discrete-time state matrix A is the 3x3 identity."""
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        A, B = robot.get_model()
        assert np.allclose(_to_np(A, bk), np.eye(3))

    def test_B_is_dt_times_identity(self, bk):
        """The discrete-time input matrix B is dt * I_3."""
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        dt = 0.05
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=dt, backend=bk)
        A, B = robot.get_model()
        assert np.allclose(_to_np(B, bk), dt * np.eye(3))

    def test_step_integrates_correctly(self, bk):
        """step() integrates the state: state += u * dt."""
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        dt = 0.01
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=dt, backend=bk)
        u = bk.array([0.5, 0.0, 0.0])
        robot.step(u)
        state = robot.get_state()
        expected = np.array([0.5 * dt, 0.0, 0.0])
        assert np.allclose(_to_np(state, bk), expected, atol=1e-10)

    def test_get_state_returns_copy(self, bk):
        """get_state() returns a copy, not a reference to the internal state."""
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        state = robot.get_state()
        state[0] = 99.0
        internal = robot.get_state()
        assert _to_np(internal, bk)[0] != 99.0

    def test_set_pose_updates_state(self, bk):
        """set_pose() directly sets the robot's pose."""
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        robot.set_pose(1.0, 2.0, 0.5)
        state = robot.get_state()
        assert np.allclose(_to_np(state, bk), [1.0, 2.0, 0.5])

    def test_step_returns_wheel_speeds(self, bk):
        """step() returns a wheel speed vector of length n_wheels."""
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        wheel_speeds = robot.step(u)
        assert _to_np(wheel_speeds, bk).shape == (3,)


class TestInvertedPendulum:
    """Verify inverted pendulum: standalone dynamics, linearized model, state bounds, from_config."""

    def test_step_falls(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(mass=0.1, length=0.5, gravity=9.81, dt=0.01, backend=bk)
        pend.state = bk.array([0.1, 0.0])
        pend.step(bk.array([0.0]))
        state = pend.get_state()
        assert _to_np(state, bk)[0] > 0.1

    def test_step_upright(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(mass=0.1, length=0.5, gravity=9.81, dt=0.01, backend=bk)
        pend.state = bk.array([0.0, 0.0])
        pend.step(bk.array([0.0]))
        state = pend.get_state()
        assert np.allclose(_to_np(state, bk), [0.0, 0.0], atol=1e-10)

    def test_step_balancing(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(mass=0.1, length=0.5, gravity=9.81, dt=0.01, backend=bk)
        theta = 0.2
        tau = -pend.m * pend.g * pend.l * np.sin(theta)
        dx = pend.dynamics(bk.array([theta, 0.0]), bk.array([tau]))
        assert abs(_to_np(dx, bk)[1]) < 1e-10

    def test_dynamics_shape(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(backend=bk)
        dx = pend.dynamics(bk.array([0.1, 0.0]), bk.array([0.0]))
        assert _to_np(dx, bk).shape == (2,)

    def test_get_model_shape(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(backend=bk)
        A, B = pend.get_model()
        assert _to_np(A, bk).shape == (2, 2)
        assert _to_np(B, bk).shape == (2, 1)

    def test_get_model_unstable(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(backend=bk)
        A, _ = pend.get_model()
        eigs = np.linalg.eigvals(_to_np(A, bk))
        assert np.any(np.real(eigs) > 0)

    def test_state_bounds(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        lo = bk.array([-1.0, -5.0])
        hi = bk.array([1.0, 5.0])
        pend = InvertedPendulum(state_bounds=(lo, hi), backend=bk)
        pend.state = bk.array([10.0, 20.0])
        pend.step(bk.array([0.0]))
        state = pend.get_state()
        assert _to_np(state, bk)[0] <= 1.0

    def test_from_config(self, bk):
        from shinro.plants.inverted_pendulum import InvertedPendulum
        config = {"mass": 0.2, "length": 1.0, "damping": 0.01, "gravity": 9.81, "dt": 0.02}
        pend = InvertedPendulum.from_config(config, backend=bk)
        assert pend.m == 0.2
        assert pend.l == 1.0
        assert pend.b == 0.01
        assert pend.dt == 0.02


class TestCartPole:
    """Verify cart-pole: standalone dynamics, linearized model, track limits, from_config."""

    def test_step_falls(self, bk):
        from shinro.plants.cartpole import CartPole
        cp = CartPole(cart_mass=0.5, pole_mass=0.1, pole_length=0.5, gravity=9.81, dt=0.01, backend=bk)
        cp.state = bk.array([0.0, 0.0, 0.1, 0.0])
        cp.step(bk.array([0.0]))
        state = cp.get_state()
        assert _to_np(state, bk)[2] > 0.1

    def test_step_upright(self, bk):
        from shinro.plants.cartpole import CartPole
        cp = CartPole(cart_mass=0.5, pole_mass=0.1, pole_length=0.5, gravity=9.81, dt=0.01, backend=bk)
        cp.state = bk.array([0.0, 0.0, 0.0, 0.0])
        cp.step(bk.array([0.0]))
        state = cp.get_state()
        assert np.allclose(_to_np(state, bk), [0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_dynamics_shape(self, bk):
        from shinro.plants.cartpole import CartPole
        cp = CartPole(backend=bk)
        dx = cp.dynamics(bk.array([0.0, 0.0, 0.1, 0.0]), bk.array([0.0]))
        assert _to_np(dx, bk).shape == (4,)

    def test_get_model_shape(self, bk):
        from shinro.plants.cartpole import CartPole
        cp = CartPole(backend=bk)
        A, B = cp.get_model()
        assert _to_np(A, bk).shape == (4, 4)
        assert _to_np(B, bk).shape == (4, 1)

    def test_get_model_unstable(self, bk):
        from shinro.plants.cartpole import CartPole
        cp = CartPole(backend=bk)
        A, _ = cp.get_model()
        eigs = np.linalg.eigvals(_to_np(A, bk))
        assert np.any(np.real(eigs) > 0)

    def test_track_limits(self, bk):
        from shinro.plants.cartpole import CartPole
        cp = CartPole(cart_mass=0.5, pole_mass=0.1, pole_length=0.5, gravity=9.81, dt=0.01,
                      track_limits=(-1.0, 1.0), backend=bk)
        cp.state = bk.array([5.0, 0.0, 0.0, 0.0])
        cp.step(bk.array([0.0]))
        state = cp.get_state()
        assert _to_np(state, bk)[0] <= 1.0

    def test_from_config(self, bk):
        from shinro.plants.cartpole import CartPole
        config = {"cart_mass": 1.0, "pole_mass": 0.2, "pole_length": 0.8, "damping": 0.01,
                  "gravity": 9.81, "dt": 0.02, "track_limits": [-2.0, 2.0]}
        cp = CartPole.from_config(config, backend=bk)
        assert cp.M == 1.0
        assert cp.m == 0.2
        assert cp.l == 0.8
        assert cp.dt == 0.02


class TestConfigGenerator:
    """Verify the XML config generator: detection, batch mode, unknown XML fallback."""

    PENDULUM_XML = """<mujoco model="pendulum">
  <worldbody>
    <body name="pole" pos="0 0 0">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01" range="-3.14 3.14"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="0.1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque" joint="hinge" gear="1" ctrlrange="-5 5"/>
  </actuator>
</mujoco>"""

    CARTPOLE_XML = """<mujoco model="cartpole">
  <worldbody>
    <body name="cart" pos="0 0 0">
      <joint name="slider" type="slide" axis="1 0 0" range="-2 2"/>
      <geom type="box" size="0.1 0.05 0.05" mass="0.5"/>
      <body name="pole" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.01" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.5" size="0.02" mass="0.1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="cart_force" joint="slider" gear="1" ctrlrange="-10 10"/>
  </actuator>
</mujoco>"""

    LEKIWI_XML = """<mujoco model="lekiwi">
  <worldbody>
    <body name="arm_base" pos="0 0 0">
      <joint name="shoulder" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.02" mass="0.1"/>
      <body name="arm_upper" pos="0 0.1 0">
        <joint name="elbow" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size="0.02" mass="0.1"/>
      </body>
    </body>
    <body name="wheel1" pos="0.1 0 0">
      <joint name="drive1" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.03" mass="0.05"/>
    </body>
    <body name="wheel2" pos="-0.05 0.086 0">
      <joint name="drive2" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.03" mass="0.05"/>
    </body>
    <body name="wheel3" pos="-0.05 -0.086 0">
      <joint name="drive3" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.03" mass="0.05"/>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder" joint="shoulder"/>
    <position name="elbow" joint="elbow"/>
    <motor name="drive1" joint="drive1"/>
    <motor name="drive2" joint="drive2"/>
    <motor name="drive3" joint="drive3"/>
  </actuator>
</mujoco>"""

    UNKNOWN_XML = """<mujoco model="unknown">
  <worldbody>
    <body name="thing" pos="0 0 0">
      <joint name="j1" type="ball"/>
      <geom type="sphere" size="0.1" mass="1.0"/>
    </body>
  </worldbody>
</mujoco>"""

    def test_detect_pendulum(self, tmp_path):
        import xml.etree.ElementTree as ET

        from scripts.generate_robot_config import detect_plant_types
        root = ET.fromstring(self.PENDULUM_XML)
        types = detect_plant_types(root)
        assert "InvertedPendulum" in types

    def test_detect_cartpole(self, tmp_path):
        import xml.etree.ElementTree as ET

        from scripts.generate_robot_config import detect_plant_types
        root = ET.fromstring(self.CARTPOLE_XML)
        types = detect_plant_types(root)
        assert "CartPole" in types

    def test_detect_lekiwi(self, tmp_path):
        import xml.etree.ElementTree as ET

        from scripts.generate_robot_config import detect_plant_types
        root = ET.fromstring(self.LEKIWI_XML)
        types = detect_plant_types(root)
        assert "ArmRobot" in types
        assert "HolonomicMobileRobot" in types

    def test_unknown_xml_fallback(self, tmp_path):
        import xml.etree.ElementTree as ET

        from scripts.generate_robot_config import detect_plant_types
        root = ET.fromstring(self.UNKNOWN_XML)
        types = detect_plant_types(root)
        assert len(types) == 0

    def test_cli_type_override(self, tmp_path):
        import xml.etree.ElementTree as ET

        from scripts.generate_robot_config import detect_plant_types
        root = ET.fromstring(self.UNKNOWN_XML)
        types = detect_plant_types(root, cli_type="InvertedPendulum")
        assert types == ["InvertedPendulum"]

    def test_generate_pendulum_config(self, tmp_path):
        from scripts.generate_robot_config import generate_config
        xml_file = tmp_path / "pendulum.xml"
        xml_file.write_text(self.PENDULUM_XML)
        config = generate_config(str(xml_file))
        assert len(config.get("plants", [])) == 1
        assert config["plants"][0]["type"] == "InvertedPendulum"

    def test_generate_cartpole_config(self, tmp_path):
        from scripts.generate_robot_config import generate_config
        xml_file = tmp_path / "cartpole.xml"
        xml_file.write_text(self.CARTPOLE_XML)
        config = generate_config(str(xml_file))
        assert len(config.get("plants", [])) == 1
        assert config["plants"][0]["type"] == "CartPole"

    def test_generate_lekiwi_config(self, tmp_path):
        from scripts.generate_robot_config import generate_config
        xml_file = tmp_path / "lekiwi.xml"
        xml_file.write_text(self.LEKIWI_XML)
        config = generate_config(str(xml_file))
        assert len(config.get("plants", [])) == 2
        types = [p["type"] for p in config["plants"]]
        assert "ArmRobot" in types
        assert "HolonomicMobileRobot" in types

    def test_batch_mode(self, tmp_path):
        from scripts.generate_robot_config import generate_config, toml_string
        input_dir = tmp_path / "models"
        input_dir.mkdir()
        (input_dir / "pendulum.xml").write_text(self.PENDULUM_XML)
        (input_dir / "cartpole.xml").write_text(self.CARTPOLE_XML)
        output_dir = tmp_path / "configs"
        output_dir.mkdir()
        for xml_file in sorted(input_dir.glob("*.xml")):
            config = generate_config(str(xml_file))
            if config.get("plants"):
                out_path = output_dir / (xml_file.stem + ".toml")
                out_path.write_text(toml_string(config))
        assert (output_dir / "pendulum.toml").exists()
        assert (output_dir / "cartpole.toml").exists()
