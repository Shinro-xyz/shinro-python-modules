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

    def test_get_model_upright_matches_analytic(self, bk):
        """get_model() at default upright equals the closed-form Jacobians."""
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(mass=0.1, length=0.5, gravity=9.81, dt=0.01, backend=bk)
        A, B = pend.get_model()
        expected_A = np.array([
            [0.0, 1.0],
            [pend.g / pend.l, -pend.b / (pend.m * pend.l**2)],
        ])
        expected_B = np.array([[0.0], [1.0 / (pend.m * pend.l**2)]])
        assert np.allclose(_to_np(A, bk), expected_A, atol=1e-6)
        assert np.allclose(_to_np(B, bk), expected_B, atol=1e-6)

    def test_get_model_default_matches_explicit_upright(self, bk):
        """get_model() with no args equals get_model(zeros, zeros)."""
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(backend=bk)
        A_default, B_default = pend.get_model()
        A_explicit, B_explicit = pend.get_model(bk.zeros(2), bk.zeros(1))
        assert np.allclose(_to_np(A_default, bk), _to_np(A_explicit, bk), atol=1e-12)
        assert np.allclose(_to_np(B_default, bk), _to_np(B_explicit, bk), atol=1e-12)

    def test_get_model_nonzero_point(self, bk):
        """Linearization at a non-upright point differs from the upright model."""
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(mass=0.1, length=0.5, gravity=9.81, dt=0.01, backend=bk)
        A_upright, _ = pend.get_model()
        A_off, B_off = pend.get_model(bk.array([0.5, 0.0]), bk.zeros(1))
        assert _to_np(A_off, bk).shape == (2, 2)
        assert _to_np(B_off, bk).shape == (2, 1)
        assert not np.allclose(_to_np(A_off, bk), _to_np(A_upright, bk), atol=1e-6)

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

    def test_get_model_upright_matches_analytic(self, bk):
        """get_model() at default upright equals the closed-form Jacobians."""
        from shinro.plants.cartpole import CartPole
        cp = CartPole(cart_mass=0.5, pole_mass=0.1, pole_length=0.5, gravity=9.81, dt=0.01, backend=bk)
        M, m, pole_len, g, b = cp.M, cp.m, cp.l, cp.g, cp.b
        A, B = cp.get_model()
        expected_A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -m * g / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (M + m) * g / (M * pole_len), -b / (M * pole_len**2)],
        ])
        expected_B = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * pole_len)]])
        assert np.allclose(_to_np(A, bk), expected_A, atol=1e-6)
        assert np.allclose(_to_np(B, bk), expected_B, atol=1e-6)

    def test_get_model_default_matches_explicit_upright(self, bk):
        """get_model() with no args equals get_model(zeros, zeros)."""
        from shinro.plants.cartpole import CartPole
        cp = CartPole(backend=bk)
        A_default, B_default = cp.get_model()
        A_explicit, B_explicit = cp.get_model(bk.zeros(4), bk.zeros(1))
        assert np.allclose(_to_np(A_default, bk), _to_np(A_explicit, bk), atol=1e-12)
        assert np.allclose(_to_np(B_default, bk), _to_np(B_explicit, bk), atol=1e-12)

    def test_get_model_nonzero_point(self, bk):
        """Linearization at a non-upright point differs from the upright model."""
        from shinro.plants.cartpole import CartPole
        cp = CartPole(cart_mass=0.5, pole_mass=0.1, pole_length=0.5, gravity=9.81, dt=0.01, backend=bk)
        A_upright, _ = cp.get_model()
        A_off, B_off = cp.get_model(bk.array([0.0, 0.0, 0.1, 0.0]), bk.zeros(1))
        assert _to_np(A_off, bk).shape == (4, 4)
        assert _to_np(B_off, bk).shape == (4, 1)
        assert not np.allclose(_to_np(A_off, bk), _to_np(A_upright, bk), atol=1e-6)

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


class TestDoublePendulum:
    """Verify double pendulum: standalone dynamics, linearized model, state bounds, from_config, engine mode."""

    def _make(self, bk):
        from shinro.plants.double_pendulum import DoublePendulum
        return DoublePendulum(mass_top=0.1, mass_bottom=0.1, length_top=0.5, length_bottom=0.5,
                              dt=0.01, g=9.81, backend=bk)

    def test_step_shape(self, bk):
        dp = self._make(bk)
        state = dp.step(bk.array([0.0, 0.0]))
        assert _to_np(state, bk).shape == (4,)

    def test_step_at_rest_stays(self, bk):
        """Zero state and zero control keep the pendulum at rest."""
        dp = self._make(bk)
        dp.state = bk.array([0.0, 0.0, 0.0, 0.0])
        dp.step(bk.array([0.0, 0.0]))
        state = dp.get_state()
        assert np.allclose(_to_np(state, bk), [0.0, 0.0, 0.0, 0.0], atol=1e-10)

    def test_step_euler_accumulates(self, bk):
        """A nonzero state accumulates via semi-implicit Euler (x_new = x + xdot * dt)."""
        dp = self._make(bk)
        x = bk.array([0.1, 0.2, 0.3, 0.4])
        u = bk.array([0.0, 0.0])
        dp.state = bk.copy(x)
        xdot = _to_np(dp.dynamics(x, u), bk)
        x_np = _to_np(x, bk)
        state = _to_np(dp.step(u), bk)
        omega_1_new = x_np[2] + xdot[2] * dp.dt
        omega_2_new = x_np[3] + xdot[3] * dp.dt
        expected = np.array([
            x_np[0] + omega_1_new * dp.dt,
            x_np[1] + omega_2_new * dp.dt,
            omega_1_new,
            omega_2_new,
        ])
        assert np.allclose(state, expected, atol=1e-10)

    def test_dynamics_shape(self, bk):
        dp = self._make(bk)
        dx = dp.dynamics(bk.array([0.1, 0.0, 0.0, 0.0]), bk.array([0.0, 0.0]))
        assert _to_np(dx, bk).shape == (4,)

    def test_dynamics_balancing(self, bk):
        """Torques that cancel gravity produce zero angular acceleration."""
        dp = self._make(bk)
        m1, m2, l1, l2, g = dp.m1, dp.m2, dp.l1, dp.l2, dp.g
        theta_1, theta_2 = 0.3, -0.2
        tau_1 = (m1 + m2) * g * l1 * np.sin(theta_1)
        tau_2 = m2 * g * l2 * np.sin(theta_2)
        dx = dp.dynamics(bk.array([theta_1, theta_2, 0.0, 0.0]), bk.array([tau_1, tau_2]))
        assert abs(_to_np(dx, bk)[2]) < 1e-8
        assert abs(_to_np(dx, bk)[3]) < 1e-8

    def test_dynamics_coriolis_matches_closed_form(self, bk):
        """At zero gravity and zero torque, thetaddot = -M^{-1} C omega.

        Verifies the Coriolis matrix against the Euler-Lagrange velocity terms:
        C omega = [m2 l1 l2 sin(dtheta) w2^2, -m2 l1 l2 sin(dtheta) w1^2].
        Uses g=0 and tau=0 so only the Coriolis term contributes.
        """
        dp = self._make(bk)
        m1, m2, l1, l2 = dp.m1, dp.m2, dp.l1, dp.l2
        theta_1, theta_2, w_1, w_2 = 0.3, 0.1, 1.5, -0.7
        dtheta = theta_1 - theta_2
        dp.g = 0.0
        M = np.array([
            [(m1 + m2) * l1**2, m2 * l1 * l2 * np.cos(dtheta)],
            [m2 * l1 * l2 * np.cos(dtheta), m2 * l2**2],
        ])
        C_omega = np.array([
            m2 * l1 * l2 * np.sin(dtheta) * w_2**2,
            -m2 * l1 * l2 * np.sin(dtheta) * w_1**2,
        ])
        expected_thetaddot = np.linalg.solve(M, -C_omega)
        dx = dp.dynamics(bk.array([theta_1, theta_2, w_1, w_2]), bk.array([0.0, 0.0]))
        assert np.allclose(_to_np(dx, bk)[2:], expected_thetaddot, atol=1e-9)
        assert np.allclose(_to_np(dx, bk)[:2], [w_1, w_2], atol=1e-12)

    def test_get_model_shape(self, bk):
        dp = self._make(bk)
        A, B = dp.get_model()
        assert _to_np(A, bk).shape == (4, 4)
        assert _to_np(B, bk).shape == (4, 2)

    def test_get_model_at_rest_matches_analytic(self, bk):
        """At rest the linearized model matches the closed-form Jacobian.

        At theta=0, omega=0, the manipulator equation gives
        thetaddot = M(0)^{-1} (tau - C*omega - G(0)), where C*omega = 0 and
        G(0) = 0. So B rows 2-3 equal M(0)^{-1} and A rows 2-3 (theta cols)
        equal -M(0)^{-1} dG/dtheta|_0.
        """
        dp = self._make(bk)
        m1, m2, l1, l2, g = dp.m1, dp.m2, dp.l1, dp.l2, dp.g
        M0 = np.array([
            [(m1 + m2) * l1**2, m2 * l1 * l2],
            [m2 * l1 * l2, m2 * l2**2],
        ])
        dG_dtheta = np.diag([(m1 + m2) * g * l1, m2 * g * l2])
        Minv = np.linalg.inv(M0)
        expected_A = np.zeros((4, 4))
        expected_A[0, 2] = 1.0
        expected_A[1, 3] = 1.0
        expected_A[2:, :2] = -Minv @ dG_dtheta
        expected_B = np.zeros((4, 2))
        expected_B[2:, :] = Minv
        A, B = dp.get_model()
        assert np.allclose(_to_np(A, bk), expected_A, atol=1e-6)
        assert np.allclose(_to_np(B, bk), expected_B, atol=1e-6)

    def test_get_model_default_matches_explicit(self, bk):
        """get_model() with no args equals get_model(zeros, zeros)."""
        dp = self._make(bk)
        A_default, B_default = dp.get_model()
        A_explicit, B_explicit = dp.get_model(bk.zeros(4), bk.zeros(2))
        assert np.allclose(_to_np(A_default, bk), _to_np(A_explicit, bk), atol=1e-12)
        assert np.allclose(_to_np(B_default, bk), _to_np(B_explicit, bk), atol=1e-12)

    def test_get_model_nonzero_point(self, bk):
        """Linearization at a non-rest point differs from the rest model."""
        dp = self._make(bk)
        A_rest, _ = dp.get_model()
        A_off, B_off = dp.get_model(bk.array([0.3, 0.2, 0.0, 0.0]), bk.zeros(2))
        assert _to_np(A_off, bk).shape == (4, 4)
        assert _to_np(B_off, bk).shape == (4, 2)
        assert not np.allclose(_to_np(A_off, bk), _to_np(A_rest, bk), atol=1e-6)

    def test_state_bounds(self, bk):
        from shinro.plants.double_pendulum import DoublePendulum
        lo = bk.array([-1.0, -1.0, -5.0, -5.0])
        hi = bk.array([1.0, 1.0, 5.0, 5.0])
        dp = DoublePendulum(state_bounds=(lo, hi), backend=bk)
        dp.state = bk.array([10.0, 10.0, 20.0, 20.0])
        dp.step(bk.array([0.0, 0.0]))
        state = dp.get_state()
        assert _to_np(state, bk)[0] <= 1.0
        assert _to_np(state, bk)[2] <= 5.0

    def test_from_config(self, bk):
        from shinro.plants.double_pendulum import DoublePendulum
        config = {"mass_top": 0.2, "mass_bottom": 0.3, "length_top": 1.0, "length_bottom": 0.8,
                  "dt": 0.02, "g": 9.81}
        dp = DoublePendulum.from_config(config, backend=bk)
        assert dp.m1 == 0.2
        assert dp.m2 == 0.3
        assert dp.l1 == 1.0
        assert dp.l2 == 0.8
        assert dp.dt == 0.02

    def test_invalid_config_raises(self, bk):
        import pytest

        from shinro.plants.double_pendulum import DoublePendulum
        with pytest.raises(ValueError, match="positive"):
            DoublePendulum(mass_top=-1.0, mass_bottom=0.1, length_top=0.5, length_bottom=0.5,
                           dt=0.01, backend=bk)

    def test_detector_matches(self):
        import xml.etree.ElementTree as ET

        from shinro.plants.double_pendulum import detect_double_pendulum
        two_hinge = """<mujoco><worldbody>
          <body name="a"><joint name="j1" type="hinge"/></body>
          <body name="b"><joint name="j2" type="hinge"/></body>
        </worldbody><actuator><motor name="m1" joint="j1"/><motor name="m2" joint="j2"/></actuator></mujoco>"""
        single_hinge = """<mujoco><worldbody>
          <body name="a"><joint name="j1" type="hinge"/></body>
        </worldbody><actuator><motor name="m1" joint="j1"/></actuator></mujoco>"""
        cartpole = """<mujoco><worldbody>
          <body name="cart"><joint name="slider" type="slide"/></body>
          <body name="pole"><joint name="hinge" type="hinge"/></body>
        </worldbody><actuator><motor name="m1" joint="slider"/></actuator></mujoco>"""
        assert detect_double_pendulum(ET.fromstring(two_hinge))
        assert not detect_double_pendulum(ET.fromstring(single_hinge))
        assert not detect_double_pendulum(ET.fromstring(cartpole))

    def test_physics_engine_attaches_and_reads_state(self, bk):
        """With a mock engine attached, get_state reads 2 hinge qpos/qvel."""
        from unittest.mock import MagicMock

        from shinro.plants.double_pendulum import DoublePendulum
        engine = MagicMock()
        engine.backend = bk
        engine.get_joint_qpos.side_effect = {"hinge_1": 0.1, "hinge_2": 0.2}.get
        engine.get_joint_vel.side_effect = {"hinge_1": 0.3, "hinge_2": 0.4}.get
        dp = DoublePendulum(backend=bk)
        dp.physics_engine(engine)
        assert dp._engine is engine
        state = dp.get_state()
        assert np.allclose(_to_np(state, bk), [0.1, 0.2, 0.3, 0.4])
        dp.physics_engine(None)
        assert dp._engine is None

    def test_step_with_engine_calls_actuators(self, bk):
        """step() with engine attached sets both motor ctrls and steps the engine."""
        from unittest.mock import MagicMock

        from shinro.plants.double_pendulum import DoublePendulum
        engine = MagicMock()
        engine.backend = bk
        engine.get_joint_qpos.return_value = 0.0
        engine.get_joint_vel.return_value = 0.0
        dp = DoublePendulum(backend=bk)
        dp.physics_engine(engine)
        u = bk.array([1.0, 2.0])
        state = dp.step(u)
        engine.set_joint_ctrl.assert_any_call("torque_1", 1.0)
        engine.set_joint_ctrl.assert_any_call("torque_2", 2.0)
        engine.step.assert_called_once()
        assert _to_np(state, bk).shape == (4,)


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
