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
        assert np.any(np.abs(eigs) > 1)  # discrete-time instability

    def test_get_model_upright_matches_analytic(self, bk):
        """get_model() at default upright equals the discretized Jacobians."""
        from shinro.plants.inverted_pendulum import InvertedPendulum
        pend = InvertedPendulum(mass=0.1, length=0.5, gravity=9.81, dt=0.01, backend=bk)
        A, B = pend.get_model()
        expected_A_c = np.array([
            [0.0, 1.0],
            [pend.g / pend.l, -pend.b / (pend.m * pend.l**2)],
        ])
        expected_B_c = np.array([[0.0], [1.0 / (pend.m * pend.l**2)]])
        expected_A = np.eye(2) + pend.dt * expected_A_c
        expected_B = pend.dt * expected_B_c
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
        assert np.any(np.abs(eigs) > 1)  # discrete-time instability

    def test_get_model_upright_matches_analytic(self, bk):
        """get_model() at default upright equals the discretized Jacobians."""
        from shinro.plants.cartpole import CartPole
        cp = CartPole(cart_mass=0.5, pole_mass=0.1, pole_length=0.5, gravity=9.81, dt=0.01, backend=bk)
        M, m, pole_len, g, b = cp.M, cp.m, cp.l, cp.g, cp.b
        A, B = cp.get_model()
        expected_A_c = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -m * g / M, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, (M + m) * g / (M * pole_len), -b / (M * pole_len**2)],
        ])
        expected_B_c = np.array([[0.0], [1.0 / M], [0.0], [-1.0 / (M * pole_len)]])
        expected_A = np.eye(4) + cp.dt * expected_A_c
        expected_B = cp.dt * expected_B_c
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
        """At rest the linearized model matches the discretized Jacobian.

        At theta=0, omega=0, the manipulator equation gives
        thetaddot = M(0)^{-1} (tau - C*omega - G(0)), where C*omega = 0 and
        G(0) = 0. So B rows 2-3 equal M(0)^{-1} and A rows 2-3 (theta cols)
        equal -M(0)^{-1} dG/dtheta|_0, then Euler-discretized at the plant's
        dt.
        """
        dp = self._make(bk)
        m1, m2, l1, l2, g = dp.m1, dp.m2, dp.l1, dp.l2, dp.g
        M0 = np.array([
            [(m1 + m2) * l1**2, m2 * l1 * l2],
            [m2 * l1 * l2, m2 * l2**2],
        ])
        dG_dtheta = np.diag([(m1 + m2) * g * l1, m2 * g * l2])
        Minv = np.linalg.inv(M0)
        expected_A_c = np.zeros((4, 4))
        expected_A_c[0, 2] = 1.0
        expected_A_c[1, 3] = 1.0
        expected_A_c[2:, :2] = -Minv @ dG_dtheta
        expected_B_c = np.zeros((4, 2))
        expected_B_c[2:, :] = Minv
        expected_A = np.eye(4) + dp.dt * expected_A_c
        expected_B = dp.dt * expected_B_c
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

    ARM_AND_BASE_XML = """<mujoco model="mobile_manipulator">
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

    def test_detect_arm_and_base(self, tmp_path):
        import xml.etree.ElementTree as ET

        from scripts.generate_robot_config import detect_plant_types
        root = ET.fromstring(self.ARM_AND_BASE_XML)
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

    def test_generate_arm_and_base_config(self, tmp_path):
        from scripts.generate_robot_config import generate_config
        xml_file = tmp_path / "mobile_manipulator.xml"
        xml_file.write_text(self.ARM_AND_BASE_XML)
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


class TestBatchCapableDynamics:
    """``Plant.dynamics`` accepts a single state or a batch, consistently.

    Every nonlinear plant must be batch-capable — that is what lets the *same*
    function serve the eager per-sample rollout, the finite-difference
    linearization, and the lowered MPPI graph (see ``Plant.dynamics``). These
    tests pin the rank contract and the batch/single agreement, so a new
    nonlinear plant cannot silently ship with a scalar-only ``dynamics``.
    """

    def _plants(self, bk):
        from shinro.plants.cartpole import CartPole
        from shinro.plants.double_pendulum import DoublePendulum
        from shinro.plants.inverted_pendulum import InvertedPendulum
        from shinro.plants.quadrotor import Quadrotor
        return {
            "inverted_pendulum": (InvertedPendulum(backend=bk), 2, 1),
            "cartpole": (CartPole(backend=bk), 4, 1),
            "double_pendulum": (DoublePendulum(backend=bk), 4, 2),
            "quadrotor": (Quadrotor(backend=bk), 12, 4),
        }

    def test_single_state_returns_single_derivative(self, bk):
        rng = np.random.default_rng(0)
        for name, (plant, n_x, n_u) in self._plants(bk).items():
            f = _to_np(plant.dynamics(bk.array(rng.normal(size=n_x)), bk.array(rng.normal(size=n_u))), bk)
            assert f.shape == (n_x,), name

    def test_batch_matches_single_calls(self, bk):
        """A ``(N, n_x)`` call equals ``N`` single-state calls, row for row."""
        rng = np.random.default_rng(1)
        for name, (plant, n_x, n_u) in self._plants(bk).items():
            x_batch = bk.array(rng.normal(size=(5, n_x)))
            u_batch = bk.array(rng.normal(size=(5, n_u)))
            f_batch = _to_np(plant.dynamics(x_batch, u_batch), bk)
            assert f_batch.shape == (5, n_x), name
            for i in range(5):
                f_i = _to_np(plant.dynamics(x_batch[i], u_batch[i]), bk)
                assert np.allclose(f_batch[i], f_i, atol=1e-12), name

    def test_scalar_control_matches_vector(self, bk):
        """A scalar control means the first input, the rest zero."""
        rng = np.random.default_rng(2)
        for name, (plant, n_x, n_u) in self._plants(bk).items():
            x = bk.array(rng.normal(size=n_x))
            scalar = _to_np(plant.dynamics(x, 0.7), bk)
            vector = _to_np(plant.dynamics(x, bk.array([0.7] + [0.0] * (n_u - 1))), bk)
            assert np.allclose(scalar, vector, atol=1e-12), name


class TestControlMatrix:
    """``Plant.control_matrix`` — g = df/du, defaulting to FD of ``dynamics``.

    The default is what the SMC deployment path already computes by hand
    (``linearize_plant``'s B), so a plant gets a correct, lowering-capable ``g``
    with no extra code; an analytic override is a pure upgrade and the default is
    its test oracle.
    """

    def _plants(self, bk):
        from shinro.plants.cartpole import CartPole
        from shinro.plants.double_pendulum import DoublePendulum
        from shinro.plants.inverted_pendulum import InvertedPendulum
        from shinro.plants.quadrotor import Quadrotor

        # (plant, n_x, n_u, control point). The quadrotor's actuation is quadratic
        # in the rotor speeds, so its g is evaluated at a nonzero trim.
        return {
            "inverted_pendulum": (InvertedPendulum(backend=bk), 2, 1, 0.0),
            "cartpole": (CartPole(backend=bk), 4, 1, 0.0),
            "double_pendulum": (DoublePendulum(backend=bk), 4, 2, 0.0),
            "quadrotor": (Quadrotor(backend=bk), 12, 4, 3.0),
        }

    def test_shape_is_n_x_by_n_u(self, bk):
        for name, (plant, n_x, n_u, u) in self._plants(bk).items():
            x = bk.array(np.linspace(0.1, 0.4, n_x))
            g = plant.control_matrix(x, bk.array(np.full(n_u, u)))
            assert _to_np(g, bk).shape == (n_x, n_u), name

    def test_matches_linearize_plant(self, bk):
        """The default reproduces ``linearize_plant``'s B (the SMC host's g today)."""
        from shinro.utils.linearization import linearize_plant

        for name, (plant, n_x, n_u, u) in self._plants(bk).items():
            x = bk.array(np.linspace(0.1, 0.4, n_x))
            ctrl = bk.array(np.full(n_u, u))
            b_ref = np.asarray(_to_np(linearize_plant(plant, x, ctrl)[1], bk), dtype=float)
            got = np.asarray(_to_np(plant.control_matrix(x, ctrl), bk), dtype=float)
            # torch evaluates the default by autograd, the reference by central FD.
            assert np.allclose(got, b_ref, atol=1e-6), name

    def test_numpy_default_is_bit_identical_to_linearize_plant(self):
        """On numpy both sides are the same central FD — bit-identical.

        This is what lets a compiled SMC graph keep gate A at exactly 0.0 against
        the live/host model instead of needing a finite-difference tolerance.
        """
        from shinro.utils.array_backend import NumpyBackend
        from shinro.utils.linearization import linearize_plant

        for name, (plant, n_x, n_u, u) in self._plants(NumpyBackend()).items():
            x = np.linspace(0.1, 0.4, n_x)
            ctrl = np.full(n_u, u)
            b_ref = np.asarray(linearize_plant(plant, x, ctrl)[1], dtype=float)
            got = np.asarray(plant.control_matrix(x, ctrl), dtype=float)
            assert np.array_equal(got, b_ref), name

    def test_traced_default_matches_eager(self):
        """In-graph the default lowers to nodes and agrees with the eager call.

        The backend probes with a (2*n_u, n_u) block whose batch axis is shared
        with the state, so the default must broadcast the state onto it rather
        than evaluate it alone.
        """
        from shinro.codegen.interpreter import interpret
        from shinro.codegen.trace_backend import TraceBackend
        from shinro.codegen.tracing import Graph, Tracer
        from shinro.utils.array_backend import NumpyBackend

        for name, (plant, n_x, n_u, u) in self._plants(NumpyBackend()).items():
            x = np.linspace(0.1, 0.4, n_x)
            ctrl = np.full(n_u, u)
            g = Graph()
            tb = TraceBackend(g)
            x_in = Tracer(g, (n_x,), g.input("x", (n_x,)))
            gx = plant.control_matrix(x_in, ctrl, bk=tb)
            g.output("g", gx.node)
            traced = interpret(g, {"x": x})["g"]
            eager = np.asarray(plant.control_matrix(x, ctrl), dtype=float)
            assert traced.shape == (n_x, n_u), name
            assert np.allclose(traced, eager, atol=1e-12), name

    def test_force_torque_plants_ignore_the_control_point(self, bk):
        """Where u enters affinely, g does not depend on u (the quadrotor is not: """
        rng = np.random.default_rng(3)
        for name, (plant, n_x, n_u, _u) in self._plants(bk).items():
            if name == "quadrotor":
                continue  # thrust ~ w^2: g(., u) is control-dependent, zero at u = 0
            x = bk.array(rng.normal(size=n_x))
            at_zero = np.asarray(_to_np(plant.control_matrix(x, bk.zeros(n_u)), bk), dtype=float)
            elsewhere = np.asarray(
                _to_np(plant.control_matrix(x, bk.array(rng.normal(size=n_u))), bk), dtype=float
            )
            assert np.allclose(at_zero, elsewhere, atol=1e-9), name

    def test_without_dynamics_is_loud(self, bk):
        """A plant with no dynamics to differentiate is told to override, not silently wrong."""
        import pytest

        from shinro.components import Plant

        class Bare(Plant):
            def get_state(self, *args, **kwargs):
                return bk.zeros(1)

            def get_model(self, *args, **kwargs):
                return None

            def step(self, *args, **kwargs):
                return None

            def physics_engine(self, *args, **kwargs):
                return None

        with pytest.raises(NotImplementedError, match="no dynamics"):
            Bare().control_matrix(bk.zeros(1), bk.zeros(1))


class TestQuadrotor:
    """Verify quadrotor: rotor mixing, hover equilibrium, 12-state dynamics, linearization, step, from_config."""

    def _make(self, bk, **kw):
        from shinro.plants.quadrotor import Quadrotor
        return Quadrotor(backend=bk, **kw)

    def test_dynamics_shape(self, bk):
        q = self._make(bk)
        dx = q.dynamics(bk.zeros(12), bk.array([1.0, 1.0, 1.0, 1.0]))
        assert _to_np(dx, bk).shape == (12,)

    def test_hover_is_equilibrium(self, bk):
        """Collective thrust balancing gravity at level attitude is a fixed point."""
        q = self._make(bk)
        w = np.sqrt(q.m * q.g / (4.0 * q.k))
        dx = _to_np(q.dynamics(bk.zeros(12), bk.array([w, w, w, w])), bk)
        assert np.allclose(dx, 0.0, atol=1e-10)

    def test_collective_thrust_lifts(self, bk):
        """Level attitude: only vertical acceleration is nonzero, above hover it is up."""
        q = self._make(bk)
        w = np.sqrt(q.m * q.g / (4.0 * q.k))
        dx = _to_np(q.dynamics(bk.zeros(12), bk.array([1.4 * w] * 4)), bk)
        assert np.allclose(dx[:8], 0.0, atol=1e-10)
        assert np.allclose(dx[9:], 0.0, atol=1e-10)
        assert dx[8] > 0.0

    def test_pitch_tilts_thrust_forward(self, bk):
        """Positive pitch accelerates along world +x (zero roll/yaw)."""
        q = self._make(bk)
        w = np.sqrt(q.m * q.g / (4.0 * q.k))
        x = bk.zeros(12)
        x[4] = 0.2
        dx = _to_np(q.dynamics(x, bk.array([w] * 4)), bk)
        assert dx[6] > 0.0
        assert abs(dx[7]) < 1e-10

    def test_roll_tilts_thrust_sideways(self, bk):
        """Positive roll accelerates along world -y (zero pitch/yaw)."""
        q = self._make(bk)
        w = np.sqrt(q.m * q.g / (4.0 * q.k))
        x = bk.zeros(12)
        x[3] = 0.2
        dx = _to_np(q.dynamics(x, bk.array([w] * 4)), bk)
        assert abs(dx[6]) < 1e-10
        assert dx[7] < 0.0

    def test_euler_rate_map(self, bk):
        """Euler rates follow eta_dot = T(roll, pitch) @ [p, q, r] (the T helper)."""
        q = self._make(bk)
        roll, pitch, p, qrate, rrate = 0.3, 0.2, 0.5, -0.4, 0.7
        x = bk.zeros(12)
        x[3], x[4], x[9], x[10], x[11] = roll, pitch, p, qrate, rrate
        dx = _to_np(q.dynamics(x, bk.array([0.0] * 4)), bk)
        T = np.asarray(q._angular_transformation_matrix(roll, pitch))
        assert np.allclose(dx[3:6], T @ np.array([p, qrate, rrate]), atol=1e-10)

    def test_thrust_direction_matches_rotation_matrix(self, bk):
        """The closed-form thrust components equal R(roll, pitch, yaw) @ [0, 0, F]."""
        q = self._make(bk)
        roll, pitch, yaw = 0.3, -0.2, 0.7
        w = np.sqrt(q.m * q.g / (4.0 * q.k))
        x = bk.zeros(12)
        x[3], x[4], x[5] = roll, pitch, yaw
        dx = _to_np(q.dynamics(x, bk.array([w] * 4)), bk)
        thrust = q.k * 4.0 * w * w
        R = np.asarray(q._rotation_matrix(roll, pitch, yaw))
        expected = (R @ np.array([0.0, 0.0, thrust])) / q.m - np.array([0.0, 0.0, q.g])
        assert np.allclose(dx[6:9], expected, atol=1e-9)

    def test_rotor_torques(self, bk):
        """At zero rates, angular accelerations equal the rotor-mixed torques over inertia."""
        q = self._make(bk)
        u = bk.array([1.0, 2.0, 3.0, 4.0])
        dx = _to_np(q.dynamics(bk.zeros(12), u), bk)
        s1, s2, s3, s4 = 1.0**2, 2.0**2, 3.0**2, 4.0**2
        assert np.allclose(dx[9], q.k * q.r * (s2 - s4) / q.I[0], atol=1e-12)
        assert np.allclose(dx[10], q.k * q.r * (s3 - s1) / q.I[1], atol=1e-12)
        assert np.allclose(dx[11], q.b * (s2 + s4 - s1 - s3) / q.I[2], atol=1e-12)

    def test_gyroscopic_coupling(self, bk):
        """Zero rotor speeds: the omega x I omega term drives the body rates."""
        q = self._make(bk)
        p, qrate, rrate = 0.4, -0.3, 0.6
        x = bk.zeros(12)
        x[9], x[10], x[11] = p, qrate, rrate
        dx = _to_np(q.dynamics(x, bk.array([0.0] * 4)), bk)
        i_xx, i_yy, i_zz = q.I
        assert np.allclose(dx[9], qrate * rrate * (i_yy - i_zz) / i_xx, atol=1e-12)
        assert np.allclose(dx[10], rrate * p * (i_zz - i_xx) / i_yy, atol=1e-12)
        assert np.allclose(dx[11], p * qrate * (i_xx - i_yy) / i_zz, atol=1e-12)

    def test_position_derivative_is_velocity(self, bk):
        q = self._make(bk)
        x = bk.zeros(12)
        x[6], x[7], x[8] = 1.0, -2.0, 0.5
        dx = _to_np(q.dynamics(x, bk.array([0.0] * 4)), bk)
        assert np.allclose(dx[:3], [1.0, -2.0, 0.5], atol=1e-12)

    def test_batch_matches_single(self, bk):
        rng = np.random.default_rng(3)
        q = self._make(bk)
        x_batch = bk.array(rng.normal(size=(6, 12)))
        u_batch = bk.array(rng.normal(size=(6, 4)))
        f_batch = _to_np(q.dynamics(x_batch, u_batch), bk)
        for i in range(6):
            assert np.allclose(f_batch[i], _to_np(q.dynamics(x_batch[i], u_batch[i]), bk), atol=1e-12)

    def test_get_model_shapes(self, bk):
        q = self._make(bk)
        A, B = q.get_model()
        assert _to_np(A, bk).shape == (12, 12)
        assert _to_np(B, bk).shape == (12, 4)

    def test_get_model_hover_input_jacobian(self, bk):
        """At hover: collective thrust drives z, rotor differences drive the body torques."""
        q = self._make(bk)
        _, B = q.get_model()
        B = _to_np(B, bk)
        w = np.sqrt(q.m * q.g / (4.0 * q.k))
        assert np.allclose(B[8], (2.0 * q.k * w / q.m) * q.dt, atol=1e-6)
        assert np.allclose(B[9, [0, 2]], 0.0, atol=1e-6)
        assert B[9, 1] > 0.0 and B[9, 3] < 0.0
        assert B[10, 2] > 0.0 and B[10, 0] < 0.0
        assert B[11, 1] > 0.0 and B[11, 3] > 0.0
        assert B[11, 0] < 0.0 and B[11, 2] < 0.0

    def test_get_model_default_matches_explicit_hover(self, bk):
        q = self._make(bk)
        w = np.sqrt(q.m * q.g / (4.0 * q.k))
        A_default, B_default = q.get_model()
        A_explicit, B_explicit = q.get_model(bk.zeros(12), bk.array([w] * 4))
        assert np.allclose(_to_np(A_default, bk), _to_np(A_explicit, bk), atol=1e-9)
        assert np.allclose(_to_np(B_default, bk), _to_np(B_explicit, bk), atol=1e-9)

    def test_get_state_returns_copy(self, bk):
        q = self._make(bk)
        state = q.get_state()
        state[0] = 99.0
        assert _to_np(q.get_state(), bk)[0] != 99.0

    def test_step_integrates_velocity_into_position(self, bk):
        q = self._make(bk)
        w = np.sqrt(q.m * q.g / (4.0 * q.k))
        q.state = bk.zeros(12)
        q.state[6] = 1.0
        new = _to_np(q.step(bk.array([w] * 4)), bk)
        assert new.shape == (12,)
        assert np.isclose(new[0], 1.0 * q.dt, atol=1e-12)
        assert np.isclose(new[6], 1.0, atol=1e-12)

    def test_step_engine_not_implemented(self, bk):
        import pytest

        q = self._make(bk)
        q._engine = object()  # type: ignore[assignment]  # pretend an engine was attached
        with pytest.raises(NotImplementedError):
            q.step(bk.array([1.0, 1.0, 1.0, 1.0]))

    def test_from_config(self, bk):
        from shinro.plants.quadrotor import Quadrotor
        config = {
            "mass": 0.8,
            "radius": 0.2,
            "inertia": [0.01, 0.02, 0.03],
            "thrust_coeff": 1.5,
            "torque_coeff": 0.2,
            "dt": 0.02,
            "g": 9.8,
        }
        q = Quadrotor.from_config(config, backend=bk)
        assert q.m == 0.8
        assert q.r == 0.2
        assert q.I == [0.01, 0.02, 0.03]
        assert q.k == 1.5
        assert q.b == 0.2
        assert q.dt == 0.02
        assert q.g == 9.8

    def test_from_config_state_bounds(self, bk):
        from shinro.plants.quadrotor import Quadrotor
        q = Quadrotor.from_config({"state_bounds": {"min": [-1.0] * 12, "max": [1.0] * 12}}, backend=bk)
        assert q.state_bounds is not None
        assert _to_np(q.state_bounds[0], bk).shape == (12,)

    def test_default_rotor_table_is_plus(self, bk):
        """With no ``rotors`` the mixer is the documented ``+`` layout."""
        q = self._make(bk, radius=0.2)
        assert q.rotors == ((0.2, 0.0, 1.0), (0.0, 0.2, -1.0), (-0.2, 0.0, 1.0), (0.0, -0.2, -1.0))

    def test_custom_rotor_table_drives_torques(self, bk):
        """A user mixer table relabels which rotors produce roll vs pitch.

        Here 1/3 sit on +/-y and 2/4 on +/-x, so roll comes from (1,3) and
        pitch from (2,4) — the opposite pairing to the default table.
        """
        r = 0.1
        q = self._make(bk, rotors=[(0.0, -r, 1.0), (-r, 0.0, -1.0), (0.0, r, 1.0), (r, 0.0, -1.0)])
        u = bk.array([1.0, 2.0, 3.0, 4.0])
        dx = _to_np(q.dynamics(bk.zeros(12), u), bk)
        s1, s2, s3, s4 = 1.0, 4.0, 9.0, 16.0
        # tau_x = k*sum(y_i w_i^2) -> (1,3); tau_y = -k*sum(x_i w_i^2) -> (2,4)
        assert np.allclose(dx[9], q.k * r * (s3 - s1) / q.I[0], atol=1e-12)
        assert np.allclose(dx[10], q.k * r * (s2 - s4) / q.I[1], atol=1e-12)
        assert np.allclose(dx[11], q.b * (s2 + s4 - s1 - s3) / q.I[2], atol=1e-12)

    def test_from_config_rotors(self, bk):
        from shinro.plants.quadrotor import Quadrotor
        config = {
            "rotors": [
                {"x": 0.0, "y": -0.1, "spin": 1},
                {"x": -0.1, "y": 0.0, "spin": -1},
                {"x": 0.0, "y": 0.1, "spin": 1},
                {"x": 0.1, "y": 0.0, "spin": -1},
            ]
        }
        q = Quadrotor.from_config(config, backend=bk)
        assert q.rotors == ((0.0, -0.1, 1.0), (-0.1, 0.0, -1.0), (0.0, 0.1, 1.0), (0.1, 0.0, -1.0))

    def test_invalid_rotor_count_raises(self, bk):
        import pytest

        from shinro.plants.quadrotor import Quadrotor
        with pytest.raises(ValueError, match="four rotors"):
            Quadrotor(rotors=[(0.1, 0.0, 1.0), (0.0, 0.1, -1.0), (-0.1, 0.0, 1.0)], backend=bk)

    def test_invalid_config_raises(self, bk):
        import pytest

        from shinro.plants.quadrotor import Quadrotor
        with pytest.raises(ValueError, match="inertia"):
            Quadrotor(inertia=[0.01, 0.02], backend=bk)
        with pytest.raises(ValueError, match="mass"):
            Quadrotor(mass=-1.0, backend=bk)
