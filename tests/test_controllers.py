import numpy as np
import pytest


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestLQR:
    """Verify LQR gain computation against analytical DARE solution."""

    def test_lqr_gain_stabilizes_1d(self, bk):
        """Closed-loop A - B @ K has eigenvalues inside the unit circle."""
        A = bk.eye(1)
        B = bk.eye(1)
        Q = bk.eye(1)
        R = bk.eye(1)
        from shinro.controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        K = lqr.K
        A_cl = A - B @ K
        eigs = np.linalg.eigvals(_to_np(A_cl, bk))
        assert np.all(np.abs(eigs) < 1)

    def test_lqr_gain_analytical_1d(self, bk):
        """For A=1, B=1, Q=1, R=1, the DARE solution is P = (1+sqrt(5))/2 and K = P/(1+P)."""
        A = bk.eye(1)
        B = bk.eye(1)
        Q = bk.eye(1)
        R = bk.eye(1)
        from shinro.controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        K = _to_np(lqr.K, bk)[0, 0]
        P_expected = (1 + np.sqrt(5)) / 2
        K_expected = P_expected / (1 + P_expected)
        assert np.allclose(K, K_expected, atol=1e-10)

    def test_lqr_dare_residual(self, bk):
        """The DARE residual ||A^T P A - P - A^T P B (R + B^T P B)^{-1} B^T P A + Q|| is near zero."""
        A = bk.eye(1)
        B = bk.eye(1)
        Q = bk.eye(1)
        R = bk.eye(1)
        from shinro.controllers.lqr import LQR
        LQR(Q, R, A, B, backend=bk)
        from scipy.linalg import solve_discrete_are
        P = solve_discrete_are(_to_np(A, bk), _to_np(B, bk), _to_np(Q, bk), _to_np(R, bk))
        residual = A.T @ P @ A - P - A.T @ P @ B @ np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A) + Q
        assert np.linalg.norm(residual) < 1e-10

    def test_lqr_gain_formula(self, bk):
        """K = (R + B^T P B)^{-1} B^T P A matches the computed gain."""
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = 0.5 * bk.eye(2)
        from shinro.controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        from scipy.linalg import solve_discrete_are
        P_np = solve_discrete_are(_to_np(A, bk), _to_np(B, bk), _to_np(Q, bk), _to_np(R, bk))
        P = bk.from_numpy(P_np)
        K_expected = bk.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        assert np.allclose(_to_np(lqr.K, bk), _to_np(K_expected, bk), atol=1e-10)

    def test_lqr_closed_loop_eigenvalues(self, bk):
        """Closed-loop A - B K has all eigenvalues strictly inside the unit circle."""
        A = bk.array([[0.9, 0.1], [0.0, 0.8]])
        B = bk.array([[0.0], [0.1]])
        Q = bk.eye(2)
        R = bk.array([[0.1]])
        from shinro.controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        A_cl = A - B @ lqr.K
        eigs = np.linalg.eigvals(_to_np(A_cl, bk))
        assert np.all(np.abs(eigs) < 1 - 1e-6)

    def test_lqr_optimal_control_law(self, bk):
        """compute(x) returns u = -K @ x exactly (regulation to zero)."""
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.eye(2)
        from shinro.controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.5, -0.7])
        u = lqr.compute(x)
        u_expected = -lqr.K @ x
        assert np.allclose(_to_np(u, bk), _to_np(u_expected, bk), atol=1e-12)

    def test_lqr_compute_shape(self, bk):
        """compute() returns a control vector of dimension n_u."""
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.eye(2)
        from shinro.controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.0, 2.0])
        u = lqr.compute(x)
        assert _to_np(u, bk).shape == (2,)

    def test_lqr_regulation_to_zero(self, bk):
        """compute(x) with no target returns u = -K @ x (regulation to origin)."""
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.eye(2)
        from shinro.controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.0, 2.0])
        u = lqr.compute(x)
        u_expected = -lqr.K @ x
        assert np.allclose(_to_np(u, bk), _to_np(u_expected, bk))

    def test_lqr_from_config(self, bk):
        """from_config creates a valid LQR controller with a gain matrix."""
        config = {"state_cost": [1.0, 1.0], "control_cost": [1.0, 1.0], "dt": 0.1}
        from shinro.controllers.lqr import LQR
        lqr = LQR.from_config(config, backend=bk)
        assert lqr.K is not None

    def test_lqr_from_config_full_matrix(self, bk):
        """from_config accepts full Q/R matrices and custom A/B dynamics."""
        config = {
            "state_cost": [[2.0, 0.5], [0.5, 1.0]],
            "control_cost": [[0.1, 0.0], [0.0, 0.2]],
            "A_dynamics": [[0.9, 0.1], [0.0, 0.9]],
            "B_dynamics": [[0.0, 0.1], [0.1, 0.0]],
            "dt": 0.1,
        }
        from shinro.controllers.lqr import LQR
        lqr = LQR.from_config(config, backend=bk)
        assert lqr.K is not None
        assert _to_np(lqr.Q, bk).shape == (2, 2)
        assert _to_np(lqr.A, bk).shape == (2, 2)
        assert _to_np(lqr.B, bk).shape == (2, 2)


class TestPID:
    """Verify PID controller: steady-state error, anti-windup, and reset."""

    def test_pid_derivative_zero_on_first_call(self, bk):
        """Derivative term is zero on the first call (no previous error)."""
        from shinro.controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.0]),
            kd=bk.array([1.0]),
            dt=0.1,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0]), bk.array([0.0]))
        p_term = 1.0 * (0.0 - 1.0)
        assert np.allclose(_to_np(u, bk)[0], p_term)

    def test_pid_derivative_on_second_call(self, bk):
        """Derivative term on the second call is kd * (e_k - e_{k-1}) / dt."""
        from shinro.controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([0.0]),
            ki=bk.array([0.0]),
            kd=bk.array([2.0]),
            dt=0.1,
            backend=bk,
        )
        pid.compute(bk.array([1.0]), bk.array([0.0]))
        u = pid.compute(bk.array([0.5]), bk.array([0.0]))
        d_term = 2.0 * ((0.0 - 0.5) - (0.0 - 1.0)) / 0.1
        assert np.allclose(_to_np(u, bk)[0], d_term)

    def test_pid_integral_accumulates(self, bk):
        """The integral term accumulates error over successive calls."""
        from shinro.controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([0.0]),
            ki=bk.array([1.0]),
            kd=bk.array([0.0]),
            dt=0.1,
            backend=bk,
        )
        u1 = pid.compute(bk.array([1.0]), bk.array([0.0]))
        u2 = pid.compute(bk.array([1.0]), bk.array([0.0]))
        i1 = (0.0 - 1.0) * 0.1
        i2 = i1 + (0.0 - 1.0) * 0.1
        assert np.allclose(_to_np(u1, bk)[0], i1)
        assert np.allclose(_to_np(u2, bk)[0], i2)

    def test_pid_output_limits_clamp(self, bk):
        """Output limits clamp the control effort to [min, max]."""
        from shinro.controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([10.0]),
            ki=bk.array([0.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            output_limits=(bk.array([-0.5]), bk.array([0.5])),
            backend=bk,
        )
        u = pid.compute(bk.array([1.0]), bk.array([0.0]))
        assert np.allclose(_to_np(u, bk)[0], -0.5)

    def test_pi_eliminates_steady_state_error(self, bk):
        """PI control drives a first-order lag plant to the target with zero steady-state error."""
        from shinro.controllers.pid import PIDController
        # First-order lag: x_{k+1} = a*x + b*dt*u, tau=0.1s, dt=0.01s.
        a = float(np.exp(-0.01 / 0.1))
        b = 1.0
        pid = PIDController(
            kp=bk.array([2.0]),
            ki=bk.array([5.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        target = bk.array([1.0])
        x = bk.array([0.0])
        for _ in range(5000):
            u = pid.compute(x, target)
            x = a * x + b * 0.01 * u
        assert np.allclose(_to_np(x, bk)[0], 1.0, atol=1e-2)

    def test_p_only_steady_state_error(self, bk):
        """P-only control leaves a non-zero steady-state error for a first-order lag plant."""
        from shinro.controllers.pid import PIDController
        # First-order lag: x_{k+1} = a*x + b*dt*u, tau=0.1s, dt=0.01s.
        a = float(np.exp(-0.01 / 0.1))
        b = 1.0
        Kp = 2.0
        pid = PIDController(
            kp=bk.array([Kp]),
            ki=bk.array([0.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        target = bk.array([1.0])
        x = bk.array([0.0])
        for _ in range(2000):
            u = pid.compute(x, target)
            x = a * x + b * 0.01 * u
        # Closed-form steady state: x* = (b*dt*Kp)/(1 - a + b*dt*Kp) * target.
        x_ss = (b * 0.01 * Kp) / (1 - a + b * 0.01 * Kp)
        x_np = _to_np(x, bk)[0]
        assert np.isclose(x_np, x_ss, atol=1e-3)
        assert not np.isclose(x_np, 1.0, atol=1e-2)

    def test_pid_anti_windup(self, bk):
        """When output is clamped, the integral term back-calculates on saturated channels."""
        from shinro.controllers.pid import PIDController
        lo = bk.array([-0.5])
        hi = bk.array([0.5])
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([10.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            output_limits=(lo, hi),
            backend=bk,
        )
        target = bk.array([10.0])
        x = bk.array([0.0])
        for _ in range(200):
            u = pid.compute(x, target)
            x = x + 0.01 * u
        assert np.allclose(_to_np(u, bk)[0], 0.5, atol=1e-3)

    def test_pid_reset(self, bk):
        """reset() clears the integral accumulator and previous error."""
        from shinro.controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([1.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        pid.compute(bk.array([1.0]), bk.array([0.0]))
        pid.reset()
        assert np.allclose(_to_np(pid._integral, bk), 0.0)
        assert np.allclose(_to_np(pid._prev_error, bk), 0.0)
        assert not pid.has_run

    def test_pid_from_config(self, bk):
        """from_config creates a valid PID controller."""
        config = {"kp": [1.0], "ki": [0.5], "kd": [0.1], "dt": 0.01}
        from shinro.controllers.pid import PIDController
        pid = PIDController.from_config(config, backend=bk)
        assert pid.kp is not None


class TestMPC:
    """Verify MPC: H symmetry, F shape, constraint satisfaction, and from_config."""

    def test_mpc_H_symmetric(self, bk):
        """The QP Hessian H is symmetric."""
        from shinro.controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        H = _to_np(mpc.H, bk)
        assert np.allclose(H, H.T)

    def test_mpc_F_shape(self, bk):
        """The QP linear term F has shape (n_x, N * n_u)."""
        from shinro.controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        F = _to_np(mpc.F, bk)
        assert F.shape == (n, 5 * m)

    def test_mpc_compute_shape(self, bk):
        """compute() returns a control vector of dimension n_u."""
        from shinro.controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        F = bk.eye(m)
        mpc.constraints(F, bk.array([1.0, 1.0]), bk.array([-1.0, -1.0]))
        x0 = bk.array([1.0, 0.0])
        u = mpc.compute(x0)
        assert _to_np(u, bk).shape == (m,)

    def test_mpc_constraints_respected(self, bk):
        """MPC respects hard input constraints |u| <= bound."""
        from shinro.controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        bound = 0.5
        F = bk.eye(m)
        mpc.constraints(F, bk.array([bound, bound]), bk.array([-bound, -bound]))
        x0 = bk.array([10.0, 10.0])
        u = mpc.compute(x0)
        u_val = _to_np(u, bk)
        assert np.all(np.abs(u_val) <= bound + 1e-4)

    def test_mpc_from_config(self, bk):
        """from_config creates a valid MPC controller with precomputed H and F."""
        config = {
            "horizon": 5,
            "state_cost": [1.0, 1.0],
            "control_cost": [1.0, 1.0],
            "dt": 0.1,
        }
        from shinro.controllers.mpc_lti import MPC_LTI_Base
        mpc = MPC_LTI_Base.from_config(config, backend=bk)
        assert mpc.H is not None
        assert mpc.F is not None

    def test_mpc_from_config_full_matrix(self, bk):
        """from_config accepts full Q/R matrices and custom A/B dynamics."""
        config = {
            "horizon": 5,
            "state_cost": [[2.0, 0.5], [0.5, 1.0]],
            "control_cost": [[0.1, 0.0], [0.0, 0.2]],
            "A_dynamics": [[0.9, 0.1], [0.0, 0.9]],
            "B_dynamics": [[0.0, 0.1], [0.1, 0.0]],
            "dt": 0.1,
        }
        from shinro.controllers.mpc_lti import MPC_LTI_Base
        mpc = MPC_LTI_Base.from_config(config, backend=bk)
        assert mpc.H is not None
        assert mpc.F is not None


class TestSMC:
    """Verify sliding mode controller: surface convergence, smoother variants, and error handling."""

    def test_smc_construction_hurwitz_rejection(self, bk):
        """Non-Hurwitz surface coefficients raise ValueError."""
        from shinro.controllers.smc import SlidingModeController
        with pytest.raises(ValueError, match="Hurwitz"):
            SlidingModeController(c=[-1.0, 1.0], k1=1.0, backend=bk)

    def test_smc_construction_unknown_smoother(self, bk):
        """Unknown smoother name raises ValueError."""
        from shinro.controllers.smc import SlidingModeController
        with pytest.raises(ValueError, match="Unknown smoother"):
            SlidingModeController(c=[1.0, 2.0], k1=1.0, smoother="foo", backend=bk)

    def test_smc_n_property(self, bk):
        """n returns the length of the surface coefficient vector."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, backend=bk)
        assert ctrl.n == 2

    def test_smc_hurwitz_polynomial_correct(self, bk):
        """c=[1,2] gives polynomial 2λ+1=0 with root at -0.5 (Hurwitz)."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, backend=bk)
        assert ctrl._is_hurwitz()

    def test_smc_hurwitz_polynomial_rejects_positive_root(self, bk):
        """c=[-1,1] gives polynomial λ-1=0 with root at +1 (not Hurwitz)."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, backend=bk)
        ctrl.c = ctrl.bk.array([-1.0, 1.0])
        assert not ctrl._is_hurwitz()

    def test_smc_compute_shape_scalar(self, bk):
        """compute() returns a 1-element array for a scalar-input system."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (1,)

    def test_smc_compute_shape_multi_input(self, bk):
        """compute() returns an m-element array for an m-input system."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 1.0], k1=1.0, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([[1.0, 0.0], [0.0, 1.0]])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (2,)

    def test_smc_equivalent_control_analytical(self, bk):
        """For x_dot = f + g u with f=0, g=[0,1]^T, the equivalent control is u = -(c^T g)^{-1} c^T f = 0."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=0.0, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        u_np = _to_np(u, bk)
        s = float(_to_np(ctrl.c @ x, bk))
        expected = -ctrl.k1 * abs(s) ** ctrl.alpha * np.clip(s / ctrl.phi, -1, 1)
        assert np.allclose(u_np[0], expected, atol=1e-10)

    def test_smc_equivalent_control_nonzero_f(self, bk):
        """For x_dot = f + g u with f=[0,1]^T, g=[0,1]^T, the control cancels f."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=0.0, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 1.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        u_np = _to_np(u, bk)
        s = float(_to_np(ctrl.c @ x, bk))
        cf = float(_to_np(ctrl.c @ f_x, bk))
        cg = float(_to_np(ctrl.c @ g_x, bk))
        smooth_s = np.clip(s / ctrl.phi, -1, 1)
        expected = (-ctrl.k1 * abs(s) ** ctrl.alpha * smooth_s - cf) / cg
        assert np.allclose(u_np[0], expected, atol=1e-10)

    def test_smc_sliding_surface_derivative_matches_desired(self, bk):
        """The actual s_dot = c^T f + c^T g u matches the desired reaching law."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.5, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        s = float(_to_np(ctrl.c @ x, bk))
        cf = float(_to_np(ctrl.c @ f_x, bk))
        cg = float(_to_np(ctrl.c @ g_x, bk))
        u_val = float(_to_np(u, bk)[0])
        s_dot_actual = cf + cg * u_val
        smooth_s = np.clip(s / ctrl.phi, -1, 1)
        s_dot_desired = -ctrl.k1 * abs(s) ** ctrl.alpha * smooth_s
        assert np.allclose(s_dot_actual, s_dot_desired, atol=1e-10)

    def test_smc_reaching_law_includes_k2_term(self, bk):
        """The reaching law includes the -k2*s term when k2 > 0."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, k2=3.0, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        s = float(_to_np(ctrl.c @ x, bk))
        cf = float(_to_np(ctrl.c @ f_x, bk))
        cg = float(_to_np(ctrl.c @ g_x, bk))
        u_val = float(_to_np(u, bk)[0])
        s_dot_actual = cf + cg * u_val
        smooth_s = np.clip(s / ctrl.phi, -1, 1)
        s_dot_desired = -ctrl.k1 * abs(s) ** ctrl.alpha * smooth_s - ctrl.k2 * s
        assert np.allclose(s_dot_actual, s_dot_desired, atol=1e-10)

    def test_smc_alpha_zero_gives_sign_law(self, bk):
        """With alpha=0, the reaching law is s_dot = -k1 * smooth(s) (|s|^0 = 1)."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=2.0, alpha=0.0, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        s = float(_to_np(ctrl.c @ x, bk))
        cf = float(_to_np(ctrl.c @ f_x, bk))
        cg = float(_to_np(ctrl.c @ g_x, bk))
        u_val = float(_to_np(u, bk)[0])
        s_dot_actual = cf + cg * u_val
        smooth_s = np.clip(s / ctrl.phi, -1, 1)
        s_dot_desired = -ctrl.k1 * smooth_s
        assert np.allclose(s_dot_actual, s_dot_desired, atol=1e-10)

    def test_smc_alpha_half_gives_sqrt_law(self, bk):
        """With alpha=0.5, the reaching law is s_dot = -k1 * |s|^0.5 * smooth(s)."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=2.0, alpha=0.5, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        s = float(_to_np(ctrl.c @ x, bk))
        cf = float(_to_np(ctrl.c @ f_x, bk))
        cg = float(_to_np(ctrl.c @ g_x, bk))
        u_val = float(_to_np(u, bk)[0])
        s_dot_actual = cf + cg * u_val
        smooth_s = np.clip(s / ctrl.phi, -1, 1)
        s_dot_desired = -ctrl.k1 * abs(s) ** 0.5 * smooth_s
        assert np.allclose(s_dot_actual, s_dot_desired, atol=1e-10)

    def test_smc_sliding_surface_converges(self, bk):
        """The sliding surface s = c^T x converges toward zero under the control law."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=2.0, phi=0.05, backend=bk)
        dt = 0.01
        x = bk.array([1.0, 0.0])
        for _ in range(2000):
            f_x = bk.array([x[1], bk.array(0.0)])
            g_x = bk.array([0.0, 1.0])
            u = ctrl.compute(x, f_x, g_x)
            x_dot = bk.array([x[1], u[0]])
            x = x + dt * x_dot
        s = float(_to_np(ctrl.c @ x, bk))
        assert abs(s) < 0.1

    def test_smc_sign_smoother_no_phi(self, bk):
        """With phi=0, the controller uses sign() and still drives s toward zero."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=2.0, phi=0.0, backend=bk)
        dt = 0.001
        x = bk.array([1.0, 0.0])
        for _ in range(3000):
            f_x = bk.array([x[1], bk.array(0.0)])
            g_x = bk.array([0.0, 1.0])
            u = ctrl.compute(x, f_x, g_x)
            x_dot = bk.array([x[1], u[0]])
            x = x + dt * x_dot
        s = float(_to_np(ctrl.c @ x, bk))
        assert abs(s) < 0.2

    def test_smc_tanh_smoother(self, bk):
        """The tanh smoother produces a valid control action."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, phi=0.1, smoother="tanh", backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (1,)

    def test_smc_sigmoid_smoother(self, bk):
        """The sigmoid smoother produces a valid control action."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, phi=0.1, smoother="sigmoid", backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (1,)

    def test_smc_alpha_affects_convergence(self, bk):
        """Non-zero alpha changes the reaching law."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, phi=0.1, alpha=0.5, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (1,)

    def test_smc_loss_of_controllability(self, bk):
        """A near-zero c^T g(x) raises RuntimeError."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0, 0.0], k1=1.0, backend=bk)
        x = bk.array([1.0, 0.0, 0.0])
        f_x = bk.array([0.0, 0.0, 0.0])
        g_x = bk.array([0.0, 0.0, 1.0])
        with pytest.raises(RuntimeError, match="loss of controllability"):
            ctrl.compute(x, f_x, g_x)

    def test_smc_reset(self, bk):
        """reset() is a no-op (does not raise)."""
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, backend=bk)
        ctrl.reset()

    def test_smc_from_config(self, bk):
        """from_config creates a valid SMC controller."""
        config = {"c": [1.0, 2.0], "k1": 1.0, "phi": 0.1}
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController.from_config(config, backend=bk)
        assert ctrl.n == 2
        assert ctrl.k1 == 1.0
        assert ctrl.phi == 0.1

    def test_smc_from_config_full(self, bk):
        """from_config accepts all optional parameters."""
        config = {
            "c": [6.0, 11.0, 6.0],
            "k1": 2.0,
            "phi": 0.05,
            "k2": 0.5,
            "smoother": "tanh",
            "alpha": 0.3,
        }
        from shinro.controllers.smc import SlidingModeController
        ctrl = SlidingModeController.from_config(config, backend=bk)
        assert ctrl.n == 3
        assert ctrl.k1 == 2.0
        assert ctrl.phi == 0.05
        assert ctrl.k2 == 0.5
        assert ctrl.alpha == 0.3
        assert ctrl._smoother_name == "tanh"


class TestMPPI:
    """Verify MPPI: softmax weighting, control penalty, receding-horizon shift, and from_config."""

    def _ctrl(self, bk, **kwargs):
        """Build a minimal MPPI controller with identity dynamics and quadratic cost."""
        from shinro.controllers.mppi import MPPIController
        params = dict(
            dynamics_fn=lambda x, u, dt: bk.copy(x),
            cost_fn=lambda x, u: bk.sum(u**2, axis=1),
            num_samples=8,
            temperature=1.0,
            dt=0.1,
            horizon=4,
            noise_sigma=[0.5],
            seed=1,
            backend=bk,
        )
        params.update(kwargs)
        return MPPIController(**params)

    def test_mppi_construction_validation(self, bk):
        """Invalid constructor parameters raise ValueError."""
        from shinro.controllers.mppi import MPPIController
        with pytest.raises(ValueError, match="num_samples"):
            MPPIController(num_samples=0, backend=bk)
        with pytest.raises(ValueError, match="temperature"):
            MPPIController(temperature=0.0, backend=bk)
        with pytest.raises(ValueError, match="horizon"):
            MPPIController(horizon=-1, backend=bk)
        with pytest.raises(ValueError, match="dt"):
            MPPIController(dt=0.0, backend=bk)

    def test_mppi_compute_shape(self, bk):
        """compute() returns a control vector of dimension D_u."""
        ctrl = self._ctrl(bk, noise_sigma=[0.5, 0.5])
        u = ctrl.compute(bk.array([1.0, 2.0]))
        assert _to_np(u, bk).shape == (2,)

    def test_mppi_compute_requires_callables(self, bk):
        """compute() without injected dynamics/cost raises RuntimeError."""
        from shinro.controllers.mppi import MPPIController
        ctrl = MPPIController(
            num_samples=5, temperature=1.0, dt=0.1, horizon=3,
            noise_sigma=[0.5], backend=bk,
        )
        with pytest.raises(RuntimeError, match="dynamics_fn and cost_fn"):
            ctrl.compute(bk.array([0.0]))

    def test_mppi_compute_respects_bounds(self, bk):
        """The returned action stays within [u_min, u_max] even under large noise."""
        bound = 0.5
        ctrl = self._ctrl(bk, num_samples=20, noise_sigma=[5.0], u_min=[-bound], u_max=[bound])
        u = ctrl.compute(bk.array([5.0, 5.0]))
        assert abs(_to_np(u, bk)[0]) <= bound + 1e-9

    def test_mppi_softmax_weights_analytical(self, bk):
        """The applied update equals the analytic softmax-weighted average of perturbations.

        With K=1, identity dynamics, nominal u=0 (first call), and cost x -> u^2,
        the total cost of rollout i is exactly epsilon_i^2 (the control penalty
        vanishes because u=0). The returned action must equal
        sum_i w_i * eps_i with w_i = softmax(-costs/lambda).
        """
        N, K = 8, 1
        ctrl = self._ctrl(bk, num_samples=N, horizon=K, noise_sigma=[1.0], seed=123)
        u = ctrl.compute(bk.array([0.0, 0.0]))
        eps = ctrl._last_epsilon[:, 0, 0]
        costs = eps**2
        beta = costs.min()
        w = np.exp(-(costs - beta) / ctrl.lam)
        w /= w.sum()
        expected_u0 = np.sum(w * eps)
        assert np.allclose(_to_np(u, bk)[0], expected_u0, atol=1e-10)

    def test_mppi_control_penalty_term(self, bk):
        """With zero stage cost, the rollout cost equals the variance-weighted control penalty.

        For a non-zero nominal sequence and cost_fn = 0, the total cost of
        rollout i reduces to exactly lam * sum_k u_k^T Sigma^{-1} eps_{i,k}.
        """
        N, K = 5, 1
        ctrl = self._ctrl(
            bk,
            num_samples=N,
            horizon=K,
            noise_sigma=[0.5],
            temperature=2.0,
            seed=7,
            cost_fn=lambda x, u: bk.zeros(x.shape[0]),
        )
        ctrl.u = np.array([[2.0]])
        ctrl.compute(bk.array([0.0, 0.0]))
        eps = ctrl._last_epsilon[:, 0, 0]
        expected = ctrl.lam * (2.0 / (0.5**2)) * eps
        assert np.allclose(ctrl._last_costs, expected, atol=1e-10)

    def test_mppi_first_action_equals_u0(self, bk):
        """The returned action is the first element of the updated nominal sequence."""
        ctrl = self._ctrl(bk, horizon=3)
        ctrl.u = np.arange(3, dtype=np.float64).reshape(3, 1) * 0.5
        u_before = ctrl.u.copy()
        u0 = ctrl.compute(bk.array([0.0, 0.0]))
        eps = ctrl._last_epsilon
        costs = ctrl._last_costs
        beta = costs.min()
        w = np.exp(-(costs - beta) / ctrl.lam)
        w /= w.sum()
        weighted_eps = np.sum(w[:, np.newaxis, np.newaxis] * eps, axis=0)
        updated = u_before + weighted_eps
        assert np.allclose(_to_np(u0, bk)[0], updated[0, 0], atol=1e-10)

    def test_mppi_nominal_sequence_shift(self, bk):
        """After compute, the nominal sequence is shifted left with the last step duplicated."""
        K = 3
        ctrl = self._ctrl(bk, horizon=K)
        ctrl.u = np.arange(K, dtype=np.float64).reshape(K, 1) * 0.5
        u_before = ctrl.u.copy()
        ctrl.compute(bk.array([0.0, 0.0]))
        eps = ctrl._last_epsilon
        costs = ctrl._last_costs
        beta = costs.min()
        w = np.exp(-(costs - beta) / ctrl.lam)
        w /= w.sum()
        weighted_eps = np.sum(w[:, np.newaxis, np.newaxis] * eps, axis=0)
        updated = u_before + weighted_eps
        expected_shift = np.concatenate([updated[1:], updated[-1:]])
        assert np.allclose(ctrl.u, expected_shift, atol=1e-10)

    def test_mppi_bounds_clamp_in_rollout(self, bk):
        """The cost function never observes a control outside the configured bounds."""
        bound = 0.3
        seen = {"max": 0.0}

        def cost_fn(x, u):
            seen["max"] = max(seen["max"], float(np.max(np.abs(_to_np(u, bk)))))
            return bk.zeros(x.shape[0])

        ctrl = self._ctrl(bk, num_samples=6, horizon=4, noise_sigma=[5.0], u_min=[-bound], u_max=[bound])
        ctrl.cost_fn = cost_fn
        ctrl.compute(bk.array([0.0, 0.0]))
        assert seen["max"] <= bound + 1e-12

    def test_mppi_regulation_to_zero(self, bk):
        """MPPI drives a first-order integrator plant to the origin under a quadratic cost."""
        dt = 0.01

        def dynamics(x, u, dt):
            return x + dt * u

        def cost(x, u):
            return bk.sum(x**2, axis=1) + 0.1 * bk.sum(u**2, axis=1)

        ctrl = self._ctrl(
            bk,
            dynamics_fn=dynamics,
            cost_fn=cost,
            num_samples=300,
            temperature=2.0,
            dt=dt,
            horizon=30,
            noise_sigma=[2.0],
            u_min=[-10.0],
            u_max=[10.0],
        )
        x = np.array([1.0])
        for _ in range(1500):
            u = ctrl.compute(bk.from_numpy(x))
            x = x + dt * _to_np(u, bk)[0]
        assert abs(x[0]) < 0.05

    def test_mppi_reset_clears_nominal_sequence(self, bk):
        """reset() zeros the nominal sequence and clears last-sample bookkeeping."""
        ctrl = self._ctrl(bk)
        ctrl.compute(bk.array([1.0, 1.0]))
        ctrl.reset()
        assert np.allclose(ctrl.u, 0.0)
        assert ctrl._last_epsilon is None
        assert ctrl._last_costs is None

    def test_mppi_from_config(self, bk):
        """from_config creates a valid MPPI controller with callables injected later."""
        config = {
            "num_samples": 50,
            "temperature": 2.0,
            "dt": 0.02,
            "horizon": 8,
            "noise_sigma": [0.4, 0.4],
            "u_min": [-1.0, -1.0],
            "u_max": [1.0, 1.0],
            "seed": 11,
        }
        from shinro.controllers.mppi import MPPIController
        ctrl = MPPIController.from_config(config, backend=bk)
        assert ctrl.N == 50
        assert ctrl.K == 8
        assert ctrl.lam == 2.0
        assert ctrl.D_u == 2
        assert ctrl.dynamics_fn is None
        assert ctrl.cost_fn is None

    def test_mppi_seed_reproducibility(self, bk):
        """The same seed produces identical control actions across instances."""
        def make():
            return self._ctrl(bk, seed=42)

        c1 = make()
        c2 = make()
        u1 = c1.compute(bk.array([1.0, 2.0]))
        u2 = c2.compute(bk.array([1.0, 2.0]))
        assert np.allclose(_to_np(u1, bk), _to_np(u2, bk), atol=1e-12)

    def test_mppi_torch_backend(self, bk):
        """The torch backend accepts torch inputs and returns torch tensors."""
        pytest.importorskip("torch")
        if not hasattr(bk, "torch"):
            pytest.skip("requires TorchBackend")
        ctrl = self._ctrl(bk)
        x0 = bk.array([1.0, 2.0])
        assert isinstance(x0, bk.torch.Tensor)
        u = ctrl.compute(x0)
        assert isinstance(u, bk.torch.Tensor)
        assert _to_np(u, bk).shape == (1,)

    def test_mppi_attach_plant_lti(self, bk):
        """attach_plant wires an LTI plant's dynamics/cost into the controller."""
        from shinro.controllers.mppi import MPPIController
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        plant = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0, radius_wheels=0.03, dt=0.02, backend=bk)
        ctrl = MPPIController(
            num_samples=8, temperature=1.0, dt=0.02, horizon=4,
            noise_sigma=[0.5, 0.5, 0.5], seed=1, backend=bk,
        )
        ctrl.attach_plant(plant)
        u = ctrl.compute(bk.array([1.0, 2.0, 0.0]))
        assert _to_np(u, bk).shape == (3,)

    def test_mppi_attach_plant_nonlinear(self, bk):
        """attach_plant wires a nonlinear plant's dynamics/cost into the controller."""
        from shinro.controllers.mppi import MPPIController
        from shinro.plants.inverted_pendulum import InvertedPendulum
        plant = InvertedPendulum(backend=bk)
        ctrl = MPPIController(
            num_samples=8, temperature=1.0, dt=0.01, horizon=4,
            noise_sigma=[0.5], seed=1, backend=bk,
        )
        ctrl.attach_plant(plant)
        u = ctrl.compute(bk.array([0.1, 0.0]))
        assert _to_np(u, bk).shape == (1,)

    def test_mppi_attach_plant_dimension_mismatch(self, bk):
        """attach_plant raises when the plant control dim disagrees with noise_sigma."""
        from shinro.controllers.mppi import MPPIController
        from shinro.plants.inverted_pendulum import InvertedPendulum
        plant = InvertedPendulum(backend=bk)
        ctrl = MPPIController(
            num_samples=8, temperature=1.0, dt=0.01, horizon=4,
            noise_sigma=[0.5, 0.5], seed=1, backend=bk,
        )
        with pytest.raises(ValueError, match="control dimension"):
            ctrl.attach_plant(plant)

    def test_mppi_tracking_with_x_ref(self, bk):
        """With x_ref set, MPPI drives a plant toward the reference."""
        from shinro.controllers.mppi import MPPIController
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        plant = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0, radius_wheels=0.03, dt=0.02, backend=bk)
        Q = bk.array([10.0, 10.0, 10.0])
        R = bk.array([0.1, 0.1, 0.1])
        ctrl = MPPIController(
            num_samples=200, temperature=1.0, dt=0.02, horizon=10,
            noise_sigma=[1.0, 1.0, 1.0], seed=1, backend=bk,
        )
        ctrl.attach_plant(plant, Q=Q, R=R)
        x_ref = bk.array([1.0, 0.0, 0.0])
        x = bk.array([0.0, 0.0, 0.0])
        for _ in range(300):
            u = ctrl.compute(x, x_ref=x_ref)
            # first-order integrator: state += u * dt (matches A=I, B=dt*I)
            x = x + 0.02 * u
        assert abs(_to_np(x, bk)[0] - 1.0) < 0.2

    def test_mppi_torch_backend_batched_ops(self, bk):
        """Torch + LTI plant: the rollout runs on torch tensors via batched matmul."""
        pytest.importorskip("torch")
        if not hasattr(bk, "torch"):
            pytest.skip("requires TorchBackend")
        from shinro.controllers.mppi import MPPIController
        from shinro.plants.holonomicmobilerobot import HolonomicMobileRobot
        plant = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0, radius_wheels=0.03, dt=0.02, backend=bk)
        ctrl = MPPIController(
            num_samples=8, temperature=1.0, dt=0.02, horizon=4,
            noise_sigma=[0.5, 0.5, 0.5], seed=1, backend=bk,
        )
        ctrl.attach_plant(plant)
        u = ctrl.compute(bk.array([1.0, 2.0, 0.0]))
        assert isinstance(u, bk.torch.Tensor)
        # the batched dynamics runs a torch matmul: verify a known rollout
        x = bk.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        u_b = bk.zeros((2, 3))
        x_next = ctrl.dynamics_fn(x, u_b, 0.02)
        assert isinstance(x_next, bk.torch.Tensor)
        assert _to_np(x_next, bk).shape == (2, 3)
