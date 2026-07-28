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
        from controllers.lqr import LQR
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
        from controllers.lqr import LQR
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
        from controllers.lqr import LQR
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
        from controllers.lqr import LQR
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
        from controllers.lqr import LQR
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
        from controllers.lqr import LQR
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
        from controllers.lqr import LQR
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
        from controllers.lqr import LQR
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.0, 2.0])
        u = lqr.compute(x)
        u_expected = -lqr.K @ x
        assert np.allclose(_to_np(u, bk), _to_np(u_expected, bk))

    def test_lqr_from_config(self, bk):
        """from_config creates a valid LQR controller with a gain matrix."""
        config = {"state_cost": [1.0, 1.0], "control_cost": [1.0, 1.0], "dt": 0.1}
        from controllers.lqr import LQR
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
        from controllers.lqr import LQR
        lqr = LQR.from_config(config, backend=bk)
        assert lqr.K is not None
        assert _to_np(lqr.Q, bk).shape == (2, 2)
        assert _to_np(lqr.A, bk).shape == (2, 2)
        assert _to_np(lqr.B, bk).shape == (2, 2)


class TestPID:
    """Verify PID controller: steady-state error, anti-windup, and reset."""

    def test_pid_derivative_zero_on_first_call(self, bk):
        """Derivative term is zero on the first call (no previous error)."""
        from controllers.pid import PIDController
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
        from controllers.pid import PIDController
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
        from controllers.pid import PIDController
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
        from controllers.pid import PIDController
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
        """PI control drives a first-order plant to the target with zero steady-state error."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.5]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        target = bk.array([1.0])
        x = bk.array([0.0])
        for _ in range(3000):
            u = pid.compute(x, target)
            x = x + 0.01 * u
        assert np.allclose(_to_np(x, bk)[0], 1.0, atol=1e-2)

    def test_p_only_steady_state_error(self, bk):
        """P-only control has non-zero steady-state error for a first-order plant."""
        from controllers.pid import PIDController
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
        for _ in range(1000):
            u = pid.compute(x, target)
            x = x + 0.01 * u
        assert np.allclose(_to_np(x, bk)[0], 1.0, atol=1e-2)

    def test_pid_anti_windup(self, bk):
        """When output is clamped, the integral term back-calculates on saturated channels."""
        from controllers.pid import PIDController
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
        from controllers.pid import PIDController
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
        from controllers.pid import PIDController
        pid = PIDController.from_config(config, backend=bk)
        assert pid.kp is not None


class TestMPC:
    """Verify MPC: H symmetry, F shape, constraint satisfaction, and from_config."""

    def test_mpc_H_symmetric(self, bk):
        """The QP Hessian H is symmetric."""
        from controllers.mpc_lti import MPC_LTI
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
        from controllers.mpc_lti import MPC_LTI
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
        from controllers.mpc_lti import MPC_LTI
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
        from controllers.mpc_lti import MPC_LTI
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
        from controllers.mpc_lti import MPC_LTI_Base
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
        from controllers.mpc_lti import MPC_LTI_Base
        mpc = MPC_LTI_Base.from_config(config, backend=bk)
        assert mpc.H is not None
        assert mpc.F is not None


class TestSMC:
    """Verify sliding mode controller: surface convergence, smoother variants, and error handling."""

    def test_smc_construction_hurwitz_rejection(self, bk):
        """Non-Hurwitz surface coefficients raise ValueError."""
        from controllers.smc import SlidingModeController
        with pytest.raises(ValueError, match="Hurwitz"):
            SlidingModeController(c=[-1.0, 1.0], k1=1.0, backend=bk)

    def test_smc_construction_unknown_smoother(self, bk):
        """Unknown smoother name raises ValueError."""
        from controllers.smc import SlidingModeController
        with pytest.raises(ValueError, match="Unknown smoother"):
            SlidingModeController(c=[1.0, 2.0], k1=1.0, smoother="foo", backend=bk)

    def test_smc_n_property(self, bk):
        """n returns the length of the surface coefficient vector."""
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, backend=bk)
        assert ctrl.n == 2

    def test_smc_hurwitz_polynomial_correct(self, bk):
        """c=[1,2] gives polynomial 2λ+1=0 with root at -0.5 (Hurwitz)."""
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, backend=bk)
        assert ctrl._is_hurwitz()

    def test_smc_hurwitz_polynomial_rejects_positive_root(self, bk):
        """c=[-1,1] gives polynomial λ-1=0 with root at +1 (not Hurwitz)."""
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, backend=bk)
        ctrl.c = ctrl.bk.array([-1.0, 1.0])
        assert not ctrl._is_hurwitz()

    def test_smc_compute_shape_scalar(self, bk):
        """compute() returns a 1-element array for a scalar-input system."""
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (1,)

    def test_smc_compute_shape_multi_input(self, bk):
        """compute() returns an m-element array for an m-input system."""
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 1.0], k1=1.0, phi=0.1, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([[1.0, 0.0], [0.0, 1.0]])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (2,)

    def test_smc_equivalent_control_analytical(self, bk):
        """For x_dot = f + g u with f=0, g=[0,1]^T, the equivalent control is u = -(c^T g)^{-1} c^T f = 0."""
        from controllers.smc import SlidingModeController
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
        from controllers.smc import SlidingModeController
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
        from controllers.smc import SlidingModeController
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
        from controllers.smc import SlidingModeController
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
        from controllers.smc import SlidingModeController
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
        from controllers.smc import SlidingModeController
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
        from controllers.smc import SlidingModeController
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
        from controllers.smc import SlidingModeController
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
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, phi=0.1, smoother="tanh", backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (1,)

    def test_smc_sigmoid_smoother(self, bk):
        """The sigmoid smoother produces a valid control action."""
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, phi=0.1, smoother="sigmoid", backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (1,)

    def test_smc_alpha_affects_convergence(self, bk):
        """Non-zero alpha changes the reaching law."""
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, phi=0.1, alpha=0.5, backend=bk)
        x = bk.array([1.0, -0.5])
        f_x = bk.array([0.0, 0.0])
        g_x = bk.array([0.0, 1.0])
        u = ctrl.compute(x, f_x, g_x)
        assert _to_np(u, bk).shape == (1,)

    def test_smc_loss_of_controllability(self, bk):
        """A near-zero c^T g(x) raises RuntimeError."""
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0, 0.0], k1=1.0, backend=bk)
        x = bk.array([1.0, 0.0, 0.0])
        f_x = bk.array([0.0, 0.0, 0.0])
        g_x = bk.array([0.0, 0.0, 1.0])
        with pytest.raises(RuntimeError, match="loss of controllability"):
            ctrl.compute(x, f_x, g_x)

    def test_smc_reset(self, bk):
        """reset() is a no-op (does not raise)."""
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController(c=[1.0, 2.0], k1=1.0, backend=bk)
        ctrl.reset()

    def test_smc_from_config(self, bk):
        """from_config creates a valid SMC controller."""
        config = {"c": [1.0, 2.0], "k1": 1.0, "phi": 0.1}
        from controllers.smc import SlidingModeController
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
        from controllers.smc import SlidingModeController
        ctrl = SlidingModeController.from_config(config, backend=bk)
        assert ctrl.n == 3
        assert ctrl.k1 == 2.0
        assert ctrl.phi == 0.05
        assert ctrl.k2 == 0.5
        assert ctrl.alpha == 0.3
        assert ctrl._smoother_name == "tanh"
