import numpy as np
import pytest


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestKalmanFilter:
    """Verify Kalman filter: estimate shape, PSD properties, convergence, and reset."""

    def test_estimate_shape(self, bk):
        """estimate() returns a state vector of shape (n_x, 1)."""
        from shinro.estimators.kalman_filter import KalmanFilter
        n, _m, p = 2, 1, 2
        A = bk.eye(n)
        B = bk.array([[1.0], [0.0]])
        Q = 0.1 * bk.eye(n)
        R = 0.1 * bk.eye(p)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0]])
        x = kf.estimate(y, u)
        assert _to_np(x, bk).shape == (n, 1)

    def test_kalman_predict_update_equations(self, bk):
        """The estimate follows the standard KF predict-update equations exactly."""
        from shinro.estimators.kalman_filter import KalmanFilter
        n = 2
        A = bk.array([[0.9, 0.1], [0.0, 0.8]])
        B = bk.array([[1.0], [0.0]])
        Q = 0.1 * bk.eye(n)
        R = 0.2 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0]])

        x_pred_manual = A @ kf.x_hat + B @ u
        P_pred_manual = A @ kf.P @ A.T + Q
        S_manual = C @ P_pred_manual @ C.T + R
        K_manual = P_pred_manual @ C.T @ bk.inv(S_manual)
        x_hat_manual = x_pred_manual + K_manual @ (y - C @ x_pred_manual)
        P_manual = (bk.eye(n) - K_manual @ C) @ P_pred_manual

        x_hat_kf = kf.estimate(y, u)
        assert np.allclose(_to_np(x_hat_kf, bk), _to_np(x_hat_manual, bk), atol=1e-12)
        assert np.allclose(_to_np(kf.P, bk), _to_np(P_manual, bk), atol=1e-12)

    def test_kalman_innovation_zero_when_perfect(self, bk):
        """When measurement matches prediction exactly, innovation is zero and x_hat unchanged."""
        from shinro.estimators.kalman_filter import KalmanFilter
        n = 2
        A = bk.eye(n)
        B = bk.zeros((n, 1))
        Q = 0.01 * bk.eye(n)
        R = 0.01 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        kf.x_hat = bk.array([[1.0], [2.0]])
        y = C @ (A @ kf.x_hat)
        u = bk.zeros((1, 1))
        x_hat_before = bk.copy(kf.x_hat)
        x_hat_after = kf.estimate(y, u)
        assert np.allclose(_to_np(x_hat_after, bk), _to_np(x_hat_before, bk), atol=1e-10)

    def test_kalman_covariance_monotonic_decrease(self, bk):
        """The trace of P decreases monotonically (or stays same) as measurements arrive."""
        from shinro.estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = 0.01 * bk.eye(n)
        R = 0.1 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        traces = []
        for _ in range(5):
            kf.estimate(y, u)
            traces.append(float(_to_np(bk.trace(kf.P), bk)))
        for i in range(1, len(traces)):
            assert traces[i] <= traces[i - 1] + 1e-10

    def test_P_remains_psd(self, bk):
        """The error covariance P remains positive semidefinite after multiple updates."""
        from shinro.estimators.kalman_filter import KalmanFilter
        n, p = 2, 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = 0.1 * bk.eye(n)
        R = 0.1 * bk.eye(p)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        for _ in range(10):
            kf.estimate(y, u)
        P = kf.P
        eigs = np.linalg.eigvals(_to_np(P, bk))
        assert np.all(eigs > -1e-10)

    def test_innovation_covariance_psd(self, bk):
        """The innovation covariance S = C P_pred C^T + R is positive semidefinite."""
        from shinro.estimators.kalman_filter import KalmanFilter
        n, p = 2, 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = 0.1 * bk.eye(n)
        R = 0.1 * bk.eye(p)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        A @ kf.x_hat + B @ u
        P_pred = A @ kf.P @ A.T + Q
        S = C @ P_pred @ C.T + R
        eigs = np.linalg.eigvals(_to_np(S, bk))
        assert np.all(eigs > -1e-10)

    def test_estimate_converges_1d(self, bk):
        """The Kalman filter estimate tracks the true state for a detectable (A, C) pair."""
        from shinro.estimators.kalman_filter import KalmanFilter
        A = bk.array([[0.9]])
        B = bk.array([[1.0]])
        Q = bk.array([[0.01]])
        R = bk.array([[0.1]])
        C = bk.array([[1.0]])
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        true_state = np.array([[5.0]])
        x = bk.zeros((1, 1))
        for _ in range(30):
            y = bk.from_numpy(true_state + 0.1 * np.random.randn(1, 1))
            u = bk.array([[0.0]])
            x = kf.estimate(y, u)
            true_state = 0.9 * true_state
        error = np.abs(_to_np(x, bk)[0, 0] - true_state[0, 0])
        assert error < 1.0

    def test_reset(self, bk):
        """reset() clears the state estimate to zero."""
        from shinro.estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = 0.1 * bk.eye(n)
        R = 0.1 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        kf.estimate(y, u)
        kf.reset()
        assert np.allclose(_to_np(kf.x_hat, bk), 0.0)

    def test_from_config_d_shape_matches_input_dim(self, bk):
        """from_config builds D as (n_y, n_u) so estimate() works when n_y != n_u."""
        from shinro.estimators.kalman_filter import KalmanFilter
        cfg = {
            "process_noise": [0.001, 0.01],
            "measurement_noise": [0.005, 0.05],
            "dt": 0.01,
            "A_dynamics": [[1.0, 0.01], [0.1962, 1.0]],
            "B_dynamics": [[0.0], [0.4]],
        }
        kf = KalmanFilter.from_config(cfg, backend=bk)
        assert _to_np(kf.D, bk).shape == (2, 1)
        y = bk.array([[0.1], [0.0]])
        u = bk.array([[0.5]])
        x = kf.estimate(y, u)
        assert _to_np(x, bk).shape == (2, 1)


class TestLuenbergerObserver:
    """Verify Luenberger observer: estimate shape, stability, convergence, and reset."""

    def test_estimate_shape(self, bk):
        """estimate() returns a state vector of shape (n_x, 1)."""
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        n, _m, _p = 2, 1, 2
        A = bk.eye(n)
        B = bk.array([[1.0], [0.0]])
        L = bk.array([[0.5, 0.0], [0.0, 0.5]])
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0]])
        x = obs.estimate(y, u)
        assert _to_np(x, bk).shape == (n, 1)

    def test_luenberger_observer_equation(self, bk):
        """The estimate follows x_hat = A x_hat + B u + L (y - C (A x_hat + B u)) exactly."""
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = bk.array([[0.9, 0.1], [0.0, 0.8]])
        B = bk.array([[1.0], [0.0]])
        L = bk.array([[0.5, 0.0], [0.0, 0.5]])
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0]])

        x_pred_manual = A @ obs.x_hat + B @ u
        x_hat_manual = x_pred_manual + L @ (y - C @ x_pred_manual)

        x_hat_obs = obs.estimate(y, u)
        assert np.allclose(_to_np(x_hat_obs, bk), _to_np(x_hat_manual, bk), atol=1e-12)

    def test_luenberger_innovation_zero_when_perfect(self, bk):
        """When measurement matches prediction, innovation is zero and x_hat unchanged."""
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = bk.eye(n)
        B = bk.zeros((n, 1))
        L = 0.5 * bk.eye(n)
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        obs.x_hat = bk.array([[1.0], [2.0]])
        y = C @ (A @ obs.x_hat)
        u = bk.zeros((1, 1))
        x_hat_before = bk.copy(obs.x_hat)
        x_hat_after = obs.estimate(y, u)
        assert np.allclose(_to_np(x_hat_after, bk), _to_np(x_hat_before, bk), atol=1e-10)

    def test_luenberger_error_dynamics_eigenvalues(self, bk):
        """The error dynamics A - L C has eigenvalues inside the unit circle."""
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = bk.array([[0.9, 0.1], [0.0, 0.8]])
        L = bk.array([[0.5, 0.0], [0.0, 0.5]])
        C = bk.eye(n)
        B = bk.eye(n)
        LuenbergerObserver(A, B, L, C=C, backend=bk)
        A_cl = A - L @ C
        eigs = np.linalg.eigvals(_to_np(A_cl, bk))
        assert np.all(np.abs(eigs) < 1)

    def test_error_dynamics_stable(self, bk):
        """The error dynamics matrix A - L @ C has all eigenvalues inside the unit circle."""
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = 0.9 * bk.eye(n)
        L = 0.5 * bk.eye(n)
        C = bk.eye(n)
        B = bk.eye(n)
        LuenbergerObserver(A, B, L, C=C, backend=bk)
        A_cl = A - L @ C
        eigs = np.linalg.eigvals(_to_np(A_cl, bk))
        assert np.all(np.abs(eigs) < 1)

    def test_estimate_converges_1d(self, bk):
        """The Luenberger observer estimate converges to the true state for stable error dynamics."""
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        A = bk.array([[0.9]])
        B = bk.array([[1.0]])
        L = bk.array([[0.5]])
        C = bk.array([[1.0]])
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        true_state = np.array([[5.0]])
        x = bk.zeros((1, 1))
        for _ in range(30):
            y = bk.from_numpy(true_state)
            u = bk.array([[0.0]])
            x = obs.estimate(y, u)
            true_state = 0.9 * true_state
        error = np.abs(_to_np(x, bk)[0, 0] - true_state[0, 0])
        assert error < 0.1

    def test_reset(self, bk):
        """reset() clears the state estimate to zero."""
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        L = 0.5 * bk.eye(n)
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        obs.estimate(y, u)
        obs.reset()
        assert np.allclose(_to_np(obs.x_hat, bk), 0.0)

    def test_from_config_d_shape_matches_input_dim(self, bk):
        """from_config builds D as (n_y, n_u) so estimate() works when n_y != n_u."""
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        cfg = {
            "observer_gain": [0.8, 0.8],
            "dt": 0.01,
            "A_dynamics": [[1.0, 0.01], [0.1962, 1.0]],
            "B_dynamics": [[0.0], [0.4]],
        }
        obs = LuenbergerObserver.from_config(cfg, backend=bk)
        assert _to_np(obs.D, bk).shape == (2, 1)
        y = bk.array([[0.1], [0.0]])
        u = bk.array([[0.5]])
        x = obs.estimate(y, u)
        assert _to_np(x, bk).shape == (2, 1)

    def test_from_config_partial_C_builds_d_by_output_dim(self, bk):
        """from_config with a partial C (n_y != n) builds D as (n_y, n_u) and estimates.

        Regression: D used the state dim n instead of n_y, so estimate()
        raised whenever C selected fewer outputs than there are states.
        """
        from shinro.estimators.luenberger_observer import LuenbergerObserver
        cfg = {
            "observer_gain": [[0.6], [0.2]],     # (n, n_y) = (2, 1)
            "A_dynamics": [[1.0, 1.0], [0.0, 1.0]],
            "B_dynamics": [[0.0], [1.0]],
            "C": [[1.0, 0.0]],                   # n_y = 1 != n = 2
        }
        obs = LuenbergerObserver.from_config(cfg, backend=bk)
        assert _to_np(obs.D, bk).shape == (1, 1)
        x = obs.estimate(bk.array([[1.0]]), bk.array([[0.0]]))
        assert _to_np(x, bk).shape == (2, 1)


class TestExtendedKalmanFilter:
    """Verify the EKF: predict-update equations, Jacobians, config, and reset.

    Flat-vector convention throughout: state (n_x,), control (n_u,),
    measurement (n_y,). ``dynamics_fn`` is continuous-time dx/dt, Euler-
    discretized at dt.
    """

    @staticmethod
    def _linear_ekf(bk, dt=0.01):
        """EKF over a linear integrator chain: f = Ac x + Bc u, h = C x.

        No analytical Jacobian is passed — the backend computes them (finite
        differences on numpy, autograd on torch).
        """
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        Ac = bk.array([[0.0, 1.0], [0.0, 0.0]])
        Bc = bk.array([[0.0], [1.0]])
        C = bk.array([[1.0, 0.0]])
        ekf = ExtendedKalmanFilter(
            lambda x, u: Ac @ x + Bc @ u,
            lambda x: C @ x,
            dt=dt,
            Q=0.01 * bk.eye(2),
            R=0.1 * bk.eye(1),
            backend=bk,
        )
        return ekf, Ac, Bc, C

    def test_estimate_shape(self, bk):
        """estimate() returns a flat state vector of shape (n_x,)."""
        ekf, *_ = self._linear_ekf(bk)
        x = ekf.estimate(bk.array([0.7]), bk.array([0.3]))
        assert _to_np(x, bk).shape == (2,)

    def test_matches_kalman_filter_linear(self, bk):
        """For linear f, h the EKF reproduces a KalmanFilter with A_d = I + dt A_c, B_d = dt B_c."""
        from shinro.estimators.kalman_filter import KalmanFilter

        dt = 0.01
        ekf, Ac, Bc, C = self._linear_ekf(bk, dt)
        kf = KalmanFilter(
            bk.eye(2) + dt * Ac, dt * Bc, ekf.Q, ekf.R, C=C, D=bk.zeros((1, 1)), backend=bk
        )
        x_ekf = ekf.estimate(bk.array([0.7]), bk.array([0.3]))
        x_kf = kf.estimate(bk.array([[0.7]]), bk.array([[0.3]]))  # KF is column-vector
        assert np.allclose(_to_np(x_ekf, bk), _to_np(x_kf, bk).ravel(), atol=1e-8)
        assert np.allclose(_to_np(ekf.P, bk), _to_np(kf.P, bk), atol=1e-8)

    def test_predict_update_equations(self, bk):
        """One EKF step follows the documented predict/update equations exactly."""
        ekf, Ac, Bc, C = self._linear_ekf(bk)
        n = 2
        x0 = bk.copy(ekf.x_hat)
        P0 = bk.copy(ekf.P)
        y, u = bk.array([0.7]), bk.array([0.3])

        x_pred = x0 + ekf.dt * (Ac @ x0 + Bc @ u)
        F = bk.eye(n) + ekf.dt * Ac
        P_pred = F @ P0 @ F.T + ekf.Q
        S = C @ P_pred @ C.T + ekf.R
        K = P_pred @ C.T @ bk.inv(S)
        x_manual = x_pred + K @ (y - C @ x_pred)
        P_manual = (bk.eye(n) - K @ C) @ P_pred

        x = ekf.estimate(y, u)
        assert np.allclose(_to_np(x, bk), _to_np(x_manual, bk), atol=1e-8)
        assert np.allclose(_to_np(ekf.P, bk), _to_np(P_manual, bk), atol=1e-8)

    def test_numerical_jacobian_matches_analytic(self, bk):
        """The backend Jacobian of f and h matches the analytic linearization."""
        ekf, Ac, _Bc, C = self._linear_ekf(bk)
        J_f = ekf._dynamics_jacobian(bk.array([0.3, -0.2]), bk.array([0.5]))
        J_h = ekf._measurement_jacobian(bk.array([0.3, -0.2]))
        assert np.allclose(_to_np(J_f, bk), _to_np(Ac, bk), atol=1e-6)
        assert np.allclose(_to_np(J_h, bk), _to_np(C, bk), atol=1e-6)

    def test_measurement_jacobian_nonlinear(self, bk):
        """H is the true local Jacobian for a nonlinear h(x) = [x0^2, x1]."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter(
            lambda x, u: bk.zeros(2),
            lambda x: bk.array([x[0] * x[0], x[1]]),
            dt=0.01,
            Q=bk.eye(2),
            R=bk.eye(2),
            backend=bk,
        )
        H = ekf._measurement_jacobian(bk.array([3.0, 5.0]))
        assert np.allclose(_to_np(H, bk), [[6.0, 0.0], [0.0, 1.0]], atol=1e-5)

    def test_converges_nonlinear_measurement(self, bk):
        """With a nonlinear h(x) = x^2 the estimate converges to the true state."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter(
            lambda x, u: bk.zeros(1),
            lambda x: bk.array([x[0] * x[0]]),
            dt=0.01,
            Q=1e-4 * bk.eye(1),
            R=0.01 * bk.eye(1),
            x0=bk.array([1.0]),
            backend=bk,
        )
        y, u = bk.array([4.0]), bk.zeros(1)
        x = ekf.x_hat
        for _ in range(60):
            x = ekf.estimate(y, u)
        assert abs(float(_to_np(x, bk)[0]) - 2.0) < 1e-3

    def test_accepts_column_vector_measurement(self, bk):
        """A (n,1) measurement is flattened, not broadcast into an (n,n) innovation."""
        ekf, *_ = self._linear_ekf(bk)
        x = ekf.estimate(bk.array([[0.7]]), bk.array([[0.3]]))
        assert _to_np(x, bk).shape == (2,)

    def test_estimate_without_callables_raises(self, bk):
        """A from_config filter with no injected callables raises a clear RuntimeError."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter.from_config(
            {"process_noise": [0.01], "measurement_noise": [0.1], "dt": 0.02}, backend=bk
        )
        with pytest.raises(RuntimeError, match="dynamics_fn"):
            ekf.estimate(bk.array([1.0]), bk.array([0.0]))

    def test_reset(self, bk):
        """reset() zeroes the estimate (and accepts a new x0)."""
        ekf, *_ = self._linear_ekf(bk)
        ekf.estimate(bk.array([0.7]), bk.array([0.3]))
        ekf.reset()
        assert np.allclose(_to_np(ekf.x_hat, bk), 0.0)
        ekf.reset(bk.array([3.0, 4.0]))
        assert np.allclose(_to_np(ekf.x_hat, bk), [3.0, 4.0])

    def test_from_config_shapes_and_defaults(self, bk):
        """from_config reads n_x from Q and n_y from R; callables start None."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter.from_config(
            {"process_noise": [0.01, 0.02, 0.03], "measurement_noise": [0.1, 0.1], "dt": 0.02},
            backend=bk,
        )
        assert ekf.n_x == 3 and ekf.n_y == 2
        assert _to_np(ekf.Q, bk).shape == (3, 3)
        assert _to_np(ekf.R, bk).shape == (2, 2)
        assert _to_np(ekf.x_hat, bk).shape == (3,)
        assert ekf.dynamics_fn is None and ekf.measurement_fn is None

    def test_from_config_full_matrix_and_initial_state(self, bk):
        """Nested lists parse as full matrices; initial_state seeds x_hat."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter.from_config(
            {
                "process_noise": [[0.01, 0.0], [0.0, 0.02]],
                "measurement_noise": [0.1],
                "dt": 0.1,
                "initial_state": [1.0, 2.0],
            },
            backend=bk,
        )
        assert np.allclose(_to_np(ekf.Q, bk), [[0.01, 0.0], [0.0, 0.02]])
        assert np.allclose(_to_np(ekf.x_hat, bk), [1.0, 2.0])

    def test_from_config_inject_and_estimate(self, bk):
        """A from_config filter works once the callables are injected."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter.from_config(
            {"process_noise": [0.01], "measurement_noise": [0.1], "dt": 0.02}, backend=bk
        )
        ekf.dynamics_fn = lambda x, u: bk.zeros(1)
        ekf.measurement_fn = lambda x: bk.array([x[0]])
        x = ekf.estimate(bk.array([1.0]), bk.zeros(1))
        assert _to_np(x, bk).shape == (1,)

    def test_from_config_requires_dt(self, bk):
        """Standalone from_config rejects a missing dt."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        with pytest.raises(ValueError, match="dt is required"):
            ExtendedKalmanFilter.from_config(
                {"process_noise": [0.01], "measurement_noise": [0.1]}, backend=bk
            )

    def test_from_config_rejects_unknown_key(self, bk):
        """Strict config parsing rejects an unknown key."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        with pytest.raises(ValueError, match="unknown key"):
            ExtendedKalmanFilter.from_config(
                {"process_noise": [0.01], "measurement_noise": [0.1], "dt": 0.1, "nope": 1},
                backend=bk,
            )

    def test_sample_config_parses(self, bk):
        """The shipped ekf_base.toml parses into a valid filter."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        cfg = ExtendedKalmanFilter.load_config("samples/estimators/ekf_base.toml")
        ekf = ExtendedKalmanFilter.from_config(cfg, backend=bk)
        assert ekf.dt == 0.02
        assert ekf.n_x == 2 and ekf.n_y == 2

    def test_registered_and_exported(self):
        """The estimator is registered and exported from the package."""
        from shinro.estimators import ExtendedKalmanFilter as Exported
        from shinro.estimators.extended_kf import ExtendedKalmanFilter
        from shinro.factories.registry import _ESTIMATOR_REGISTRY

        assert Exported is ExtendedKalmanFilter
        assert _ESTIMATOR_REGISTRY["ExtendedKalmanFilter"] is ExtendedKalmanFilter

    def test_measurement_matrix_gives_exact_h_and_H(self, bk):
        """A linear measurement_matrix supplies h(x) = C x and the exact H = C, no callable."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        C = bk.array([[1.0, 0.0], [0.0, 1.0]])
        ekf = ExtendedKalmanFilter(
            lambda x, u: bk.zeros(2),
            None,
            dt=0.01,
            Q=0.01 * bk.eye(2),
            R=0.1 * bk.eye(2),
            backend=bk,
            measurement_matrix=C,
        )
        x = bk.array([1.5, -2.0])
        assert np.allclose(_to_np(ekf._measurement(x), bk), _to_np(C @ x, bk))
        assert np.allclose(_to_np(ekf._measurement_jacobian(x), bk), _to_np(C, bk))
        out = ekf.estimate(bk.array([1.5, -2.0]), bk.zeros(2))
        assert _to_np(out, bk).shape == (2,)

    def test_measurement_matrix_shape_mismatch_is_loud(self, bk):
        """A C that does not match R's n_y and Q's n_x is rejected at construction."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        with pytest.raises(ValueError, match="measurement_matrix shape"):
            ExtendedKalmanFilter(
                lambda x, u: bk.zeros(2),
                None,
                dt=0.01,
                Q=0.01 * bk.eye(2),
                R=0.1 * bk.eye(2),
                backend=bk,
                measurement_matrix=bk.array([[1.0, 0.0]]),  # (1, 2) but R is (2, 2)
            )

    def test_from_config_parses_measurement_matrix(self, bk):
        """from_config carries measurement_matrix into the filter; a C-only filter estimates."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter.from_config(
            {
                "process_noise": [0.01, 0.01],
                "measurement_noise": [0.1, 0.1],
                "dt": 0.02,
                "measurement_matrix": [[1.0, 0.0], [0.0, 1.0]],
            },
            backend=bk,
        )
        C = ekf.C
        assert C is not None
        assert _to_np(C, bk).shape == (2, 2)
        assert ekf.measurement_fn is None
        ekf.dynamics_fn = lambda x, u: bk.zeros(2)
        assert _to_np(ekf.estimate(bk.array([1.0, 2.0]), bk.zeros(2)), bk).shape == (2,)

    def test_bk_aware_callable_receives_the_backend(self, bk):
        """A callable taking ``bk`` is given the filter's backend (the tracing route)."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        seen = []

        def dynamics(x, u, bk=None):
            seen.append(bk)
            assert bk is not None
            return bk.zeros(2)

        ekf = ExtendedKalmanFilter(dynamics, None, dt=0.01, Q=0.01 * bk.eye(2), R=0.1 * bk.eye(2),
                                    backend=bk, measurement_matrix=bk.eye(2))
        ekf.estimate(bk.array([1.0, 2.0]), bk.zeros(2))
        assert seen, "bk-aware dynamics was never called"
        assert all(s is bk for s in seen), "the filter did not forward its own backend"

    def test_attach_plant_injects_dynamics(self, bk):
        """attach_plant fills an unset dynamics_fn with the plant's model; explicit wins."""
        import tomllib as _tomllib

        from shinro.estimators.extended_kf import ExtendedKalmanFilter
        from shinro.factories.registry import _PLANT_REGISTRY
        from shinro.utils.array_backend import NumpyBackend
        from shinro.utils.config_resolver import resolve_config_path

        with open(resolve_config_path("samples/plants/cartpole.toml"), "rb") as f:
            plant = _PLANT_REGISTRY["CartPole"].from_config(_tomllib.load(f), backend=NumpyBackend())

        ekf = ExtendedKalmanFilter.from_config(
            {"process_noise": [0.01] * 4, "measurement_noise": [0.1] * 4, "dt": 0.01,
             "measurement_matrix": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]]},
            backend=bk,
        )
        assert ekf.dynamics_fn is None
        ekf.attach_plant(plant)
        assert ekf.dynamics_fn is not None
        assert _to_np(ekf.estimate(bk.array([0.1, 0.0, 0.2, 0.0]), bk.zeros(1)), bk).shape == (4,)

        explicit = lambda x, u: bk.zeros(4)  # noqa: E731
        ekf.dynamics_fn = explicit
        ekf.attach_plant(plant)
        assert ekf.dynamics_fn is explicit, "attach_plant overwrote an explicit dynamics_fn"

    def test_attach_plant_without_dynamics_is_loud(self, bk):
        """A plant exposing no dynamics and a filter with none is a loud error."""
        from shinro.estimators.extended_kf import ExtendedKalmanFilter

        ekf = ExtendedKalmanFilter(None, None, dt=0.01, Q=bk.eye(1), R=bk.eye(1), backend=bk)
        with pytest.raises(ValueError, match="no dynamics"):
            ekf.attach_plant(object())


class TestUnscentedKF:
    """Verify the UKF: sigma-point weights, UT exactness, convergence, config, reset.

    Flat-vector convention throughout: state (n_x,), control (n_u,),
    measurement (n_y,). ``dynamics_fn`` is continuous-time dx/dt, Euler-
    discretized at dt (the same contract as ExtendedKalmanFilter).
    """

    @staticmethod
    def _linear_ukf(bk, dt=0.01, alpha=1.0, beta=2.0, kappa=0.0):
        """UKF over a linear integrator chain: f = Ac x + Bc u, h = C x.

        alpha=1 / kappa=0 keeps the scaled-UKF weights well conditioned so the
        unscented transform is exact for the linear model (checked against KF).
        """
        from shinro.estimators.unscented_kf import UnscentedKF

        Ac = bk.array([[0.0, 1.0], [0.0, 0.0]])
        Bc = bk.array([[0.0], [1.0]])
        C = bk.array([[1.0, 0.0]])
        ukf = UnscentedKF(
            lambda x, u: Ac @ x + Bc @ u,
            lambda x: C @ x,
            dt=dt,
            alpha=alpha,
            beta=beta,
            kappa=kappa,
            Q=0.01 * bk.eye(2),
            R=0.1 * bk.eye(1),
            backend=bk,
        )
        return ukf, Ac, Bc, C

    def test_estimate_shape(self, bk):
        """estimate() returns a flat state vector of shape (n_x,)."""
        ukf, *_ = self._linear_ukf(bk)
        x = ukf.estimate(bk.array([0.7]), bk.array([0.3]))
        assert _to_np(x, bk).shape == (2,)

    def test_weights_normalized(self, bk):
        """Mean weights sum to 1; covariance weights sum to 2 - alpha^2 + beta."""
        ukf, *_ = self._linear_ukf(bk, alpha=1.0, beta=2.0)
        assert abs(float(_to_np(bk.sum(ukf.w_mean), bk)) - 1.0) < 1e-12
        expected = 2.0 - ukf.alpha**2 + ukf.beta
        assert abs(float(_to_np(bk.sum(ukf.w_covar), bk)) - expected) < 1e-12
        assert _to_np(ukf.w_mean, bk).shape == (2 * ukf.nx + 1,)
        assert _to_np(ukf.w_covar, bk).shape == (2 * ukf.nx + 1,)

    def test_sigma_points_recover_mean_and_covariance(self, bk):
        """The unscented transform recovers (x, P) exactly from its sigma points."""
        ukf, *_ = self._linear_ukf(bk)
        x = bk.array([0.3, -0.2])
        P = bk.array([[0.5, 0.1], [0.1, 0.4]])
        chi = ukf._sigma_points(x, P)
        mean = ukf._weighted_mean(chi)
        cov = ukf._weighted_cov(chi, mean)
        assert np.allclose(_to_np(mean, bk), _to_np(x, bk), atol=1e-10)
        assert np.allclose(_to_np(cov, bk), _to_np(P, bk), atol=1e-10)

    def test_matches_kalman_filter_linear(self, bk):
        """For linear f, h the UKF reproduces a KalmanFilter (the UT is exact for linear maps)."""
        from shinro.estimators.kalman_filter import KalmanFilter

        dt = 0.01
        ukf, Ac, Bc, C = self._linear_ukf(bk, dt=dt)
        kf = KalmanFilter(
            bk.eye(2) + dt * Ac, dt * Bc, ukf.Q, ukf.R, C=C, D=bk.zeros((1, 1)), backend=bk
        )
        x_ukf = ukf.estimate(bk.array([0.7]), bk.array([0.3]))
        x_kf = kf.estimate(bk.array([[0.7]]), bk.array([[0.3]]))  # KF is column-vector
        assert np.allclose(_to_np(x_ukf, bk), _to_np(x_kf, bk).ravel(), atol=1e-8)
        assert np.allclose(_to_np(ukf.P, bk), _to_np(kf.P, bk), atol=1e-8)

    def test_covariance_stays_psd(self, bk):
        """P stays positive semidefinite after repeated updates."""
        ukf, *_ = self._linear_ukf(bk)
        for _ in range(20):
            ukf.estimate(bk.array([0.7]), bk.array([0.3]))
        eigs = np.linalg.eigvals(_to_np(ukf.P, bk))
        assert np.all(eigs > -1e-10)

    def test_converges_nonlinear_measurement(self, bk):
        """With a nonlinear h(x) = x^2 the estimate converges to the true state."""
        from shinro.estimators.unscented_kf import UnscentedKF

        ukf = UnscentedKF(
            lambda x, u: bk.zeros(1),
            lambda x: bk.array([x[0] * x[0]]),
            dt=0.01,
            alpha=1.0,
            beta=2.0,
            kappa=0.0,
            Q=1e-4 * bk.eye(1),
            R=0.01 * bk.eye(1),
            x0=bk.array([1.0]),
            backend=bk,
        )
        y, u = bk.array([4.0]), bk.zeros(1)
        x = ukf.x_hat
        for _ in range(60):
            x = ukf.estimate(y, u)
        assert abs(float(_to_np(x, bk)[0]) - 2.0) < 1e-3

    def test_accepts_column_vector_measurement(self, bk):
        """A (n_y,1) measurement stays a flat (n_y,) innovation (no (n,n) broadcast)."""
        ukf, *_ = self._linear_ukf(bk)
        x = ukf.estimate(bk.array([[0.7]]), bk.array([[0.3]]))
        assert _to_np(x, bk).shape == (2,)

    def test_estimate_without_callables_raises(self, bk):
        """A from_config filter with no injected callables raises a clear RuntimeError."""
        from shinro.estimators.unscented_kf import UnscentedKF

        ukf = UnscentedKF.from_config(
            {"process_noise": [0.01], "measurement_noise": [0.1], "dt": 0.02}, backend=bk
        )
        with pytest.raises(RuntimeError, match="dynamics_fn"):
            ukf.estimate(bk.array([1.0]), bk.array([0.0]))

    def test_reset(self, bk):
        """reset() restores x_hat=0 / P=0.1*I and accepts a new x0."""
        ukf, *_ = self._linear_ukf(bk)
        ukf.estimate(bk.array([0.7]), bk.array([0.3]))
        ukf.reset()
        assert np.allclose(_to_np(ukf.x_hat, bk), 0.0)
        assert np.allclose(_to_np(ukf.P, bk), 0.1 * np.eye(2))
        ukf.reset(bk.array([3.0, 4.0]))
        assert np.allclose(_to_np(ukf.x_hat, bk), [3.0, 4.0])

    def test_reset_accepts_column_vector(self, bk):
        """reset(x0) flattens an (n,1) x0 so the sigma-point step cannot broadcast-fail."""
        ukf, *_ = self._linear_ukf(bk)
        ukf.reset(bk.array([[3.0], [4.0]]))
        assert _to_np(ukf.x_hat, bk).shape == (2,)
        assert np.allclose(_to_np(ukf.x_hat, bk), [3.0, 4.0])

    def test_from_config_shapes_and_defaults(self, bk):
        """from_config reads n_x from Q and n_y from R; callables start None."""
        from shinro.estimators.unscented_kf import UnscentedKF

        ukf = UnscentedKF.from_config(
            {"process_noise": [0.01, 0.02, 0.03], "measurement_noise": [0.1, 0.1], "dt": 0.02},
            backend=bk,
        )
        assert ukf.nx == 3 and ukf.ny == 2
        assert _to_np(ukf.Q, bk).shape == (3, 3)
        assert _to_np(ukf.R, bk).shape == (2, 2)
        assert _to_np(ukf.x_hat, bk).shape == (3,)
        assert _to_np(ukf.w_mean, bk).shape == (7,)
        assert (ukf.alpha, ukf.beta, ukf.kappa) == (1e-3, 2.0, 0.0)
        assert ukf.dynamics_fn is None and ukf.measurement_fn is None

    def test_from_config_full_matrix_and_initial_state(self, bk):
        """Nested lists parse as full matrices; initial_state and scale params are carried."""
        from shinro.estimators.unscented_kf import UnscentedKF

        ukf = UnscentedKF.from_config(
            {
                "process_noise": [[0.01, 0.0], [0.0, 0.02]],
                "measurement_noise": [0.1],
                "dt": 0.1,
                "initial_state": [1.0, 2.0],
                "alpha": 0.5,
                "beta": 2.0,
                "kappa": 1.0,
            },
            backend=bk,
        )
        assert np.allclose(_to_np(ukf.Q, bk), [[0.01, 0.0], [0.0, 0.02]])
        assert np.allclose(_to_np(ukf.x_hat, bk), [1.0, 2.0])
        assert (ukf.alpha, ukf.beta, ukf.kappa) == (0.5, 2.0, 1.0)

    def test_from_config_inject_and_estimate(self, bk):
        """A from_config filter works once the callables are injected."""
        from shinro.estimators.unscented_kf import UnscentedKF

        ukf = UnscentedKF.from_config(
            {"process_noise": [0.01], "measurement_noise": [0.1], "dt": 0.02}, backend=bk
        )
        ukf.dynamics_fn = lambda x, u: bk.zeros(1)
        ukf.measurement_fn = lambda x: bk.array([x[0]])
        x = ukf.estimate(bk.array([1.0]), bk.zeros(1))
        assert _to_np(x, bk).shape == (1,)

    def test_from_config_requires_dt(self, bk):
        """Standalone from_config rejects a missing dt."""
        from shinro.estimators.unscented_kf import UnscentedKF

        with pytest.raises(ValueError, match="dt is required"):
            UnscentedKF.from_config(
                {"process_noise": [0.01], "measurement_noise": [0.1]}, backend=bk
            )

    def test_from_config_rejects_unknown_key(self, bk):
        """Strict config parsing rejects an unknown key."""
        from shinro.estimators.unscented_kf import UnscentedKF

        with pytest.raises(ValueError, match="unknown key"):
            UnscentedKF.from_config(
                {"process_noise": [0.01], "measurement_noise": [0.1], "dt": 0.1, "nope": 1},
                backend=bk,
            )

    def test_measurement_matrix_gives_exact_measurement(self, bk):
        """A linear measurement_matrix supplies h(x) = C x with no callable."""
        from shinro.estimators.unscented_kf import UnscentedKF

        C = bk.array([[1.0, 0.0], [0.0, 1.0]])
        ukf = UnscentedKF(
            lambda x, u: bk.zeros(2),
            None,
            dt=0.01,
            alpha=1.0,
            beta=2.0,
            kappa=0.0,
            Q=0.01 * bk.eye(2),
            R=0.1 * bk.eye(2),
            backend=bk,
            measurement_matrix=C,
        )
        x = bk.array([1.5, -2.0])
        assert np.allclose(_to_np(ukf._measurement(x), bk), _to_np(C @ x, bk))
        out = ukf.estimate(bk.array([1.5, -2.0]), bk.zeros(2))
        assert _to_np(out, bk).shape == (2,)

    def test_measurement_matrix_shape_mismatch_is_loud(self, bk):
        """A C that does not match R's n_y and Q's n_x is rejected at construction."""
        from shinro.estimators.unscented_kf import UnscentedKF

        with pytest.raises(ValueError, match="measurement_matrix shape"):
            UnscentedKF(
                lambda x, u: bk.zeros(2),
                None,
                dt=0.01,
                alpha=1.0,
                beta=2.0,
                kappa=0.0,
                Q=0.01 * bk.eye(2),
                R=0.1 * bk.eye(2),
                backend=bk,
                measurement_matrix=bk.array([[1.0, 0.0]]),  # (1, 2) but R is (2, 2)
            )

    def test_from_config_parses_measurement_matrix(self, bk):
        """from_config carries measurement_matrix into the filter; a C-only filter estimates."""
        from shinro.estimators.unscented_kf import UnscentedKF

        ukf = UnscentedKF.from_config(
            {
                "process_noise": [0.01, 0.01],
                "measurement_noise": [0.1, 0.1],
                "dt": 0.02,
                "alpha": 1.0,
                "measurement_matrix": [[1.0, 0.0], [0.0, 1.0]],
            },
            backend=bk,
        )
        assert ukf.C is not None
        assert _to_np(ukf.C, bk).shape == (2, 2)
        assert ukf.measurement_fn is None
        ukf.dynamics_fn = lambda x, u: bk.zeros(2)
        assert _to_np(ukf.estimate(bk.array([1.0, 2.0]), bk.zeros(2)), bk).shape == (2,)

    def test_bk_aware_callable_receives_the_backend(self, bk):
        """A callable taking ``bk`` is given the filter's backend (the tracing route)."""
        from shinro.estimators.unscented_kf import UnscentedKF

        seen = []

        def dynamics(x, u, bk=None):
            seen.append(bk)
            assert bk is not None
            return bk.zeros_like(x)

        ukf = UnscentedKF(
            dynamics, None, dt=0.01, alpha=1.0, beta=2.0, kappa=0.0,
            Q=0.01 * bk.eye(2), R=0.1 * bk.eye(2), backend=bk, measurement_matrix=bk.eye(2),
        )
        ukf.estimate(bk.array([1.0, 2.0]), bk.zeros(2))
        assert seen, "bk-aware dynamics was never called"
        assert all(s is bk for s in seen), "the filter did not forward its own backend"

    def test_attach_plant_injects_dynamics(self, bk):
        """attach_plant fills an unset dynamics_fn with the plant's model; explicit wins."""
        import tomllib as _tomllib

        from shinro.estimators.unscented_kf import UnscentedKF
        from shinro.factories.registry import _PLANT_REGISTRY
        from shinro.utils.array_backend import NumpyBackend
        from shinro.utils.config_resolver import resolve_config_path

        with open(resolve_config_path("samples/plants/cartpole.toml"), "rb") as f:
            plant = _PLANT_REGISTRY["CartPole"].from_config(_tomllib.load(f), backend=NumpyBackend())

        ukf = UnscentedKF.from_config(
            {
                "process_noise": [0.01] * 4,
                "measurement_noise": [0.1] * 4,
                "dt": 0.01,
                "alpha": 1.0,
                "measurement_matrix": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
            },
            backend=bk,
        )
        assert ukf.dynamics_fn is None
        ukf.attach_plant(plant)
        assert ukf.dynamics_fn is not None
        assert _to_np(ukf.estimate(bk.array([0.1, 0.0, 0.2, 0.0]), bk.zeros(1)), bk).shape == (4,)

        explicit = lambda x, u: bk.zeros(4)  # noqa: E731
        ukf.dynamics_fn = explicit
        ukf.attach_plant(plant)
        assert ukf.dynamics_fn is explicit, "attach_plant overwrote an explicit dynamics_fn"

    def test_attach_plant_without_dynamics_is_loud(self, bk):
        """A plant exposing no dynamics and a filter with none is a loud error."""
        from shinro.estimators.unscented_kf import UnscentedKF

        ukf = UnscentedKF(
            None, None, dt=0.01, alpha=1.0, beta=2.0, kappa=0.0,
            Q=bk.eye(1), R=bk.eye(1), backend=bk,
        )
        with pytest.raises(ValueError, match="no dynamics"):
            ukf.attach_plant(object())

    def test_auto_batching_detects_bk_aware_models(self, bk):
        """Auto mode batches bk-aware (plant/compiled) models, loops plain lambdas."""
        from shinro.estimators.unscented_kf import UnscentedKF

        plain = lambda x, u: bk.zeros(2)  # noqa: E731

        def aware(x, u, bk=None):
            assert bk is not None
            return bk.zeros_like(x)

        ukf = UnscentedKF(
            plain, None, dt=0.01, alpha=1.0, beta=2.0, kappa=0.0,
            Q=bk.eye(2), R=bk.eye(2), backend=bk, measurement_matrix=bk.eye(2),
        )
        assert ukf._batchable(plain) is False
        assert ukf._batchable(aware) is True
        ukf.batched = False
        assert ukf._batchable(aware) is False
        ukf.batched = True
        assert ukf._batchable(plain) is True

    def test_batched_and_per_point_paths_agree(self, bk):
        """A batch-capable model gives identical results batched or row-by-row."""
        import tomllib as _tomllib

        from shinro.estimators.unscented_kf import UnscentedKF
        from shinro.factories.registry import _PLANT_REGISTRY
        from shinro.utils.array_backend import NumpyBackend
        from shinro.utils.config_resolver import resolve_config_path

        with open(resolve_config_path("samples/plants/cartpole.toml"), "rb") as f:
            plant = _PLANT_REGISTRY["CartPole"].from_config(_tomllib.load(f), backend=NumpyBackend())

        cfg = {
            "process_noise": [0.01] * 4,
            "measurement_noise": [0.1] * 4,
            "dt": 0.01,
            "alpha": 1.0,
            "measurement_matrix": [[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]],
        }
        ukf_batched = UnscentedKF.from_config(cfg, backend=bk)
        ukf_batched.attach_plant(plant)
        ukf_looped = UnscentedKF.from_config(cfg, backend=bk)
        ukf_looped.attach_plant(plant)
        ukf_looped.batched = False

        x_batched = x_looped = bk.zeros(4)
        for _ in range(5):
            y, u = bk.array([0.1, 0.0, 0.2, 0.0]), bk.array([0.3])
            x_batched = ukf_batched.estimate(y, u)
            x_looped = ukf_looped.estimate(y, u)
        assert np.allclose(_to_np(x_batched, bk), _to_np(x_looped, bk), atol=1e-12)
        assert np.allclose(_to_np(ukf_batched.P, bk), _to_np(ukf_looped.P, bk), atol=1e-12)

    def test_registered_and_exported(self):
        """The estimator is registered and exported from the package."""
        from shinro.estimators import UnscentedKF as Exported
        from shinro.estimators.unscented_kf import UnscentedKF
        from shinro.factories.registry import _ESTIMATOR_REGISTRY

        assert Exported is UnscentedKF
        assert _ESTIMATOR_REGISTRY["UnscentedKF"] is UnscentedKF
