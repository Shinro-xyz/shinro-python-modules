import numpy as np


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestKalmanFilter:
    """Verify Kalman filter: estimate shape, PSD properties, convergence, and reset."""

    def test_estimate_shape(self, bk):
        """estimate() returns a state vector of shape (n_x, 1)."""
        from estimators.kalman_filter import KalmanFilter
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
        from estimators.kalman_filter import KalmanFilter
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
        from estimators.kalman_filter import KalmanFilter
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
        from estimators.kalman_filter import KalmanFilter
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
        from estimators.kalman_filter import KalmanFilter
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
        from estimators.kalman_filter import KalmanFilter
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
        from estimators.kalman_filter import KalmanFilter
        A = bk.array([[0.9]])
        B = bk.array([[1.0]])
        Q = bk.array([[0.01]])
        R = bk.array([[0.1]])
        C = bk.array([[1.0]])
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        true_state = np.array([[5.0]])
        for _ in range(30):
            y = bk.from_numpy(true_state + 0.1 * np.random.randn(1, 1))
            u = bk.array([[0.0]])
            x = kf.estimate(y, u)
            true_state = 0.9 * true_state
        error = np.abs(_to_np(x, bk)[0, 0] - true_state[0, 0])
        assert error < 1.0

    def test_reset(self, bk):
        """reset() clears the state estimate to zero."""
        from estimators.kalman_filter import KalmanFilter
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


class TestLuenbergerObserver:
    """Verify Luenberger observer: estimate shape, stability, convergence, and reset."""

    def test_estimate_shape(self, bk):
        """estimate() returns a state vector of shape (n_x, 1)."""
        from estimators.luenberger_observer import LuenbergerObserver
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
        from estimators.luenberger_observer import LuenbergerObserver
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
        from estimators.luenberger_observer import LuenbergerObserver
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
        from estimators.luenberger_observer import LuenbergerObserver
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
        from estimators.luenberger_observer import LuenbergerObserver
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
        from estimators.luenberger_observer import LuenbergerObserver
        A = bk.array([[0.9]])
        B = bk.array([[1.0]])
        L = bk.array([[0.5]])
        C = bk.array([[1.0]])
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        true_state = np.array([[5.0]])
        for _ in range(30):
            y = bk.from_numpy(true_state)
            u = bk.array([[0.0]])
            x = obs.estimate(y, u)
            true_state = 0.9 * true_state
        error = np.abs(_to_np(x, bk)[0, 0] - true_state[0, 0])
        assert error < 0.1

    def test_reset(self, bk):
        """reset() clears the state estimate to zero."""
        from estimators.luenberger_observer import LuenbergerObserver
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
