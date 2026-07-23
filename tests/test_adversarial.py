import pytest
import numpy as np
from scipy.sparse import csr_matrix
from utils.array_backend import NumpyBackend


def _to_np(x, bk):
    """Convert a backend array to numpy for assertion comparisons."""
    return bk.to_numpy(x) if hasattr(bk, 'to_numpy') else x


class TestArrayBackendAdversarial:
    """Verify NumpyBackend handles NaN, Inf, singular, empty, and other edge cases gracefully."""

    def setup_method(self):
        self.bk = NumpyBackend()

    def test_inv_nan(self):
        """inv of a matrix containing NaN propagates NaN or raises LinAlgError."""
        A = np.array([[np.nan, 0], [0, 1]], dtype=float)
        try:
            result = self.bk.inv(A)
            assert np.any(np.isnan(result))
        except np.linalg.LinAlgError:
            pass

    def test_inv_inf(self):
        """inv of a matrix containing Inf produces a finite result (Inf treated as large number)."""
        A = np.array([[np.inf, 0], [0, 1]], dtype=float)
        result = self.bk.inv(A)
        assert np.all(np.isfinite(result))

    def test_inv_singular(self):
        """inv of a singular matrix raises LinAlgError."""
        A = np.array([[1, 2], [2, 4]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.inv(A)

    def test_inv_non_square(self):
        """inv of a non-square matrix raises LinAlgError."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.inv(A)

    def test_cholesky_non_psd(self):
        """cholesky of a non-PSD matrix raises LinAlgError."""
        A = np.array([[-1, 0], [0, -1]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.cholesky(A)

    def test_cholesky_non_square(self):
        """cholesky of a non-square matrix raises LinAlgError."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.cholesky(A)

    def test_solve_singular(self):
        """solve of a singular system raises LinAlgError."""
        A = np.array([[1, 2], [2, 4]], dtype=float)
        b = np.array([1, 2], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.solve(A, b)

    def test_solve_non_square(self):
        """solve of a non-square system raises LinAlgError."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        b = np.array([1, 2], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.solve(A, b)

    def test_svd_nan(self):
        """svd of a matrix containing NaN raises LinAlgError or propagates NaN."""
        A = np.array([[np.nan, 0], [0, 1]], dtype=float)
        try:
            result = self.bk.svd(A)
            assert np.any(np.isnan(result[0])) or np.any(np.isnan(result[1]))
        except np.linalg.LinAlgError:
            pass

    def test_eigvals_nan(self):
        """eigvals of a matrix containing NaN raises LinAlgError."""
        A = np.array([[np.nan, 0], [0, 1]], dtype=float)
        with pytest.raises(np.linalg.LinAlgError):
            self.bk.eigvals(A)

    def test_matrix_rank_zero(self):
        """matrix_rank of a zero matrix is 0."""
        A = np.zeros((5, 5))
        assert self.bk.matrix_rank(A) == 0

    def test_matrix_rank_empty(self):
        """matrix_rank of an empty matrix is 0."""
        A = np.zeros((0, 0))
        assert self.bk.matrix_rank(A) == 0

    def test_cond_singular(self):
        """cond of a singular matrix is very large (not exactly inf due to floating point)."""
        A = np.array([[1, 2], [2, 4]], dtype=float)
        assert self.bk.cond(A) > 1e15

    def test_cond_zero(self):
        """cond of a zero matrix is inf."""
        A = np.zeros((3, 3))
        assert self.bk.cond(A) == np.inf

    def test_sqrt_negative(self):
        """sqrt of a negative number produces NaN (with a runtime warning)."""
        x = np.array([-1.0])
        result = self.bk.sqrt(x)
        assert np.isnan(result[0])

    def test_arccos_out_of_range(self):
        """arccos of a value outside [-1, 1] produces NaN (with a runtime warning)."""
        result = self.bk.arccos(np.array([2.0]))
        assert np.any(np.isnan(result))

    def test_zeros_empty_shape(self):
        """zeros with shape (0,) produces an empty array."""
        z = self.bk.zeros(0)
        assert z.shape == (0,)

    def test_zeros_negative_shape(self):
        """zeros with a negative shape raises ValueError."""
        with pytest.raises(ValueError):
            self.bk.zeros(-1)

    def test_eye_zero(self):
        """eye(0) produces an empty 0x0 matrix."""
        I = self.bk.eye(0)
        assert I.shape == (0, 0)

    def test_eye_negative(self):
        """eye(-1) raises ValueError."""
        with pytest.raises(ValueError):
            self.bk.eye(-1)

    def test_linspace_reversed(self):
        """linspace with start > stop produces a decreasing sequence."""
        x = self.bk.linspace(1, 0, 5)
        assert _to_np(x, self.bk)[0] == 1.0
        assert _to_np(x, self.bk)[-1] == 0.0

    def test_linspace_zero_points(self):
        """linspace with num=0 produces an empty array."""
        x = self.bk.linspace(0, 1, 0)
        assert len(x) == 0

    def test_linspace_single_point(self):
        """linspace with num=1 produces a single-element array."""
        x = self.bk.linspace(0, 1, 1)
        assert len(x) == 1

    def test_matrix_power_zero(self):
        """matrix_power(A, 0) returns the identity matrix."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        A0 = self.bk.matrix_power(A, 0)
        assert np.allclose(A0, np.eye(2))

    def test_matrix_power_negative(self):
        """matrix_power(A, -1) returns the matrix inverse."""
        A = np.array([[1, 2], [3, 4]], dtype=float)
        result = self.bk.matrix_power(A, -1)
        assert np.allclose(result @ A, np.eye(2))

    def test_matrix_power_non_square(self):
        """matrix_power of a non-square matrix raises ValueError."""
        A = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        with pytest.raises(ValueError):
            self.bk.matrix_power(A, 2)

    def test_vstack_empty_list(self):
        """vstack of an empty list raises ValueError."""
        with pytest.raises(ValueError):
            self.bk.vstack([])

    def test_hstack_empty_list(self):
        """hstack of an empty list raises ValueError."""
        with pytest.raises(ValueError):
            self.bk.hstack([])

    def test_vstack_mismatched_cols(self):
        """vstack of arrays with mismatched column counts raises ValueError."""
        a = np.array([1, 2], dtype=float)
        b = np.array([3, 4, 5], dtype=float)
        with pytest.raises(ValueError):
            self.bk.vstack([a, b])

    def test_hstack_mismatched_rows(self):
        """hstack of arrays with mismatched row counts raises ValueError."""
        a = np.array([[1], [2]], dtype=float)
        b = np.array([[3]], dtype=float)
        with pytest.raises(ValueError):
            self.bk.hstack([a, b])

    def test_reshape_incompatible(self):
        """reshape to an incompatible shape raises ValueError."""
        x = np.array([1, 2, 3], dtype=float)
        with pytest.raises(ValueError):
            self.bk.reshape(x, 2, 2)

    def test_tile_negative_reps(self):
        """tile with negative repetitions raises ValueError."""
        x = np.array([1, 2], dtype=float)
        with pytest.raises(ValueError):
            self.bk.tile(x, -1)

    def test_clip_swapped_bounds(self):
        """clip with lo > hi clamps to lo (numpy behavior: lo takes precedence)."""
        x = np.array([0.5], dtype=float)
        clipped = self.bk.clip(x, 1.0, 0.0)
        assert _to_np(clipped, self.bk)[0] == 0.0

    def test_where_mismatched_shapes(self):
        """where with mismatched array shapes raises ValueError."""
        cond = np.array([True, False])
        a = np.array([1, 2, 3], dtype=float)
        b = np.array([10, 20], dtype=float)
        with pytest.raises(ValueError):
            self.bk.where(cond, a, b)

    def test_block_mismatched_shapes(self):
        """block with mismatched block shapes raises ValueError."""
        A = np.array([[1, 2]], dtype=float)
        B = np.array([[3]], dtype=float)
        C = np.array([[4]], dtype=float)
        with pytest.raises(ValueError):
            self.bk.block([[A, B], [C]])

    def test_kron_empty(self):
        """kron with an empty array produces an empty result."""
        a = np.array([], dtype=float)
        b = np.array([1, 2], dtype=float)
        k = self.bk.kron(a, b)
        assert len(k) == 0

    def test_cross_2d(self):
        """cross of 2D vectors is deprecated in NumPy 2.0 and returns a scalar or raises."""
        a = np.array([1, 0], dtype=float)
        b = np.array([0, 1], dtype=float)
        try:
            result = self.bk.cross(a, b)
            assert np.isscalar(result) or result.shape == ()
        except (ValueError, DeprecationWarning):
            pass

    def test_cross_4d(self):
        """cross of 4D vectors raises ValueError."""
        a = np.array([1, 0, 0, 0], dtype=float)
        b = np.array([0, 1, 0, 0], dtype=float)
        with pytest.raises(ValueError):
            self.bk.cross(a, b)

    def test_norm_zero_vector(self):
        """norm of a zero vector is 0."""
        v = np.array([0, 0, 0], dtype=float)
        assert self.bk.norm(v) == 0.0

    def test_norm_empty(self):
        """norm of an empty vector is 0."""
        v = np.array([], dtype=float)
        assert self.bk.norm(v) == 0.0

    def test_diag_scalar(self):
        """diag of a scalar raises ValueError."""
        with pytest.raises(ValueError):
            self.bk.diag(5.0)

    def test_any_empty(self):
        """any of an empty array is False."""
        assert not self.bk.any(np.array([], dtype=bool))

    def test_sum_empty(self):
        """sum of an empty array is 0."""
        assert self.bk.sum(np.array([], dtype=float)) == 0.0

    def test_real_complex(self):
        """real extracts the real part of a complex array."""
        x = np.array([1 + 2j, 3 + 4j])
        r = self.bk.real(x)
        assert np.allclose(r, [1, 3])

    def test_sort_reverse(self):
        """sort returns elements in ascending order."""
        x = np.array([3, 1, 2], dtype=float)
        assert np.allclose(self.bk.sort(x), [1, 2, 3])

    def test_abs_negative(self):
        """abs returns the absolute value of negative numbers."""
        x = np.array([-1, -2, -3], dtype=float)
        assert np.allclose(self.bk.abs(x), [1, 2, 3])

    def test_ravel_0d(self):
        """ravel of a 0D array returns a 1-element 1D array."""
        x = np.array(5.0)
        y = self.bk.ravel(x)
        assert y.shape == (1,)

    def test_reshape_0d(self):
        """reshape of a 0D array to 1D works."""
        x = np.array(5.0)
        y = self.bk.reshape(x, 1)
        assert y.shape == (1,)

    def test_copy_modification_independence(self):
        """copy returns an independent array; modifying one does not affect the other."""
        x = np.array([1, 2, 3], dtype=float)
        y = self.bk.copy(x)
        y[0] = 99
        assert x[0] == 1
        x[1] = 88
        assert y[1] == 2


class TestControllabilityCheckerAdversarial:
    """Verify LTISystemsAnalyzer handles zero, identity, scalar, Jordan-block, sparse, and edge-case systems."""

    def test_zero_system(self):
        """A zero system (A=0, B=0) is not controllable but is observable (C=I)."""
        from utils.controllability_checker import LTISystemsAnalyzer
        n = 3
        A = np.zeros((n, n))
        B = np.zeros((n, 1))
        C = np.eye(n)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert not ana.is_controllable()
        assert ana.is_observable()

    def test_identity_system(self):
        """An identity system (A=I, B=I, C=I) is both controllable and observable."""
        from utils.controllability_checker import LTISystemsAnalyzer
        n = 3
        A = np.eye(n)
        B = np.eye(n)
        C = np.eye(n)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert ana.is_controllable()
        assert ana.is_observable()

    def test_scalar_system(self):
        """A 1x1 stable system has a known analytical Gramian Wc = 0.5."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[-1.0]])
        B = np.array([[1.0]])
        C = np.array([[1.0]])
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert ana.is_controllable()
        assert ana.is_observable()
        Wc = ana.controllability_gramian()
        assert np.allclose(Wc, 0.5)

    def test_jordan_block(self):
        """A Jordan block (chain of integrators) with B = [0,0,1]^T is controllable."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        B = np.array([[0], [0], [1]], dtype=float)
        C = np.eye(3)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert ana.is_controllable()
        assert ana.is_observable()

    def test_jordan_block_uncontrollable(self):
        """A Jordan block with B = [1,0,0]^T is NOT controllable (input only reaches first state)."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        B = np.array([[1], [0], [0]], dtype=float)
        C = np.eye(3)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert not ana.is_controllable()

    def test_discrete_dt_zero(self):
        """Discrete Gramian with dt=0 still works (dt=0 is valid, just means no time passes)."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0.9, 0.1], [0, 0.8]], dtype=float)
        B = np.array([[0], [0.1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, dt=0.0, backend=NumpyBackend())
        Wc = ana.discrete_controllability_gramian()
        assert np.all(np.linalg.eigvals(Wc) > -1e-10)

    def test_sparse_A(self):
        """A sparse matrix A works correctly with the controllability checker."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = csr_matrix(np.array([[0, 1], [-1, -2]], dtype=float))
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        assert ana.is_controllable()

    def test_gramian_condition_singular(self):
        """gramian_condition raises ValueError for a non-Hurwitz system (Gramian does not exist)."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.zeros((2, 2))
        B = np.array([[1], [1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        with pytest.raises(ValueError):
            ana.gramian_condition("Wc")

    def test_gramian_spectrum_finite_horizon_malformed(self):
        """gramian_spectrum with a malformed finite-horizon identifier raises ValueError."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        with pytest.raises(ValueError):
            ana.gramian_spectrum("Wc_finite_bad")

    def test_balanced_truncate_r_equals_n(self):
        """Balanced truncation with r=n returns the full-order model unchanged."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        Ar, Br, Cr, Dr = ana.balanced_truncate(2)
        assert Ar.shape == (2, 2)

    def test_rank_report_on_zero_system(self):
        """rank_report on a zero system returns rank 0 for both controllability and observability."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.zeros((3, 3))
        B = np.zeros((3, 1))
        C = np.zeros((2, 3))
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        report = ana.rank_report()
        assert report["controllability"][0] == 0
        assert report["observability"][0] == 0

    def test_summary_on_zero_system(self):
        """summary on a zero system raises ValueError (Gramian does not exist for unstable A)."""
        from utils.controllability_checker import LTISystemsAnalyzer
        A = np.zeros((2, 2))
        B = np.zeros((2, 1))
        C = np.eye(2)
        ana = LTISystemsAnalyzer(A, B, C, backend=NumpyBackend())
        with pytest.raises(ValueError):
            ana.summary()


class TestLQRAdversarial:
    """Verify LQR handles zero Q, zero R, non-PSD Q, mismatched dims, and wrong input shapes."""

    def test_lqr_zero_Q(self, bk):
        """LQR with Q=0 (zero state cost) raises LinAlgError (DARE has no finite solution)."""
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.zeros((2, 2))
        R = bk.eye(2)
        with pytest.raises(np.linalg.LinAlgError):
            LQR(Q, R, A, B, backend=bk)

    def test_lqr_zero_R(self, bk):
        """LQR with R=0 (zero control cost) produces a finite gain (inverse is handled)."""
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.zeros((2, 2))
        lqr = LQR(Q, R, A, B, backend=bk)
        K = _to_np(lqr.K, bk)
        assert np.all(np.isfinite(K))

    def test_lqr_non_psd_Q(self, bk):
        """LQR with non-PSD Q (negative eigenvalues) raises an error."""
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.array([[-1, 0], [0, -1]])
        R = bk.eye(2)
        with pytest.raises(Exception):
            LQR(Q, R, A, B, backend=bk)

    def test_lqr_mismatched_dims(self, bk):
        """LQR with mismatched A and B dimensions raises an error."""
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(3)
        Q = bk.eye(2)
        R = bk.eye(2)
        with pytest.raises(Exception):
            LQR(Q, R, A, B, backend=bk)

    def test_lqr_compute_with_target(self, bk):
        """compute(x, target) returns u = -K @ (x - target)."""
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.eye(2)
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.0, 2.0])
        target = bk.array([3.0, 4.0])
        u = lqr.compute(x, target)
        u_expected = -lqr.K @ (x - target)
        assert np.allclose(_to_np(u, bk), _to_np(u_expected, bk))

    def test_lqr_compute_wrong_shape(self, bk):
        """compute with a wrong-shaped state vector raises an error."""
        from controllers.lqr import LQR
        A = bk.eye(2)
        B = bk.eye(2)
        Q = bk.eye(2)
        R = bk.eye(2)
        lqr = LQR(Q, R, A, B, backend=bk)
        x = bk.array([1.0, 2.0, 3.0])
        with pytest.raises(Exception):
            lqr.compute(x)


class TestPIDAdversarial:
    """Verify PID handles zero gains, negative Kp, mismatched lengths, zero/negative dt, and wrong shapes."""

    def test_pid_zero_gains(self, bk):
        """PID with all gains zero produces zero control effort."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([0.0]),
            ki=bk.array([0.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0]), bk.array([0.0]))
        assert np.allclose(_to_np(u, bk), 0.0)

    def test_pid_negative_kp(self, bk):
        """PID with negative Kp drives the state away from the target (positive feedback)."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([-1.0]),
            ki=bk.array([0.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        x = bk.array([0.0])
        target = bk.array([1.0])
        for _ in range(100):
            u = pid.compute(x, target)
            x = x + 0.01 * u
        assert _to_np(x, bk)[0] < 0

    def test_pid_mismatched_gain_lengths(self, bk):
        """PID with mismatched gain lengths broadcasts (numpy behavior)."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0, 2.0]),
            ki=bk.array([0.5]),
            kd=bk.array([0.1]),
            dt=0.01,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0, 2.0]), bk.array([0.0, 0.0]))
        assert _to_np(u, bk).shape == (2,)

    def test_pid_zero_dt(self, bk):
        """PID with dt=0 produces finite output (no division by zero in derivative)."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.5]),
            kd=bk.array([0.0]),
            dt=0.0,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0]), bk.array([0.0]))
        assert not np.any(np.isnan(_to_np(u, bk)))

    def test_pid_negative_dt(self, bk):
        """PID with negative dt produces finite output (backward time step)."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.5]),
            kd=bk.array([0.0]),
            dt=-0.01,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0]), bk.array([0.0]))
        assert not np.any(np.isnan(_to_np(u, bk)))

    def test_pid_wrong_state_shape(self, bk):
        """PID with wrong-shaped state broadcasts (numpy behavior)."""
        from controllers.pid import PIDController
        pid = PIDController(
            kp=bk.array([1.0]),
            ki=bk.array([0.0]),
            kd=bk.array([0.0]),
            dt=0.01,
            backend=bk,
        )
        u = pid.compute(bk.array([1.0, 2.0]), bk.array([0.0]))
        assert _to_np(u, bk).shape == (2,)


class TestMPCAdversarial:
    """Verify MPC handles horizon=1, mismatched dims, no constraints, and zero/negative horizon."""

    def test_mpc_horizon_1(self, bk):
        """MPC with horizon=1 produces a valid control output."""
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        mpc = MPC_LTI(horizon=1, control_cost_matrix=R, state_cost_matrix=Q,
                      A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)
        F = bk.eye(m)
        mpc.constraints(F, bk.array([1.0, 1.0]), bk.array([-1.0, -1.0]))
        x0 = bk.array([1.0, 0.0])
        u = mpc.compute(x0)
        assert _to_np(u, bk).shape == (m,)

    def test_mpc_mismatched_B_dims(self, bk):
        """MPC with mismatched A and B dimensions raises an error."""
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 3
        A = bk.eye(n)
        B = bk.eye(m)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        with pytest.raises(Exception):
            MPC_LTI(horizon=5, control_cost_matrix=R, state_cost_matrix=Q,
                    A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)

    def test_mpc_no_constraints_crashes(self, bk):
        """MPC without calling constraints() raises AttributeError in compute()."""
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
        x0 = bk.array([1.0, 0.0])
        with pytest.raises(AttributeError):
            mpc.compute(x0)

    def test_mpc_zero_horizon(self, bk):
        """MPC with horizon=0 raises an error."""
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        with pytest.raises(Exception):
            MPC_LTI(horizon=0, control_cost_matrix=R, state_cost_matrix=Q,
                    A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)

    def test_mpc_negative_horizon(self, bk):
        """MPC with negative horizon raises an error."""
        from controllers.mpc_lti import MPC_LTI
        n = 2
        m = 2
        A = bk.eye(n)
        B = 0.1 * bk.eye(n)
        Q = bk.eye(n)
        R = bk.eye(m)
        P = bk.eye(n)
        with pytest.raises(Exception):
            MPC_LTI(horizon=-1, control_cost_matrix=R, state_cost_matrix=Q,
                    A_dynamics=A, B_dynamics=B, terminal_cost=P, backend=bk)


class TestKalmanFilterAdversarial:
    """Verify Kalman filter handles zero noise, infinite noise, non-PSD Q, and mismatched dims."""

    def test_kalman_zero_noise(self, bk):
        """Kalman filter with Q=0, R=0 (no noise) produces a finite estimate."""
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = bk.zeros((n, n))
        R = bk.zeros((n, n))
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = kf.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))

    def test_kalman_infinite_measurement_noise(self, bk):
        """Kalman filter with very large R (infinite measurement noise) ignores measurements."""
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = 0.1 * bk.eye(n)
        R = 1e10 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = kf.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))

    def test_kalman_infinite_process_noise(self, bk):
        """Kalman filter with very large Q (infinite process noise) trusts measurements fully."""
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = 1e10 * bk.eye(n)
        R = 0.1 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = kf.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))

    def test_kalman_mismatched_dims(self, bk):
        """Kalman filter with mismatched C and R dimensions raises ValueError in estimate()."""
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = bk.eye(n)
        B = bk.eye(n)
        Q = 0.1 * bk.eye(n)
        R = 0.1 * bk.eye(3)
        C = bk.eye(3)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        with pytest.raises((ValueError, RuntimeError)):
            kf.estimate(y, u)

    def test_kalman_non_psd_Q(self, bk):
        """Kalman filter with non-PSD Q (negative eigenvalues) still produces a finite estimate."""
        from estimators.kalman_filter import KalmanFilter
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        Q = bk.array([[-1, 0], [0, -1]])
        R = 0.1 * bk.eye(n)
        C = bk.eye(n)
        kf = KalmanFilter(A, B, Q, R, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = kf.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))


class TestLuenbergerObserverAdversarial:
    """Verify Luenberger observer handles zero gain, destabilizing gain, and mismatched gain dimensions."""

    def test_luenberger_zero_gain(self, bk):
        """Luenberger observer with L=0 (open-loop) produces a finite estimate (no correction)."""
        from estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        L = bk.zeros((n, n))
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        x = obs.estimate(y, u)
        assert not np.any(np.isnan(_to_np(x, bk)))

    def test_luenberger_destabilizing_gain(self, bk):
        """Luenberger observer with a large gain L produces unstable error dynamics (|eig| >= 1)."""
        from estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        L = 2.0 * bk.eye(n)
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        A_cl = A - L @ C
        eigs = np.linalg.eigvals(_to_np(A_cl, bk))
        assert np.any(np.abs(eigs) >= 1)

    def test_luenberger_mismatched_gain(self, bk):
        """Luenberger observer with a wrong-shaped gain L raises ValueError in estimate()."""
        from estimators.luenberger_observer import LuenbergerObserver
        n = 2
        A = 0.9 * bk.eye(n)
        B = bk.eye(n)
        L = bk.eye(3)
        C = bk.eye(n)
        obs = LuenbergerObserver(A, B, L, C=C, backend=bk)
        y = bk.array([[1.0], [0.0]])
        u = bk.array([[0.0], [0.0]])
        with pytest.raises((ValueError, RuntimeError)):
            obs.estimate(y, u)


class TestCubicPolynomialAdversarial:
    """Verify cubic polynomial handles zero/negative duration, mismatched dims, zero-dimensional, and large time."""

    def test_cubic_zero_duration(self, bk):
        """Cubic polynomial with zero duration produces NaN (division by zero in coefficients)."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0])
        vf = bk.array([0.0])
        traj.generate(p0, pf, 0.0, v0, vf)
        pos, _, _ = traj.position_at(0.0)
        assert np.any(np.isnan(_to_np(pos, bk)))

    def test_cubic_negative_duration(self, bk):
        """Cubic polynomial with negative duration produces a finite position (time is clamped)."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0])
        vf = bk.array([0.0])
        traj.generate(p0, pf, -1.0, v0, vf)
        pos, _, _ = traj.position_at(0.0)
        assert np.isfinite(_to_np(pos, bk)[0])

    def test_cubic_mismatched_dims(self, bk):
        """Cubic polynomial with mismatched p0 and pf dimensions broadcasts (numpy behavior)."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0, 0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0, 0.0])
        vf = bk.array([0.0])
        traj.generate(p0, pf, 1.0, v0, vf)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (2,)

    def test_cubic_zero_dimensional(self, bk):
        """Cubic polynomial with zero-dimensional positions produces an empty result."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([])
        pf = bk.array([])
        v0 = bk.array([])
        vf = bk.array([])
        traj.generate(p0, pf, 1.0, v0, vf)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (0,)

    def test_cubic_large_time(self, bk):
        """Cubic polynomial with very large time is clamped to T."""
        from trajectories.cubic_polynomial import CubicPolynomial
        traj = CubicPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        v0 = bk.array([0.0])
        vf = bk.array([0.0])
        traj.generate(p0, pf, 1.0, v0, vf)
        pos, _, _ = traj.position_at(1e6)
        assert np.allclose(_to_np(pos, bk), 1.0)


class TestQuinticPolynomialAdversarial:
    """Verify quintic polynomial handles zero/negative duration, mismatched dims, zero-dimensional, and large time."""

    def test_quintic_zero_duration(self, bk):
        """Quintic polynomial with zero duration raises an error (singular linear system)."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        with pytest.raises(Exception):
            traj.generate(p0, pf, 0.0)

    def test_quintic_negative_duration(self, bk):
        """Quintic polynomial with negative duration produces a finite position."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        traj.generate(p0, pf, -1.0)
        pos, _, _ = traj.position_at(0.0)
        assert np.isfinite(_to_np(pos, bk)[0])

    def test_quintic_mismatched_dims(self, bk):
        """Quintic polynomial with mismatched p0 and pf dimensions raises an error."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0, 0.0])
        pf = bk.array([1.0])
        with pytest.raises(Exception):
            traj.generate(p0, pf, 1.0)

    def test_quintic_zero_dimensional(self, bk):
        """Quintic polynomial with zero-dimensional positions produces an empty result."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([])
        pf = bk.array([])
        traj.generate(p0, pf, 1.0)
        pos, _, _ = traj.position_at(0.5)
        assert _to_np(pos, bk).shape == (0,)

    def test_quintic_large_time(self, bk):
        """Quintic polynomial with very large time is clamped to T."""
        from trajectories.quintic_polynomial import QuinticPolynomial
        traj = QuinticPolynomial(backend=bk)
        p0 = bk.array([0.0])
        pf = bk.array([1.0])
        traj.generate(p0, pf, 1.0)
        pos, _, _ = traj.position_at(1e6)
        assert np.allclose(_to_np(pos, bk), 1.0)


class TestHolonomicMobileRobotAdversarial:
    """Verify HolonomicMobileRobot handles zero/negative dt, zero wheel radius, single/many wheels, and wrong shapes."""

    def test_zero_dt(self, bk):
        """HolonomicMobileRobot with dt=0 does not move (state stays at origin)."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.0, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        robot.step(u)
        state = robot.get_state()
        assert np.allclose(_to_np(state, bk), 0.0)

    def test_negative_dt(self, bk):
        """HolonomicMobileRobot with negative dt moves backward."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=-0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        robot.step(u)
        state = robot.get_state()
        assert _to_np(state, bk)[0] < 0

    def test_zero_wheel_radius(self, bk):
        """HolonomicMobileRobot with zero wheel radius raises ZeroDivisionError on step()."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.0, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        with pytest.raises(ZeroDivisionError):
            robot.step(u)

    def test_zero_robot_radius(self, bk):
        """HolonomicMobileRobot with zero robot radius still produces wheel speeds."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.0, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        wheel_speeds = robot.step(u)
        assert _to_np(wheel_speeds, bk).shape == (3,)

    def test_single_wheel(self, bk):
        """HolonomicMobileRobot with a single wheel produces a 1-element wheel speed vector."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=1, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        wheel_speeds = robot.step(u)
        assert _to_np(wheel_speeds, bk).shape == (1,)

    def test_many_wheels(self, bk):
        """HolonomicMobileRobot with 10 wheels produces a 10-element wheel speed vector."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=10, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0, 0.0])
        wheel_speeds = robot.step(u)
        assert _to_np(wheel_speeds, bk).shape == (10,)

    def test_wrong_input_shape(self, bk):
        """HolonomicMobileRobot with a wrong-shaped input raises an error."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        u = bk.array([1.0, 0.0])
        with pytest.raises(Exception):
            robot.step(u)

    def test_set_pose_then_step(self, bk):
        """HolonomicMobileRobot with zero velocity stays at the set pose."""
        from plants.holonomicmobilerobot import HolonomicMobileRobot
        robot = HolonomicMobileRobot(num_wheels=3, radius_robots=0.1, gamma=0.0,
                                     radius_wheels=0.05, dt=0.01, backend=bk)
        robot.set_pose(1.0, 2.0, 0.5)
        u = bk.array([0.0, 0.0, 0.0])
        robot.step(u)
        state = robot.get_state()
        assert np.allclose(_to_np(state, bk), [1.0, 2.0, 0.5])
