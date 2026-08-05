import numpy as np
import pytest
from scipy.linalg import cholesky, solve_continuous_lyapunov

from shinro.utils.array_backend import NumpyBackend
from shinro.utils.controllability_checker import LTISystemsAnalyzer


def _make_ana(A, B, C, dt=None, bk=None):
    """Create an LTISystemsAnalyzer with a NumpyBackend."""
    bk = bk or NumpyBackend()
    return LTISystemsAnalyzer(A, B, C, dt=dt, backend=bk)


class TestControllability:
    """Verify the Kalman rank test for controllability."""

    def test_double_integrator_controllable(self):
        """Double integrator with B = [0, 1]^T is controllable (rank = 2)."""
        A = np.array([[0, 1], [0, 0]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        assert ana.is_controllable()

    def test_parallel_integrators_uncontrollable(self):
        """Two identical integrators driven by the same input: rank = 1 < 2."""
        A = np.zeros((2, 2))
        B = np.array([[1], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        assert not ana.is_controllable()

    def test_damped_oscillator_controllable(self):
        """Damped oscillator with B = [0, 1]^T is controllable."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        assert ana.is_controllable()

    def test_triple_integrator_controllable(self):
        """Chain of three integrators is controllable."""
        A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=float)
        B = np.array([[0], [0], [1]], dtype=float)
        C = np.eye(3)
        ana = _make_ana(A, B, C)
        assert ana.is_controllable()

    def test_mimo_controllable(self):
        """MIMO system with 2 inputs and 3 states is controllable."""
        A = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]], dtype=float)
        B = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        C = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        ana = _make_ana(A, B, C)
        assert ana.is_controllable()


class TestObservability:
    """Verify the Kalman rank test for observability."""

    def test_double_integrator_observable(self):
        """Double integrator with C = I is observable (rank = 2)."""
        A = np.array([[0, 1], [0, 0]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        assert ana.is_observable()

    def test_velocity_only_unobservable(self):
        """Only velocity is measured: observability matrix has rank = 1 < 2."""
        A = np.array([[0, 1], [0, 0]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.array([[0, 1]], dtype=float)
        ana = _make_ana(A, B, C)
        assert not ana.is_observable()

    def test_damped_oscillator_observable(self):
        """Damped oscillator with C = I is observable."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        assert ana.is_observable()


class TestGramianContinuous:
    """Verify continuous-time Gramian properties."""

    def test_infinite_gramian_psd_for_hurwitz(self):
        """Infinite-horizon Gramians are PSD for Hurwitz A."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        Wc = ana.controllability_gramian()
        assert np.all(np.linalg.eigvals(Wc) > -1e-10)
        Wo = ana.observability_gramian()
        assert np.all(np.linalg.eigvals(Wo) > -1e-10)

    def test_infinite_gramian_raises_for_unstable(self):
        """Infinite-horizon Gramian raises ValueError for non-Hurwitz A."""
        A = np.array([[0, 1], [0, 0]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        with pytest.raises(ValueError, match="not Hurwitz"):
            ana.controllability_gramian()

    def test_finite_gramian_works_for_unstable(self):
        """Finite-horizon Gramian works even for unstable A."""
        A = np.array([[0, 1], [0, 0]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        Wc = ana.controllability_gramian_finite(T=5.0)
        assert np.all(np.linalg.eigvals(Wc) > -1e-10)

    def test_finite_gramian_raises_for_nonpositive_T(self):
        """Finite-horizon Gramian raises ValueError for T <= 0."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        with pytest.raises(ValueError):
            ana.controllability_gramian_finite(0)
        with pytest.raises(ValueError):
            ana.controllability_gramian_finite(-1)

    def test_gramian_matches_lyapunov_solution(self):
        """Controllability Gramian matches the direct Lyapunov solution."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        Wc = ana.controllability_gramian()
        Wc_expected = solve_continuous_lyapunov(A, -B @ B.T)
        assert np.allclose(Wc, Wc_expected)


class TestGramianDiscrete:
    """Verify discrete-time Gramian properties."""

    def test_discrete_gramian_psd(self):
        """Discrete Gramians are PSD for asymptotically stable A."""
        A = np.array([[0.9, 0.1], [0, 0.8]], dtype=float)
        B = np.array([[0], [0.1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C, dt=0.1)
        Wc = ana.discrete_controllability_gramian()
        assert np.all(np.linalg.eigvals(Wc) > -1e-10)
        Wo = ana.discrete_observability_gramian()
        assert np.all(np.linalg.eigvals(Wo) > -1e-10)

    def test_discrete_gramian_raises_without_dt(self):
        """Discrete Gramian raises ValueError when dt is None."""
        A = np.array([[0.9, 0.1], [0, 0.8]], dtype=float)
        B = np.array([[0], [0.1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        with pytest.raises(ValueError, match="dt is None"):
            ana.discrete_controllability_gramian()

    def test_discrete_gramian_raises_for_unstable(self):
        """Discrete Gramian raises ValueError for unstable A (|eig| >= 1)."""
        A = np.array([[1.1, 0], [0, 1.2]], dtype=float)
        B = np.array([[0], [0.1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C, dt=0.1)
        with pytest.raises(ValueError, match="not asymptotically stable"):
            ana.discrete_controllability_gramian()


class TestHankelAndBalanced:
    """Verify Hankel singular values and balanced realization."""

    def test_hankel_singular_values_sorted(self):
        """Hankel singular values are sorted in descending order."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        sigma = ana.hankel_singular_values()
        assert sigma[0] >= sigma[1] >= 0

    def test_balanced_realization_diagonal_gramians(self):
        """Balanced Gramians are diagonal (off-diagonal sum < 1e-8)."""
        A = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]], dtype=float)
        B = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        C = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        ana = _make_ana(A, B, C)
        Ab, Bb, Cb = ana.balanced_realization()
        Wc = ana.controllability_gramian()
        Wo = ana.observability_gramian()
        Lc = cholesky(Wc, lower=True)
        Lo = cholesky(Wo, lower=True)
        U, s, Vh = np.linalg.svd(Lo.T @ Lc)
        T = Lc @ Vh.T @ np.diag(1.0 / np.sqrt(s))
        Tinv = np.diag(1.0 / np.sqrt(s)) @ U.T @ Lo.T
        Wc_bal = Tinv @ Wc @ Tinv.T
        Wo_bal = T.T @ Wo @ T
        off_wc = np.sum(np.abs(Wc_bal - np.diag(np.diag(Wc_bal))))
        off_wo = np.sum(np.abs(Wo_bal - np.diag(np.diag(Wo_bal))))
        assert off_wc < 1e-8
        assert off_wo < 1e-8

    def test_balanced_realization_preserves_eigenvalues(self):
        """Balanced transformation preserves the eigenvalues of A."""
        A = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]], dtype=float)
        B = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        C = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        ana = _make_ana(A, B, C)
        Ab, Bb, Cb = ana.balanced_realization()
        assert np.allclose(np.sort(np.linalg.eigvals(Ab)), np.sort(np.linalg.eigvals(A)))

    def test_balanced_truncation_shapes(self):
        """Balanced truncation produces matrices of the correct reduced order."""
        A = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]], dtype=float)
        B = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        C = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        ana = _make_ana(A, B, C)
        Ar, Br, Cr, Dr = ana.balanced_truncate(2)
        assert Ar.shape == (2, 2)
        assert Br.shape == (2, 2)
        assert Cr.shape == (2, 2)
        assert Dr.shape == (2, 2)

    def test_balanced_truncation_error_bound(self):
        """Balanced truncation stores a positive error bound."""
        A = np.array([[0, 1, 0], [0, 0, 1], [-6, -11, -6]], dtype=float)
        B = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
        C = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        ana = _make_ana(A, B, C)
        ana.balanced_truncate(2)
        assert "balanced_trunc_error_bound" in ana._cached_values
        assert ana._cached_values["balanced_trunc_error_bound"] > 0

    def test_balanced_truncation_invalid_r(self):
        """Balanced truncation raises ValueError for r = 0 or r > n."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        with pytest.raises(ValueError):
            ana.balanced_truncate(0)
        with pytest.raises(ValueError):
            ana.balanced_truncate(3)


class TestSpectralDiagnostics:
    """Verify Gramian spectrum and condition number methods."""

    def test_gramian_spectrum(self):
        """Gramian spectrum returns eigenvalues of the controllability Gramian."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        eigs = ana.gramian_spectrum("Wc")
        assert len(eigs) == 2

    def test_gramian_condition(self):
        """Gramian condition returns a finite positive number for full-rank Gramians."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        cond = ana.gramian_condition("Wc")
        assert cond > 0

    def test_gramian_spectrum_unknown(self):
        """Gramian spectrum raises ValueError for an unknown identifier."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        with pytest.raises(ValueError):
            ana.gramian_spectrum("Wx")

    def test_gramian_condition_unknown(self):
        """Gramian condition raises ValueError for an unknown identifier."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        with pytest.raises(ValueError):
            ana.gramian_condition("Wx")


class TestUtility:
    """Verify utility methods: rank_report, summary, reset_cache."""

    def test_rank_report(self):
        """rank_report returns a dict with controllability and observability keys."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        report = ana.rank_report()
        assert "controllability" in report
        assert "observability" in report

    def test_summary(self):
        """summary returns a string containing the system order."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        summary = ana.summary()
        assert "System order" in summary

    def test_reset_cache(self):
        """reset_cache clears all memoised Gramian results."""
        A = np.array([[0, 1], [-1, -2]], dtype=float)
        B = np.array([[0], [1]], dtype=float)
        C = np.eye(2)
        ana = _make_ana(A, B, C)
        ana.controllability_gramian()
        assert "Wc" in ana._cached_values
        ana.reset_cache()
        assert "Wc" not in ana._cached_values


class TestErrorHandling:
    """Verify that invalid inputs raise appropriate errors."""

    def test_non_square_A(self):
        """A must be square; a 3x2 matrix raises ValueError."""
        with pytest.raises(ValueError):
            _make_ana(np.array([[1, 2], [3, 4], [5, 6]]), np.eye(2), np.eye(2))

    def test_B_row_mismatch(self):
        """B must have the same number of rows as A."""
        with pytest.raises(ValueError):
            _make_ana(np.eye(2), np.eye(3), np.eye(2))

    def test_C_col_mismatch(self):
        """C must have the same number of columns as A."""
        with pytest.raises(ValueError):
            _make_ana(np.eye(2), np.eye(2), np.eye(3))
