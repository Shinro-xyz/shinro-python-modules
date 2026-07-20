import numpy as np
from typing import Optional, Tuple, Dict
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov
from scipy.sparse import issparse, csr_matrix
from scipy.integrate import solve_ivp
from utils.array_backend import ArrayBackend, NumpyBackend


class LTISystemsAnalyzer:
    """Analyze linear time-invariant state-space systems.

    Provides controllability/observability checks, Gramian computations
    (continuous, discrete, finite-horizon), spectral diagnostics, Hankel
    singular values, balanced realization, and balanced truncation.

    Args:
        A: State matrix (n, n).
        B: Input matrix (n, m). Defaults to zeros.
        C: Output matrix (p, n). Defaults to zeros.
        D: Feedthrough matrix (p, m). Defaults to zeros.
        dt: Sampling time for discrete-time analysis. Defaults to None.
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        A: np.ndarray,
        B: Optional[np.ndarray] = None,
        C: Optional[np.ndarray] = None,
        D: Optional[np.ndarray] = None,
        dt: Optional[float] = None,
        backend: Optional[ArrayBackend] = None,
    ) -> None:
        self.bk = backend if backend is not None else NumpyBackend()
        self.A: np.ndarray = A
        self.B: np.ndarray = B if B is not None else self.bk.zeros((A.shape[0], 0))
        self.C: np.ndarray = C if C is not None else self.bk.zeros((0, A.shape[0]))
        self.D: np.ndarray = D if D is not None else self.bk.zeros((self.C.shape[0], self.B.shape[1]))
        self.dt: Optional[float] = dt
        self._cached_values = {}
        self._validate_dimensions()

    def _validate_dimensions(self) -> None:
        """Validate dimensions and fill missing matrices with zeros."""
        n = self.A.shape[0]

        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square (n×n).")

        def _zero(shape):
            if issparse(self.A):
                return csr_matrix(shape, dtype=float)
            return self.bk.zeros(shape)

        if self.B.shape[0] != n:
            raise ValueError("B must have same row count as A.")
        if self.C.shape[1] != n:
            raise ValueError("C must have same column count as A.")
        if self.D.shape != (self.C.shape[0], self.B.shape[1]):
            raise ValueError("D must be (p×m) consistent with C and B.")

    def controllabilty(self):
        """Build the controllability matrix C = [B, AB, A²B, ..., A^{n-1}B].

        Returns:
            Controllability matrix (n, n*m).
        """
        n = self.A.shape[0]
        cols = [self.B]
        for i in range(1, n):
            cols.append(self.A @ cols[-1])
        C = self.bk.hstack(cols)
        rank = self.bk.matrix_rank(C)
        is_controllable = (rank == n)
        return C

    def observability(self):
        """Build the observability matrix O = [C; CA; CA²; ...; CA^{n-1}].

        Returns:
            Observability matrix (n*p, n).
        """
        n = self.A.shape[0]
        cols = [self.C]
        for i in range(1, n):
            cols.append(self.C @ self.bk.matrix_power(self.A, i))
        O = self.bk.vstack(cols)
        rank = self.bk.matrix_rank(O)
        is_observable = (rank == n)
        return O

    def _rank_and_condition_check(self, M: np.ndarray):
        """Compute rank and condition number of a matrix.

        Args:
            M: Input matrix (n, n).

        Returns:
            Tuple of (rank, condition number). Condition is inf if singular.
        """
        rank = self.bk.matrix_rank(M)
        cond = self.bk.cond(M) if rank == M.shape[0] else np.inf
        return rank, cond

    def is_controllable(self):
        """Check whether the system is controllable via Kalman rank test.

        Returns:
            True if the controllability matrix has full row rank.
        """
        C = self.controllabilty()
        rank = self.bk.matrix_rank(C)
        return rank == self.A.shape[0]

    def is_observable(self):
        """Check whether the system is observable via Kalman rank test.

        Returns:
            True if the observability matrix has full column rank.
        """
        O = self.observability()
        rank = self.bk.matrix_rank(O)
        return rank == self.A.shape[0]

    def _solve_continuous_lyap(self, Q: np.ndarray):
        """Solve the continuous Lyapunov equation A W + W A^T + Q = 0.

        Args:
            Q: Right-hand side matrix (n, n).

        Returns:
            Solution W (n, n).
        """
        A_np = self.bk.to_numpy(self.A)
        Q_np = self.bk.to_numpy(-Q)
        Wc_np = solve_continuous_lyapunov(A_np, Q_np)
        return self.bk.from_numpy(Wc_np)

    def controllability_gramian(self) -> np.ndarray:
        """Infinite-horizon controllability Gramian (continuous-time).

        Returns cached result unless the system has changed.

        Returns:
            Controllability Gramian Wc (n, n).

        Raises:
            ValueError: If A is not Hurwitz (eigenvalues with non-negative real part).
        """
        if "Wc" in self._cached_values:
            return self._cached_values["Wc"]

        eigA = self.bk.eigvals(self.A)
        if self.bk.any(self.bk.real(eigA) >= 0):
            raise ValueError(
                "A is not Hurwitz; infinite-horizon controllability Gramian does not exist. "
                "Use controllability_gramian_finite(T) instead."
            )
        Q = self.B @ self.B.T
        Wc = self._solve_continuous_lyap(Q)
        self._cached_values["Wc"] = Wc
        return Wc

    def observability_gramian(self) -> np.ndarray:
        """Infinite-horizon observability Gramian (continuous-time).

        Returns cached result unless the system has changed.

        Returns:
            Observability Gramian Wo (n, n).

        Raises:
            ValueError: If A is not Hurwitz (eigenvalues with non-negative real part).
        """
        if "Wo" in self._cached_values:
            return self._cached_values["Wo"]

        eigA = self.bk.eigvals(self.A)
        if self.bk.any(self.bk.real(eigA) >= 0):
            raise ValueError(
                "A is not Hurwitz; infinite-horizon observability Gramian does not exist. "
                "Use observability_gramian_finite(T) instead."
            )
        Q = self.C.T @ self.C
        Wo = self._solve_continuous_lyap(Q)
        self._cached_values["Wo"] = Wo
        return Wo

    def _solve_discrete_lyap(self, A: np.ndarray, M: np.ndarray):
        """Solve the discrete Lyapunov equation A W A^T - W + M = 0.

        Args:
            A: State matrix (n, n).
            M: Right-hand side matrix (n, n).

        Returns:
            Solution W (n, n).

        Raises:
            ValueError: If the system is not asymptotically stable or dt is None.
        """
        eigs = self.bk.eigvals(self.A)
        if self.bk.any(self.bk.abs(eigs) >= 1.0):
            raise ValueError(
                "Discrete-time system is not asymptotically stable (|eig| >= 1). "
                "The infinite-horizon Gramian does not exist. "
                "Use a finite-horizon Gramian instead."
            )

        if self.dt is None:
            raise ValueError(
                "dt is None, use controllability_gramian() for continuous functions, "
                "set dt > 0 for discrete-time systems"
            )

        A_np = self.bk.to_numpy(A)
        M_np = self.bk.to_numpy(M)
        W_np = solve_discrete_lyapunov(A_np, M_np)
        return self.bk.from_numpy(W_np)

    def discrete_controllability_gramian(self) -> np.ndarray:
        """Discrete-time infinite-horizon controllability Gramian.

        Solves A Wc A^T - Wc + B B^T = 0.

        Returns:
            Discrete controllability Gramian Wc (n, n).
        """
        if "Wc_discrete" in self._cached_values:
            return self._cached_values["Wc_discrete"]

        M = self.B @ self.B.T
        Wc_discrete = self._solve_discrete_lyap(self.A, M)
        self._cached_values["Wc_discrete"] = Wc_discrete
        return Wc_discrete

    def discrete_observability_gramian(self) -> np.ndarray:
        """Discrete-time infinite-horizon observability Gramian.

        Solves A^T Wo A - Wo + C^T C = 0.

        Returns:
            Discrete observability Gramian Wo (n, n).
        """
        if "Wo_discrete" in self._cached_values:
            return self._cached_values["Wo_discrete"]

        M = self.C.T @ self.C
        Wo_discrete = self._solve_discrete_lyap(self.A.T, M)
        self._cached_values["Wo_discrete"] = Wo_discrete
        return Wo_discrete

    def _gramian_ode(self, t, w_flat, Q, A_mat):
        """ODE for finite-horizon Gramian integration: dW/dt = A W + W A^T + Q.

        Args:
            t: Current time (unused, for solve_ivp interface).
            w_flat: Flattened W matrix (n*n,).
            Q: Right-hand side matrix (n, n).
            A_mat: State matrix (n, n).

        Returns:
            Flattened dW/dt (n*n,).
        """
        n = A_mat.shape[0]
        W = w_flat.reshape((n, n))
        dW = A_mat @ W + W @ A_mat.T + Q
        return dW.ravel()

    def _finite_horizon_gramian(
        self,
        Q: np.ndarray,
        T: float,
        method: str = "RK45",
        num_pts: int = 500,
    ) -> np.ndarray:
        """Common routine for finite-horizon Gramians via IVP integration.

        Args:
            Q: Right-hand side matrix (n, n).
            T: Horizon length.
            method: ODE solver method. Defaults to "RK45".
            num_pts: Number of evaluation points. Defaults to 500.

        Returns:
            Gramian at time T (n, n).

        Raises:
            RuntimeError: If IVP integration fails.
        """
        n = self.A.shape[0]
        A_mat = self.A.toarray() if issparse(self.A) else self.A
        sol = solve_ivp(
            fun=self._gramian_ode,
            t_span=(0.0, T),
            y0=self.bk.to_numpy(self.bk.zeros(n * n)),
            args=(Q, A_mat),
            method=method,
            t_eval=self.bk.to_numpy(self.bk.linspace(0.0, T, num_pts)),
        )
        if not sol.success:
            raise RuntimeError("IVP integration for Gramian failed.")
        return self.bk.from_numpy(sol.y[:, -1].reshape((n, n)))

    def controllability_gramian_finite(self, T: float) -> np.ndarray:
        """Finite-horizon controllability Gramian: Wc(T) = ∫₀ᵀ e^{Aτ} B B^T e^{A^Tτ} dτ.

        Works for any A (stable or unstable).

        Args:
            T: Horizon length.

        Returns:
            Finite-horizon controllability Gramian (n, n).

        Raises:
            ValueError: If T is not positive.
        """
        if T <= 0:
            raise ValueError("T (horizon) must be positive.")
        Q = self.B @ self.B.T
        return self._finite_horizon_gramian(Q, T)

    def observability_gramian_finite(self, T: float) -> np.ndarray:
        """Finite-horizon observability Gramian: Wo(T) = ∫₀ᵀ e^{A^Tτ} C^T C e^{Aτ} dτ.

        Works for any A (stable or unstable).

        Args:
            T: Horizon length.

        Returns:
            Finite-horizon observability Gramian (n, n).

        Raises:
            ValueError: If T is not positive.
        """
        if T <= 0:
            raise ValueError("T (horizon) must be positive.")
        Q = self.C.T @ self.C
        return self._finite_horizon_gramian(Q, T)

    def gramian_spectrum(self, gramian: str = "Wc") -> np.ndarray:
        """Return eigenvalues of a chosen Gramian.

        Args:
            gramian: Which Gramian to inspect. One of "Wc", "Wo",
                "Wc_finite:<T>", "Wo_finite:<T>". Defaults to "Wc".

        Returns:
            Real parts of Gramian eigenvalues (n,).
        """
        if gramian == "Wc":
            G = self.controllability_gramian()
        elif gramian == "Wo":
            G = self.observability_gramian()
        elif gramian.startswith("Wc_"):
            _, horizon = gramian.split(":")
            G = self.controllability_gramian_finite(float(horizon))
        elif gramian.startswith("Wo_"):
            _, horizon = gramian.split(":")
            G = self.observability_gramian_finite(float(horizon))
        else:
            raise ValueError(f"Unknown gramian identifier: {gramian}")

        return self.bk.real(self.bk.eigvals(G))

    def gramian_condition(self, gramian: str = "Wc") -> float:
        """Return the 2-norm condition number of the selected Gramian.

        Args:
            gramian: Which Gramian to inspect. One of "Wc", "Wo". Defaults to "Wc".

        Returns:
            Condition number, or inf if the Gramian is singular.
        """
        if gramian == "Wc":
            G = self.controllability_gramian()
        elif gramian == "Wo":
            G = self.observability_gramian()
        else:
            raise ValueError("Only infinite-horizon 'Wc' / 'Wo' supported for cond().")
        rank = self.bk.matrix_rank(G)
        if rank < G.shape[0]:
            return np.inf
        return self.bk.cond(G)

    def hankel_singular_values(self) -> np.ndarray:
        """Compute the Hankel singular values σ_i = sqrt(λ_i(Wc Wo)).

        Returns:
            Hankel singular values sorted in descending order (n,).
        """
        Wc = self.controllability_gramian()
        Wo = self.observability_gramian()
        prod = Wc @ Wo
        eigs = self.bk.real(self.bk.eigvals(prod))
        eigs = self.bk.where(eigs < 0, self.bk.zeros_like(eigs), eigs)
        sigma = self.bk.sqrt(self.bk.sort(eigs)[::-1])
        return sigma

    def balanced_realization(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the balanced state-space matrices (Abal, Bbal, Cbal).

        The transformation T satisfies:
            T^{-1} A T = Abal,
            T^{-1} B   = Bbal,
            C T        = Cbal,
        and the balanced Gramians are diag(σ_1, ..., σ_n).

        Returns:
            Tuple of (Abal, Bbal, Cbal) with shapes (n, n), (n, m), (p, n).
        """
        sigma = self.hankel_singular_values()

        Wc = self.controllability_gramian()
        Wo = self.observability_gramian()
        Lc = self.bk.cholesky(Wc)
        Lo = self.bk.cholesky(Wo)

        U, s, Vh = self.bk.svd(Lo.T @ Lc)
        T = Lc @ Vh.T @ self.bk.diag(1.0 / self.bk.sqrt(s))
        Tinv = self.bk.diag(1.0 / self.bk.sqrt(s)) @ U.T @ Lo.T

        Ab = Tinv @ self.A @ T
        Bb = Tinv @ self.B
        Cb = self.C @ T

        return Ab, Bb, Cb

    def balanced_truncate(self, r: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Perform balanced truncation to obtain an order-r reduced model.

        Args:
            r: Desired reduced order (0 < r <= n).

        Returns:
            Tuple of (Ar, Br, Cr, Dr) with shapes (r, r), (r, m), (p, r), (p, m).
        """
        n = self.A.shape[0]
        if not (0 < r <= n):
            raise ValueError("Reduced order r must satisfy 0 < r <= n.")

        Ab, Bb, Cb = self.balanced_realization()
        sigma = self.hankel_singular_values()

        Ar = Ab[:r, :r]
        Br = Bb[:r, :]
        Cr = Cb[:, :r]
        Dr = self.D

        error_bound = 2.0 * self.bk.sum(sigma[r:])
        self._cached_values["balanced_trunc_error_bound"] = error_bound

        return Ar, Br, Cr, Dr

    def reset_cache(self) -> None:
        """Clear all memoised results after manually changing A, B, or C."""
        self._cached_values.clear()

    def rank_report(self) -> Dict[str, Tuple[int, float]]:
        """Return rank and condition of controllability and observability matrices.

        Returns:
            Dict with keys 'controllability' and 'observability', each a
            tuple of (rank, condition_number).
        """
        C = self.controllabilty()
        O = self.observability()
        rank_c = self.bk.matrix_rank(C)
        rank_o = self.bk.matrix_rank(O)
        cond_c = self.bk.cond(C) if rank_c == C.shape[0] else np.inf
        cond_o = self.bk.cond(O) if rank_o == O.shape[0] else np.inf
        return {
            "controllability": (rank_c, cond_c),
            "observability": (rank_o, cond_o),
        }

    def summary(self) -> str:
        """Produce a human-readable report of system properties.

        Includes rank/condition, Gramian eigenvalues, Hankel singular values,
        and a balanced-truncation error bound if previously computed.

        Returns:
            Formatted summary string.
        """
        n = self.A.shape[0]
        rank_info = self.rank_report()
        wc = self.controllability_gramian() if self.is_controllable() else None
        wo = self.observability_gramian() if self.is_observable() else None

        lines = [
            f"System order n = {n}",
            f"Controllable?      {self.is_controllable()}",
            f"Observable?       {self.is_observable()}",
            "",
            "Kalman rank / condition:",
            f"  Controllability : rank = {rank_info['controllability'][0]},  cond = {rank_info['controllability'][1]:.2e}",
            f"  Observability   : rank = {rank_info['observability'][0]},    cond = {rank_info['observability'][1]:.2e}",
            "",
        ]

        if wc is not None:
            eig_wc = self.bk.real(self.bk.eigvals(wc))
            lines.append("Controllability Gramian eigenvalues (sorted):")
            lines.append("  " + ", ".join(f"{ev:.3e}" for ev in self.bk.to_numpy(self.bk.sort(eig_wc))[::-1]))
        else:
            lines.append("Controllability Gramian: *not defined* (unstable A).")

        if wo is not None:
            eig_wo = self.bk.real(self.bk.eigvals(wo))
            lines.append("Observability Gramian eigenvalues (sorted):")
            lines.append("  " + ", ".join(f"{ev:.3e}" for ev in self.bk.to_numpy(self.bk.sort(eig_wo))[::-1]))
        else:
            lines.append("Observability Gramian: *not defined* (unstable A).")

        sigma = self.hankel_singular_values()
        lines.append("")
        lines.append("Hankel singular values (σ1 >= σ2 ...):")
        lines.append("  " + ", ".join(f"{sv:.3e}" for sv in self.bk.to_numpy(sigma)))

        err = self._cached_values.get("balanced_trunc_error_bound")
        if err is not None:
            lines.append("")
            lines.append(f"Balanced-truncation error bound (2*sum(σ_{{r+1..n}})) = {err:.3e}")

        return "\n".join(lines)
