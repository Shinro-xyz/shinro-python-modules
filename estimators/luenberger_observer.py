import numpy as np
from components import StateEstimator

class LuenbergerObserver(StateEstimator):
    """Luenberger observer for linear state estimation.

    Implements the discrete-time observer dynamics:
        x̂ₖ₊₁ = A x̂ₖ + B uₖ + L (yₖ − C x̂ₖ − D uₖ)

    where L is the observer gain chosen to place the eigenvalues of (A − LC)
    inside the unit circle for stable estimation.

    Attributes:
        A: State transition matrix (n x n).
        B: Control input matrix (n x m).
        C: Output matrix (p x n). Defaults to identity if None.
        D: Feedthrough matrix (p x m). Defaults to zeros if None.
        L: Observer gain matrix (n x p).
        x_hat: Current state estimate (n x 1).
    """

    def __init__(self, A:np.ndarray, B:np.ndarray, observer_gain:np.ndarray,C:np.ndarray | None=None, D: np.ndarray | None=None, x_hat:np.ndarray | None=None,x0:np.ndarray | None=None):
        """Initialize the Luenberger observer with system matrices and initial state.

        Args:
            A: State transition matrix (n x n).
            B: Control input matrix (n x m).
            observer_gain: Observer gain matrix L (n x p).
            C: Output matrix (p x n). If None, defaults to identity.
            D: Feedthrough matrix (p x m). If None, defaults to zeros.
            x_hat: Deprecated, use x0 instead.
            x0: Initial state estimate (n x 1). If None, defaults to zeros.
        """
        self.A=A
        self.B=B
        if C is None:
            self.C=np.eye(A.shape[0])
        else:
            self.C=C

        if D is None:
            self.D=np.zeros((self.C.shape[0], B.shape[1]))

        else: 
            self.D=D

        self.L= observer_gain

        if x0 is None:
            self.x_hat=np.zeros((A.shape[0],1))

        else: 
            self.x_hat=x0.copy()

    def estimate(self, measurement: np.ndarray, control_input: np.ndarray) -> np.ndarray:
        """Perform one step of state estimation using the Luenberger observer.

        Computes the predicted state from the dynamics, calculates the
        innovation (measurement residual), and corrects the prediction
        using the observer gain.

        Args:
            measurement: Output measurement yₖ (p x 1).
            control_input: Control input uₖ (m x 1).

        Returns:
            Updated state estimate x̂ₖ₊₁ (n x 1).
        """
        x_pred= self.A@self.x_hat+self.B@control_input
        innovations=measurement-(self.C@x_pred+self.D@control_input)
        self.x_hat=x_pred+self.L@innovations
        return self.x_hat

    def reset(self, x0:np.ndarray | None=None):
        """Reset the state estimate to a given value or zeros.

        Args:
            x0: New initial state estimate (n x 1). If None, resets to zeros.
        """
        self.x_hat=np.zeros((self.A.shape[0],1)) if x0 is None else x0.copy()
        