import numpy as np
from typing import Optional, Tuple, List
from components import Plant


class HolonomicMobileRobot(Plant):
    """Holonomic mobile robot with Mecanum/omni-wheel kinematics.

    Models a robot with N wheels arranged symmetrically around a center.
    Maps world-frame velocity commands [vx, vy, ω] to individual wheel speeds
    via the kinematic matrix A_kin.

    When a MuJoCo engine is attached, step() also stores wheel rotation deltas
    for visual rolling in the physics simulation.

    Usage:
        base = HolonomicMobileRobot(
            num_wheels=4,
            radius_robots=0.12,
            gamma=np.pi / 4,
            radius_wheels=0.03,
            dt=0.02,
        )
        wheel_speeds = base.step(np.array([0.2, 0.0, 0.0]))
    """

    def __init__(
        self,
        num_wheels: int,
        radius_robots: float,
        gamma: float,
        radius_wheels: float,
        dt: float,
    ):
        """Initialize the holonomic robot kinematics.

        Args:
            num_wheels: Number of wheels (e.g., 3 for omni, 4 for mecanum).
            radius_robots: Distance from robot center to each wheel (m).
            gamma: Angle of the first wheel relative to the robot base (rad).
            radius_wheels: Radius of each wheel (m).
            dt: Simulation time step (s).
        """
        self.n = num_wheels
        self.R = radius_robots
        self.gamma = gamma
        self.r = radius_wheels
        self.dt = dt
        self.state = np.zeros(3, dtype=np.float64)
        self.A_kinematics, self.A_pinv_kin = self.mobilerobotkinematics()
        self._engine = None

    def physics_engine(self, engine):
        """Attach a MuJoCo physics engine.

        After attachment, step() uses the engine for visual wheel rolling.

        Args:
            engine: MuJoCoEngine instance or None to detach.
        """
        self._engine = engine

    def mobilerobotkinematics(self):
        """Compute the wheel kinematics matrix and its pseudoinverse.

        The forward kinematics matrix A_kin maps body-frame velocities to
        wheel speeds:
            ω_wheels = (1/r) * A_kin @ [vx_body, vy_body, ω]^T

        where each row of A_kin is [sin(θ_i), -cos(θ_i), -R] for wheel i
        at angle θ_i.

        Returns:
            Tuple of (A_kin, A_pinv) where A_kin is (n_wheels, 3) and
            A_pinv is its Moore-Penrose pseudoinverse (3, n_wheels).
        """
        theta_per_wheel = 2 * np.pi / self.n
        angle_list = []
        for i in range(self.n):
            angle = i * theta_per_wheel + self.gamma
            angle_list.append(angle)
        sin_list = np.sin(angle_list)
        cos_list = np.cos(angle_list)
        A_kin = np.column_stack((sin_list, -cos_list, np.full_like(sin_list, -self.R)))
        return A_kin, np.linalg.pinv(A_kin)

    def step(self, u_world: np.ndarray):
        """Update robot state and compute wheel speeds.

        Transforms the world-frame velocity command to body frame, computes
        wheel speeds via the kinematic matrix, and integrates the state.

        Args:
            u_world: Desired velocity [vx, vy, ω] in world frame (3,).

        Returns:
            Wheel speed vector (n_wheels,).
        """
        if self._engine is not None:
            theta = self.state[2]
            c, s = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            u_body = rot_matrix @ u_world
            wheel_speeds = (1.0 / self.r) * self.A_kinematics @ u_body
            self.state += u_world * self.dt
            self._target_wheel_delta = wheel_speeds * self.dt
            return wheel_speeds

        theta = self.state[2]
        c, s = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        u_body = rot_matrix @ u_world
        wheel_speeds = (1 / self.r) * self.A_kinematics @ u_body
        self.state += u_world * self.dt
        return wheel_speeds

    def set_pose(self, x: float, y: float, theta: float):
        """Set the robot's pose directly.

        Args:
            x: X position (m).
            y: Y position (m).
            theta: Orientation (rad).
        """
        self.state = np.array([x, y, theta], dtype=np.float64)

    def get_state(self):
        """Return the current pose [x, y, θ].

        Returns:
            State vector (3,) — [x, y, theta].
        """
        return self.state.copy()

    def get_model(self):
        """Get the discrete-time state-space model.

        Returns:
            Tuple of (A, B) where A = I₃ and B = dt * I₃.
        """
        A = np.eye(3)
        B = self.dt * np.eye(3)
        return A, B