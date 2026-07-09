from typing import Optional
from components import Plant
from factories.registry import register_plant
from utils.array_backend import ArrayBackend, NumpyBackend
import numpy as np


@register_plant("HolonomicMobileRobot")
class HolonomicMobileRobot(Plant):
    """Holonomic mobile robot with omni-wheel kinematics.

    Models a robot with N wheels arranged symmetrically around a center.
    Maps world-frame velocity commands :math:`[v_x, v_y, \\omega]` to
    individual wheel speeds via the kinematic matrix :math:`A_{\\text{kin}}`.

    The forward kinematics matrix :math:`A_{\\text{kin}}` maps body-frame
    velocities to wheel speeds:

    .. math::

        \\omega_{\\text{wheels}} = \\frac{1}{r} A_{\\text{kin}} [v_x, v_y, \\omega]^T

    where each row of :math:`A_{\\text{kin}}` is
    :math:`[\\sin(\\theta_i), -\\cos(\\theta_i), -R]` for wheel i at angle
    :math:`\\theta_i`.

    When a MuJoCo engine is attached, ``step()`` also stores wheel rotation
    deltas for visual rolling in the physics simulation.

    The kinematics matrix is built once in ``__init__`` using numpy (scalar
    trig from Python floats), then converted to the backend via
    ``bk.from_numpy``. All per-step operations use ``bk.xxx``.

    Args:
        num_wheels: Number of wheels (e.g., 3 for omni, 4 for mecanum).
        radius_robots: Distance from robot center to each wheel (m).
        gamma: Angle of the first wheel relative to the robot base (rad).
        radius_wheels: Radius of each wheel (m).
        dt: Simulation time step (s).
        backend: Array backend. Defaults to NumpyBackend.
    """

    def __init__(
        self,
        num_wheels: int,
        radius_robots: float,
        gamma: float,
        radius_wheels: float,
        dt: float,
        backend: Optional[ArrayBackend] = None,
    ):
        self.bk = backend or NumpyBackend()
        self.n = num_wheels
        self.R = radius_robots
        self.gamma = gamma
        self.r = radius_wheels
        self.dt = dt
        self.state = self.bk.zeros(3)
        self.A_kinematics, self.A_pinv_kin = self._build_kinematics()
        self._engine = None

    def physics_engine(self, engine):
        """Attach a physics engine.

        After attachment, ``step()`` uses the engine for visual wheel rolling.
        The backend is inherited from the engine.

        Args:
            engine: PhysicsEngine instance or None to detach.
        """
        self._engine = engine
        if engine is not None:
            self.bk = engine.backend

    def _build_kinematics(self):
        """Compute the wheel kinematics matrix and its pseudoinverse.

        Uses numpy for the one-time construction (scalar trig from Python
        floats), then converts to the backend.

        Returns:
            Tuple of (A_kin, A_pinv) where A_kin is (n_wheels, 3) and
            A_pinv is its Moore-Penrose pseudoinverse (3, n_wheels).
        """
        theta_per_wheel = 2 * np.pi / self.n
        angle_list = [i * theta_per_wheel + self.gamma for i in range(self.n)]
        sin_list = np.sin(angle_list)
        cos_list = np.cos(angle_list)
        A_kin = np.column_stack((sin_list, -cos_list, np.full_like(sin_list, -self.R)))
        return self.bk.from_numpy(A_kin), self.bk.from_numpy(np.linalg.pinv(A_kin))

    def _rot_matrix(self, theta):
        """Build the 2D rotation matrix for world-to-body frame transform.

        Args:
            theta: Robot orientation (rad).

        Returns:
            Rotation matrix of shape (3, 3).
        """
        c = self.bk.cos(theta)
        s = self.bk.sin(theta)
        return self.bk.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])

    def step(self, u_world):
        """Update robot state and compute wheel speeds.

        Transforms the world-frame velocity command to body frame, computes
        wheel speeds via the kinematic matrix, and integrates the state.

        Args:
            u_world: Desired velocity :math:`[v_x, v_y, \\omega]` in world
                frame (3,).

        Returns:
            Wheel speed vector (n_wheels,).
        """
        theta = self.state[2]
        rot_matrix = self._rot_matrix(theta)
        u_body = rot_matrix @ u_world
        wheel_speeds = (1.0 / self.r) * self.A_kinematics @ u_body
        self.state = self.state + u_world * self.dt
        if self._engine is not None:
            self._target_wheel_delta = wheel_speeds * self.dt
        return wheel_speeds

    def set_pose(self, x: float, y: float, theta: float):
        """Set the robot's pose directly.

        Args:
            x: X position (m).
            y: Y position (m).
            theta: Orientation (rad).
        """
        self.state = self.bk.array([x, y, theta])

    def get_state(self):
        """Return the current pose :math:`[x, y, \\theta]`.

        Returns:
            State vector (3,) — [x, y, theta].
        """
        return self.bk.copy(self.state)

    def get_model(self):
        """Get the discrete-time state-space model.

        Returns:
            Tuple of (A, B) where A = I_3 and B = dt * I_3.
        """
        A = self.bk.eye(3)
        B = self.dt * self.bk.eye(3)
        return A, B

    @classmethod
    def from_config(cls, config, backend: Optional[ArrayBackend] = None):
        """Create a HolonomicMobileRobot from a TOML config dict.

        Config fields:
            num_wheels: Number of wheels.
            radius_robots: Distance from center to each wheel (m).
            gamma: First wheel angle offset (rad).
            radius_wheels: Wheel radius (m).
            dt: Time step.
            engine: Optional PhysicsEngine instance to attach.

        Args:
            config: TOML config dict.
            backend: Array backend. Defaults to NumpyBackend.

        Returns:
            HolonomicMobileRobot instance.
        """
        bk = backend or NumpyBackend()
        plant = cls(
            num_wheels=config["num_wheels"],
            radius_robots=config["radius_robots"],
            gamma=config["gamma"],
            radius_wheels=config["radius_wheels"],
            dt=config["dt"],
            backend=bk,
        )
        engine = config.get("engine")
        if engine is not None:
            plant.physics_engine(engine)
        return plant
