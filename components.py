from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np
from utils.array_backend import ArrayBackend, NumpyBackend


class PhysicsEngine(ABC):
    """
    Abstract base class for physics engines (MuJoCo, PyBullet, Drake, etc.).

    Provides name-based access to joints, bodies, and actuators. Plants use
    this interface instead of importing a physics engine directly.
    """

    @abstractmethod
    def get_joint_qpos(self, name: str) -> float:
        """Get a single joint position by name."""
        pass

    @abstractmethod
    def set_joint_qpos(self, name: str, value: float):
        """Set a single joint position by name."""
        pass

    @abstractmethod
    def get_joint_vel(self, name: str) -> float:
        """Get a single joint velocity by name."""
        pass

    @abstractmethod
    def set_joint_ctrl(self, name: str, value: float):
        """Set a single actuator control signal by name."""
        pass

    @abstractmethod
    def get_joint_limits(self, name: str) -> tuple[float, float]:
        """Get [min, max] limits for a joint by name."""
        pass

    @abstractmethod
    def get_body_xpos(self, name: str) -> np.ndarray:
        """Get 3D position of a body by name."""
        pass

    @abstractmethod
    def get_body_id(self, name: str) -> int:
        """Get the internal body ID for a named body. Returns -1 if not found."""
        pass

    @abstractmethod
    def compute_jacobian(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (jacp, jacr) — 3×nv position and orientation Jacobians for a body."""
        pass

    @abstractmethod
    def compute_jacobian_for_joints(self, body_name: str, joint_names: list[str]) -> np.ndarray:
        """Return the 6×k Jacobian (position + orientation) for a body, extracting only
        the columns corresponding to the named joints. Returns shape (6, k)."""
        pass

    @abstractmethod
    def forward(self):
        """Run forward kinematics (mj_forward equivalent)."""
        pass

    @abstractmethod
    def step(self):
        """Advance physics by one timestep."""
        pass

    @abstractmethod
    def reset(self, qpos: Optional[np.ndarray] = None):
        """Reset simulation state."""
        pass

    @abstractmethod
    def get_sensor_data(self) -> dict:
        """Return a dict of all sensor data (qpos, qvel, ctrl, time, etc.)."""
        pass

    @property
    @abstractmethod
    def dt(self) -> float:
        """Simulation timestep."""
        pass

    @property
    @abstractmethod
    def nv(self) -> int:
        """Number of velocity DOFs (for Jacobian column count)."""
        pass

    @property
    @abstractmethod
    def joint_names(self) -> list[str]:
        """List of all joint names in the model."""
        pass

    @property
    @abstractmethod
    def actuator_names(self) -> list[str]:
        """List of all actuator names in the model."""
        pass

    @property
    @abstractmethod
    def body_names(self) -> list[str]:
        """List of all body names in the model."""
        pass

    @property
    def backend(self) -> ArrayBackend:
        """Array backend for this engine. Subclasses may override."""
        return NumpyBackend()


class Controller(ABC):
    """
    Abstract base class for all controllers.

    A controller computes control actions (e.g., torques, voltages) from
    reference signals and/or state feedback. Subclasses implement specific
    control laws (PID, MPC, LQR, etc.).

    Usage:
        controller = MyController(...)
        action = controller.compute(reference=ref, state=x)
        controller.reset()
    """
    @abstractmethod
    def compute(self,*args:Any,**kwargs:Any)-> Any:
        """
        Compute the control action.

        Args:
            *args: Positional arguments for computation.
            **kwargs: Keyword arguments for computation.

        Returns:
            The computed control action.
        """
        pass

    def reset(self,*args:Any,**kwargs:Any)-> Any:
        """
        Reset the controller to its initial state.

        Args:
            *args: Positional arguments for reset.
            **kwargs: Keyword arguments for reset.

        Returns:
            None.
        """
        pass

class Plant(ABC):
    """
    Abstract base class for a system plant.

    A plant represents the system to be controlled (e.g., a robot, motor,
    or dynamical system). It provides the current state, a model for
    prediction/optimization, and a step method for simulation.

    Usage:
        plant = MyPlant(...)
        state = plant.get_state()
        model = plant.get_model()
        next_state = plant.step(control_input)
    """
    @abstractmethod
    def get_state(self, *args:Any, **kwargs:Any)->Any:
        """
        Get the current state of the plant.

        Returns:
            The current state (e.g., joint positions, velocities).
        """
        pass

    @abstractmethod
    def get_model(self, *args:Any, **kwargs:Any)->Any:
        """
        Get the mathematical model of the plant.

        For linear plants, returns (A, B) state-space matrices.
        For nonlinear plants, accepts optional ``x0`` and ``u0`` keyword
        arguments to linearize the dynamics around an operating point.

        Returns:
            The plant model (e.g., a dynamics function, matrices, or
            a callable used by the controller for prediction).
        """
        pass

    @abstractmethod
    def step(self, *args:Any, **kwargs:Any)->Any:
        """
        Perform a single time step simulation or execution of the plant.

        Args:
            *args: Positional arguments (e.g., control input).
            **kwargs: Keyword arguments.

        Returns:
            The next state or result of the step.
        """
        pass

    @abstractmethod
    def physics_engine(self, engine: Optional[PhysicsEngine], *args: Any, **kwargs: Any)-> Any:
        """
        Attach a physics engine to the plant.

        Args:
            engine: A PhysicsEngine instance, or None to detach.
        """
        pass

    def dynamics(self, state: Any, control: Any) -> Any:
        """Continuous-time dynamics :math:`\\dot{x} = f(x, u)`.

        Override in nonlinear plants to expose the dynamics function for
        linearization. Returns None by default (linear plants need not
        override this).

        Args:
            state: Current state vector (n_x,).
            control: Control input vector (n_u,).

        Returns:
            Time derivative of the state (n_x,), or None if not implemented.
        """
        return None

class StateEstimator(ABC):
    """
    Abstract base class for state estimators.

    A state estimator reconstructs the full system state from noisy
    measurements and known control inputs. Subclasses implement filters
    such as Kalman filters, observers, or complementary filters.

    Usage:
        estimator = MyEstimator(...)
        state = estimator.estimate(measurement=y, control_input=u)
        estimator.reset()
    """
    @abstractmethod
    def estimate(self,measurement:Any,control_input:Any)->Any:
        """
        Estimate the current state from a measurement and control input.

        Args:
            measurement: Sensor measurement (e.g., encoder reading, IMU).
            control_input: Control input applied to the system.

        Returns:
            The estimated state.
        """
        pass
    def reset(self):
        """
        Reset the estimator to its initial condition.

        Clears any internal buffer or covariance state.
        """
        pass

class TrajectoryGenerator(ABC):
    """
    Abstract base class for trajectory generators.

    A trajectory generator produces a reference path (position, velocity,
    acceleration) from a start to an end configuration. Subclasses
    implement splines, minimum-jerk, or motion-primitive generators.

    Usage:
        generator = MyTrajectoryGenerator(...)
        trajectory = generator.generate(start=start_pos, end=end_pos)
        generator.reset()
    """
    @abstractmethod
    def generate(self, start_position: Any, end_position: Any, duration: Any, *args: Any, **kwargs: Any) -> Any:
        """
        Generate a trajectory from start to end position over a given duration.

        Args:
            start_position: Initial configuration.
            end_position: Final target configuration.
            duration: Total time duration of the trajectory in seconds.

        Returns:
            The generated trajectory (e.g., a sequence of waypoints or
            a callable that returns setpoints at a given time).
        """
        pass

    def reset(self):
        """
        Reset the trajectory generator to its initial state.

        Clears any cached or ongoing trajectory data.
        """
        pass
        
    @abstractmethod
    def position_at(self,t:float, *args:Any, **kwargs: Any)->Any:
        pass
