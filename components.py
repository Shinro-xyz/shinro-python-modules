from abc import ABC, abstractmethod
from typing import Any

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
    def generate(self, start_postion:Any, end_position: Any, duration: Any)-> Any:
        """
        Generate a trajectory from start to end position over a given duration.

        Args:
            start_postion: Initial configuration.
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
