import numpy as np
from components import Plant

class HolonomicMobileRobot(Plant):
    """
    Represents a holonomic mobile robot with multiple wheels.
    """
    def __init__(self, num_wheels:int,radius_robots:float,gamma:float,radius_wheels:float, dt:float):
        """
        Initialize the holonomic robot.

        Args:
            num_wheels: Number of wheels.
            radius_robots: Distance from center to wheels.
            gamma: First wheel angle from the base.
            radius_wheels: Radius of each wheel.
            dt: Time step.
        """
        # n:number of wheels
        # gamma:first wheel angle from base of the robot, use the MJCF definition
        # R: is the robot’s radius / the distance between the robot’s center and the wheels.
        self.n=num_wheels
        self.R=radius_robots
        self.gamma=gamma
        self.r=radius_wheels
        self.dt=dt
        self.state=np.zeros(3, dtype=np.float64)
        self.A_kinematics, self.A_pinv_kin=self.mobilerobotkinematics()


    def mobilerobotkinematics(self):
        """
        Compute the robot kinematics matrix and its pseudoinverse.

        Returns:
            A tuple containing the kinematics matrix A_kin and its pseudoinverse.
        """
        theta_perwheel=2*np.pi/self.n
        angle_list=[]
        for i in range(self.n):
            angle=i*theta_perwheel+self.gamma
            angle_list.append(angle)
        sin_list=np.sin(angle_list)
        cos_list=np.cos(angle_list)
        #inverse kinematics, LTI matrix
        A_kin=np.column_stack((sin_list,-cos_list,np.full_like(sin_list, -self.R)))
        return A_kin, np.linalg.pinv(A_kin)

    def step(self,u_world):
        """
        Update robot state and compute wheel speeds for a given world-frame velocity.

        If a MuJoCoEngine is attached (via _engine), the arm uses MuJoCo physics
        but the base uses the simple integrator (omni-wheel contact physics is
        unreliable with simplified collision geoms).

        Args:
            u_world: Desired velocity vector in world coordinates.

        Returns:
            Calculated wheel speeds.
        """
        if hasattr(self, '_engine') and self._engine is not None:
            # Base: simple integrator (accurate for holonomic kinematics)
            theta = self.state[2]
            c, s = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
            u_body = rot_matrix @ u_world
            wheel_speeds = (1.0 / self.r) * self.A_kinematics @ u_body
            self.state += u_world * self.dt

            # Update MuJoCo base position to match
            self._engine.data.qpos[0] = self.state[0]
            self._engine.data.qpos[1] = self.state[1]
            # Convert yaw to quaternion
            yaw = self.state[2]
            self._engine.data.qpos[3] = np.cos(yaw / 2)
            self._engine.data.qpos[4] = 0.0
            self._engine.data.qpos[5] = 0.0
            self._engine.data.qpos[6] = np.sin(yaw / 2)

            # Step arm physics only (base is kinematic)
            self._engine.step()

            return wheel_speeds

        # Fallback: simple integrator
        theta=self.state[2]
        c, s= np.cos(theta), np.sin(theta)
        rot_matrix= np.array([[c,s,0],[-s,c,0],[0,0,1]])
        u_body=rot_matrix@u_world
        wheel_speeds=(1/self.r)*self.A_kinematics@u_body
        self.state+=u_world*self.dt
        return wheel_speeds

    def set_pose(self,x,y,theta):
        """
        Set the robot's current pose.

        Args:
            x: X-coordinate.
            y: Y-coordinate.
            theta: Orientation.
        """
        self.state=np.array([x,y,theta], dtype=np.float64)

    def get_state(self):
        """
        Return the current state of the robot.

        Returns:
            The state vector [x, y, theta].
        """
        return self.state.copy()

    def get_model(self):
        """
        Get the state-space model matrices A and B.

        Returns:
            A tuple containing the state transition matrix A and the input matrix B.
        """
        A=np.eye(3)
        B=self.dt*np.eye(3)
        return A,B