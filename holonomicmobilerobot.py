import numpy as np

class HolonomicMobileRobot:
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
        self.state=np.zeros(3)

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

        Args:
            u_world: Desired velocity vector in world coordinates.

        Returns:
            Calculated wheel speeds.
        """
        theta=self.state[2]
        c, s= np.cos(theta), np.sin(theta)
        rot_matrix= np.array([[c,s,0],[-s,c,0],[0,0,1]])
        A_kinematics, A_pinv_kin=self.mobilerobotkinematics()
        u_body=rot_matrix@u_world
        wheel_speeds=(1/self.r)*A_kinematics@u_body
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
        self.state=np.array([x,y,theta])

    def get_state(self):
        """
        Return the current state of the robot.

        Returns:
            The state vector [x, y, theta].
        """
        return self.state

    def get_model_matrices(self):
        A=np.eye(3)
        B=self.dt*np.eye(3)
        return A,B
        

        
        

    


        
        
        