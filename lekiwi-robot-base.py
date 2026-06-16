import numpy as np

class HolonomicMobileRobot:
    def __init__(self, num_wheels:int,radius_robots:float,gamma:float,radius_wheels:float, dt:float):
        """Covers holonomic robots, which lekiwi is """
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
        theta=self.state[2]
        c, s= np.cos(theta), np.sin(theta)
        rot_matrix= np.array([[c,s,0],[-s,c,0],[0,0,1]])
        A_kinematics, A_pinv_kin=self.mobilerobotkinematics()
        u_body=rot_matrix@u_world
        wheel_speeds=
        
        
