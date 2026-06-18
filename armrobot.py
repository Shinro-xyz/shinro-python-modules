import numpy as np
from components import Plant

class ArmRobot(Plant):
    def __init__(self, num_dof:int, dt:float, joint_limits:np.ndarray,joint_offsets:np.ndarray):
        self.num_dof=num_dof
        self.dt=dt
        self.state=np.zeros(num_dof)
        self.joint_offsets=joint_offsets
        self.joint_limits=joint_limits

    def get_state(self):
        return self.state.copy()

    def get_model(self):
        A=np.eye(self.num_dof)
        B=self.dt*np.eye(self.num_dof)
        return A,B
    def _homogenous_transform(self,joint_angle:float,axis:str, joint_index:int):
        s,c= np.sin(joint_angle), np.cos(joint_angle)
        if axis=='x':
            R=np.array([[1,0,0],[0,c,-s],[0,s,c]])
        elif axis=='y':
            R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
        elif axis=='z':
            R=np.array([[c,-s,0],[s,c,0],[0,0,1]])

        offset_vector= self.joint_offsets([])
        
    def forward_kinematics(self,joint_angles:np.ndarray):
        pass