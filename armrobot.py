import numpy as np
from components import Plant

class ArmRobot(Plant):
    def __init__(self, num_dof:int, dt:float, joint_limits:np.ndarray,joint_offsets:np.ndarray, rot_axes: list[str]):
        self.num_dof=num_dof
        self.dt=dt
        self.state=np.zeros(num_dof)
        self.joint_offsets=joint_offsets
        self.joint_limits=joint_limits
        self.axes=rot_axes

    def get_state(self):
        return self.state.copy()

    def get_model(self):
        A=np.eye(self.num_dof)
        B=self.dt*np.eye(self.num_dof)
        return A,B
    def _homogenous_transform(self,joint_angles:np.ndarray):
        sines,cosines= np.sin(joint_angles), np.cos(joint_angles)
        T=np.zeros((self.num_dof,4,4))
        for i in range(self.num_dof):
            axis=self.axes[i]
            s,c=sines[i], cosines[i]
            if axis=='x':
                R=np.array([[1,0,0],[0,c,-s],[0,s,c]])
            elif axis=='y':
                R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
            elif axis=='z':
                R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            else:
                raise ValueError(f"Invalid axis '{axis}' at joint {i}. Choose 'x', 'y', or 'z'.")
            offset_vector= self.joint_offsets[i,:3]
            T[i,:3,:3]=R
            T[i,:3,3]=offset_vector
            T[i,3,3]=1.0
        return T

    def forward_kinematics(self,joint_angles:np.ndarray):
        T_joints= self._homogenous_transform(joint_angles)
        T_cumulatative=np.eye(4)
        positions=[]
        axes=[]
        for i in range(self.num_dof):
            T_cumulatative=T_cumulatative@T_joints[i]
            axis_local = {'x': [1,0,0], 'y': [0,1,0], 'z': [0,0,1]}[self.axes[i]]
            z_i = T_cumulatative[:3, :3] @ axis_local
            positions.append(T_cumulatative[:3, 3])
            axes.append(z_i)

        return T_cumulatative, positions, axes

    def _jacobian(self,joint_angles:np.ndarray):
        T_endeffector, pos, axes= self.forward_kinematics(joint_angles)
        p_endeffector=pos[-1]
        J=np.zeros((6,self.num_dof))
        for i in range(self.num_dof):
            J[:3,i]=np.cross(self.axes[i],p_endeffector-pos[i])
            J[3:,i]=axes[i]
        return J

    def inverse_kinematics(self,target_pos:np.ndarray,max_iters:int, q_init=None,tol:float=1e-4):
        q=q_init if q_init is not None else self.get_state()
        for j in range(max_iters):
            T_cur, positions,axes=self.forward_kinematics(q)
            pos_err=target_pos[:3,3]-T_cur[:3,3]
            R_err=target_pos[:3,:3]@T_cur[:3,:3].T
            angle=np.arccos(np.clip(np.))
            
            
            
            
        
        
        
           
            
        
        