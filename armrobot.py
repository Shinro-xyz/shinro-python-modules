import numpy as np
from components import Plant

class ArmRobot(Plant):
    """
    A robotic arm plant model for Model Predictive Control (MPC).
    Tracks end-effector pose in 6D space and manages joint-level constraints.
    """
    def __init__(self, num_dof:int, dt:float, joint_limits:np.ndarray,joint_offsets:np.ndarray, rot_axes: list[str]):
        """
        Initialize the ArmRobot.

        Args:
            num_dof (int): Number of degrees of freedom.
            dt (float): Time step for system integration.
            joint_limits (np.ndarray): Shape (num_dof, 2) array of [min, max] limits per joint.
            joint_offsets (np.ndarray): Shape (num_dof, 3) translation offsets for each joint.
            rot_axes (list[str]): List of rotation axes ('x', 'y', or 'z') for each joint.
        """
        self.num_dof=num_dof
        self.dt=dt
        self.state=np.zeros(6)
        self.joint_offsets=joint_offsets
        self.joint_limits=joint_limits
        self.axes=rot_axes
        self._last_joints=np.zeros(num_dof)

    def get_state(self):
        """Returns a copy of the current 6D end-effector pose state.
        
        If a MuJoCoEngine is attached, reads the actual physics state.
        """
        if hasattr(self, '_engine') and self._engine is not None:
            return self._engine.get_arm_qpos()
        return self.state.copy()
 
    def get_model(self):
        """
        Returns the linear system matrices A and B for the plant model.
        Returns:
            A (np.ndarray): State transition matrix.
            B (np.ndarray): Control input matrix.
        """
        A=np.eye(6)
        B=self.dt*np.eye(6)
        return A,B
 
    def _pose_to_transform(self,pose:np.ndarray):
        """
        Converts a 6D pose (x, y, z, roll, pitch, yaw) to a 4x4 homogeneous transformation matrix.
        Args:
            pose (np.ndarray): 6D pose vector.
        Returns:
            T (np.ndarray): 4x4 transformation matrix.
        """
        x,y,z,roll,pitch,yaw=pose
        Rx=np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
        Ry=np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
        Rz=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
        T=np.eye(4)
        T[:3,:3]=Rz@Ry@Rx
        T[:3,3]=[x,y,z]
        return T

    def step(self,u:np.ndarray):
        """
        Updates the system state based on control input u and returns joint angles.

        If a MuJoCoEngine is attached (via _engine), delegates to physics.
        Otherwise uses the simple integrator model.

        Args:
            u (np.ndarray): 6D control input (velocity/displacement).
        Returns:
            np.ndarray: Clipped joint angles for the new state.
        """
        if hasattr(self, '_engine') and self._engine is not None:
            # MuJoCo backend: send position targets to arm actuators
            self._engine.set_arm_ctrl(u)
            # Don't step here — LeKiwiSim.step() does the single physics step
            self._last_joints = self._engine.get_arm_qpos()
            self.state = self._last_joints.copy()
            return self._last_joints

        # Fallback: simple integrator
        self.state+=self.dt*u
        target=self._pose_to_transform(self.state)
        q=self.inverse_kinematics(target)
        self._last_joints=np.clip(q,self.joint_limits[:,0],self.joint_limits[:,1])
        return self._last_joints
    
    def _homogenous_transform(self,joint_angles:np.ndarray):
        """
        Computes individual joint transformation matrices based on current angles and axes.
        Args:
            joint_angles (np.ndarray): Current joint angles.
        Returns:
            T (np.ndarray): Array of 4x4 transformation matrices per joint.
        """
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
        """
        Computes the end-effector pose and intermediate joint positions/axes.
        Args:
            joint_angles (np.ndarray): Input joint angles.
        Returns:
            T_cumulative (np.ndarray): Final 4x4 end-effector transformation matrix.
            positions (list): List of 3D positions for each joint.
            axes (list): List of 3D rotation axes for each joint.
        """
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
        """
        Computes the 6xN geometric Jacobian matrix for the arm.
        Args:
            joint_angles (np.ndarray): Current joint angles.
        Returns:
            J (np.ndarray): 6xN Jacobian matrix mapping joint velocities to end-effector twist.
        """
        T_endeffector, pos, axes= self.forward_kinematics(joint_angles)
        p_endeffector=pos[-1]
        J=np.zeros((6,self.num_dof))
        for i in range(self.num_dof):
            J[:3,i]=np.cross(axes[i],p_endeffector-pos[i])
            J[3:,i]=axes[i]
        return J
 
    def inverse_kinematics(self,target_pose:np.ndarray,max_iters:int=100, q_init=None,tol:float=1e-4, max_step:float=0.2):
        """
        Solves for joint angles using an iterative Newton-Raphson method (Jacobian pseudo-inverse).
        Args:
            target_pose (np.ndarray): 4x4 target transformation matrix.
            max_iters (int): Maximum iterations for convergence.
            q_init (np.ndarray, optional): Initial joint angles guess.
            tol (float): Convergence tolerance for position and orientation error.
            max_step (float): Maximum joint step per iteration to maintain stability.
        Returns:
            q (np.ndarray): Resulting joint angles clipped to joint limits.
        """
        q=q_init if q_init is not None else self._last_joints.copy()
        for j in range(max_iters):
            T_cur, positions,axes=self.forward_kinematics(q)
            pos_err=target_pose[:3,3]-T_cur[:3,3]
            R_err=target_pose[:3,:3]@T_cur[:3,:3].T
            angle=np.arccos(np.clip((np.trace(R_err)-1)/2,-1,1))
 
            if angle<tol and np.linalg.norm(pos_err)<tol:
                break
 
            axis = np.array([R_err[2,1]-R_err[1,2],
                                 R_err[0,2]-R_err[2,0],
                                 R_err[1,0]-R_err[0,1]])
 
            if np.linalg.norm(axis) > 1e-6:
                ori_err = (axis / np.linalg.norm(axis)) * angle
            else:
                ori_err = np.zeros(3)
            
            v = np.concatenate([pos_err, ori_err])  # 6D twist
          
            J=self._jacobian(q)
            # 4. Step toward target
            dq = np.linalg.pinv(J)@v
            dq = np.clip(dq,-max_step,max_step)
            q = q + dq
            q = np.clip(q, self.joint_limits[:, 0], self.joint_limits[:, 1])
        
        return q