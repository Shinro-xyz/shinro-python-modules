import numpy as np

def MobileRobotBaseKinematics(n:int,R:float,gamma:float, r:float):

    """Covers holonomic robots, which lekiwi is """
    # n:number of wheels
    # gamma:first wheel angle from base of the robot, use the MJCF definition
    # R: is the robot’s radius / the distance between the robot’s center and the wheels.
    theta=2*np.pi/n
    angle_list=[]
    for i in range(n):
        angle=i*theta+gamma
        angle_list.append(angle)
        
    A=np.zeros((n,n))
    sin_list=np.sin(angle_list)
    cos_list=np.cos(angle_list)

    
    

lekiwi=MobileRobotBaseKinematics(n=3,R=1,gamma=-np.pi/2,r=1)
print(type(lekiwi))