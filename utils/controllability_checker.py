import numpy as np

class LTISystemsAnalyzer:
    def __init__(self, A:np.ndarray, B:np.ndarray= None, C:np.ndarray=None, D:np.ndarray= None, dt:float=None):
        self.A=A #state matrix
        self.B=B if B is not None else np.zeros((A.shape[0],0)) #input matrix
        self.C=C if C is not None else np.zeros((0,A.shape[0])) #output matrix
        self.D = D if D is not None else np.zeros((self.C.shape[0],self.B.shape[1])) #feedthrough matrix
        self.dt=dt
        self._cached_values={}

    def __post_init__(self)->None:
        n=self.A.shape[0] #how many states?
        if self.A.shape[0]!=self.A.shape[1]:
            raise ValueError("A must be a square matrix")

    def controllabilty(self):
        #verifying the inputs A and B
        
        n=self.A.shape[0]
    
        #constructing the controllability matrix
    
        cols=[self.B]
        for i in range(1,n):
            cols.append(self.A@cols[-1])
    
        C=np.hstack(cols)
    
        # check the rank of the matrix and controllability
        rank=np.linalg.matrix_rank(C)
        is_controllable=(rank==n)
        return C
    
    def observability(self):
        n= self.A.shape[0]
        cols=[self.C]
    
        for i in range(1,n):
            cols.append(self.C@np.linalg.matrix_power(self.A,i))
    
        O=np.vstack(cols) #stacking the list vertically
    
        rank= np.linalg.matrix_rank(O)
        is_observable=(rank==n)
    
        return O

    def _rank_and_conidtion