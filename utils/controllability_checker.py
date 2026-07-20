import numpy as np

def controllabilty(A:np.ndarray, B:np.ndarray):
    #verifying the inputs A and B
    A=np.asarray(A) # state matrix, dim: nxn
    B=np.asarray(B) # input matrix, dim: nxm

    n=A.shape[0]

    #constructing the controllability matrix

    cols=[B]
    for i in range(1,n):
        cols.append(A@cols[-1])

    C=np.hstack(cols)

    # check the rank of the matrix and controllability
    rank=np.linalg.matrix_rank(C)
    is_controllable=(rank==n)

    # checking for singular values
    U,s,t= np.linalg.svd(C)

    return C, is_controllable, rank

def observability(A:np.ndarray, C: np.ndarray):
    A=np.ndarray(A) #state matrix, dim: nxn
    C=np.ndarray(C) #output matrix, dim: nxm