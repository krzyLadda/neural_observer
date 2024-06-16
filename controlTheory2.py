import sympy as sp
import control as con
import numpy as np


def matrixNorm(A):
    foo = np.matmul(A.T, A)
    maxEig = max(eig(foo)[0])
    return np.sqrt(maxEig)


def vectorNorm(v):
    return np.sqrt(np.matmul(v.T, v))


def eig(A):
    return np.linalg.eig(A)


# returns the gain matrix from the state
# so that it locates the pole of the linear object \dot{x} = Ax
def place(A, B, setPoles):
    try:
        return con.place(A, B, setPoles)
    except ValueError:
        return False


def placeObserverPoles(A, C, setPoles):
    K = place(np.transpose(A), np.transpose(C), setPoles)
    if K is False:
        raise "A i C is not observable"
    return np.transpose(K)


# performs the kalman test for the observability of a linear system
def kalmanObservability(A, C):
    M = con.obsv(A, C)
    n = A.shape[0]
    if np.linalg.matrix_rank(M) >= n:
        return True
    else:
        return False


# checks that matrix A has all eigenvalues in the left half-plane
# All (Re(eig(A)) < 0
def isHurwitzMatrix(A):
    eigenValues, _ = np.linalg.eig(A)
    if all(x.real < 0 for x in eigenValues):
        return True
    else:
        return False