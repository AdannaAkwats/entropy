import numpy as np
from numpy import linalg as LA
import math
import random


def isMatrixSame(A, B):
    """
    Returns true if matrices A and B are equal
    """
    dimA = A.shape
    dimB = B.shape

    if(dimA != dimB):
        return False

    for i in range(dimA[0]):
        for j in range(dimA[0]):
            if(not isclose(A[i,j],B[i,j])):
                return False
    return True


def matrixInList(A, L):
    """
    Returns true if matrix A is the list L
    """
    # if list empty
    if(not L):
        return False

    for l in L:
        if(isMatrixSame(A, l)):
            return True
    return False


def isclose(a, b, rel_tol=1e-14, abs_tol=0.0):
    """
    Compares floating point numbers
    """
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def allocSub(p):
    """
    Allocates space needed to store each separated density matrix of p
    """
    dim = p.shape[0]
    sub_dim = 0
    
    if(isPowerof2(dim)): # qubit
        sub_dim = dim / 2
    elif(isPowerof3(dim)): # qutrit
        sub_dim = dim / 3

    # Error if sub_dim is still = 0
    if sub_dim == 0:
        print "Density matrix dimension not power of 2 (qubit) or 3 (qutrit)"

    temp = np.zeros((sub_dim, sub_dim))
    sub_p = np.matrix(temp, dtype=np.complex128)

    return sub_p

def isPowerof2(n):
    """
    Returns true if n is a power of 2 i.e can be written as 2^q = n
    """
    return (n and not (n & (n-1)))


def isPowerof3(n):
    """
    Returns true if n is a power of 3 i.e can be written as 3^q = n
    """
    # 3^19 = 1162261467
    return 1162261467 % n == 0;
