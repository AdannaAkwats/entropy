import numpy as np
from numpy import linalg as LA
import math
import random



# Returns true if matrices A and B are equal
def isMatrixSame(A, B):
    dimA = A.shape
    dimB = B.shape

    if(dimA != dimB):
        return False

    for i in range(dimA[0]):
        for j in range(dimA[0]):
            if(A[i,j] != B[i,j]):
                return False
    return True



# Returns true if matrix A is the list L
def matrixInList(A, L):
    # if list empty
    if(not L):
        return False

    for l in L:
        if(isMatrixSame(A, l)):
            return True
    return False

# Compares floating point numbers
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
