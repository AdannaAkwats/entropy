import numpy as np
from numpy import linalg as LA
import math
import random
from utils import isMatrixSame
from utils import matrixInList

# Class that cmputes partial trace in 2, 3 and 4 qubits
# and 2,3 and 4 qutrits systems

# Computes partial trace of density matrix (pure quantum system) p
# Systems are saved in list systems
systems = []

def separate(p):
    dim = p.shape[0]

    # 2^q = dim, q is the number of qubits
    q = math.log(dim) / math.log(2)

    # p_AB 2 qubits
    if(q == 2):
        # pA
        temp = np.zeros((2,2))
        pA = np.matrix(temp, dtype=np.complex128)

        pA[0,0] = p[0,0] + p[1,1]
        pA[0,1] = p[0,2] + p[1,3]
        pA[1,0] = p[2,0] + p[3,1]
        pA[1,1] = p[2,2] + p[3,3]

        # pB
        pB = np.matrix(temp, dtype=np.complex128)
        pB[0,0] = p[0,0] + p[2,2]
        pB[0,1] = p[0,1] + p[2,3]
        pB[1,0] = p[1,0] + p[3,2]
        pB[1,1] = p[1,1] + p[3,3]


        if(not matrixInList(pA, systems)):
            systems.append(pA)
        if(not matrixInList(pB, systems)):
            systems.append(pB)

    # p_ABC 3 qubits
    elif(q == 3):
        __separate3(p)
    elif(q == 4):
        __separate4(p)

    return systems


# Private method that calculates partial trace of 3 qubit systems
def __separate3(p):

    half_dim = p.shape[0] / 2

    # pAB
    temp = np.zeros((half_dim, half_dim))
    pAB = np.matrix(temp, dtype=np.complex128)

    pAB[0,0] = p[0,0] + p[1,1]
    pAB[0,1] = p[0,2] + p[1,4]
    pAB[0,2] = p[0,3] + p[1,5]
    pAB[0,3] = p[0,6] + p[1,7]

    pAB[1,0] = p[2,0] + p[4,1]
    pAB[1,1] = p[2,2] + p[4,4]
    pAB[1,2] = p[2,3] + p[4,5]
    pAB[1,3] = p[2,6] + p[4,7]

    pAB[2,0] = p[3,0] + p[5,1]
    pAB[2,1] = p[3,2] + p[5,4]
    pAB[2,2] = p[3,3] + p[5,5]
    pAB[2,3] = p[3,6] + p[5,7]

    pAB[3,0] = p[6,0] + p[7,1]
    pAB[3,1] = p[6,2] + p[7,4]
    pAB[3,2] = p[6,3] + p[7,5]
    pAB[3,3] = p[6,6] + p[7,7]

    # pBC TODO (use for validation for B?)
    temp = np.zeros((half_dim, half_dim))
    pBC = np.matrix(temp, dtype=np.complex128)

    pBC[0,0] = p[0,0] + p[3,3]
    pBC[0,1] = p[0,1] + p[3,5]
    pBC[0,2] = p[0,2] + p[3,6]
    pBC[0,3] = p[0,4] + p[3,7]

    pBC[1,0] = p[1,0] + p[5,3]
    pBC[1,1] = p[1,1] + p[5,5]
    pBC[1,2] = p[1,2] + p[5,6]
    pBC[1,3] = p[1,4] + p[5,7]

    pBC[2,0] = p[2,0] + p[6,3]
    pBC[2,1] = p[2,1] + p[6,5]
    pBC[2,2] = p[2,2] + p[6,6]
    pBC[2,3] = p[2,4] + p[6,7]

    pBC[3,0] = p[4,0] + p[7,3]
    pBC[3,1] = p[4,1] + p[7,5]
    pBC[3,2] = p[4,2] + p[7,6]
    pBC[3,3] = p[4,4] + p[7,7]

    # pAC TODO (use for validation?)
    temp = np.zeros((half_dim, half_dim))
    pAC = np.matrix(temp, dtype=np.complex128)

    pAC[0,0] = p[0,0] + p[2,2]
    pAC[0,1] = p[0,1] + p[2,4]
    pAC[0,2] = p[0,3] + p[2,6]
    pAC[0,3] = p[0,5] + p[2,7]

    pAC[1,0] = p[1,0] + p[4,2]
    pAC[1,1] = p[1,1] + p[4,4]
    pAC[1,2] = p[1,3] + p[4,6]
    pAC[1,3] = p[1,5] + p[4,7]

    pAC[2,0] = p[3,0] + p[6,2]
    pAC[2,1] = p[3,1] + p[6,4]
    pAC[2,2] = p[3,3] + p[6,6]
    pAC[2,3] = p[3,5] + p[6,7]

    pAC[3,0] = p[5,0] + p[7,2]
    pAC[3,1] = p[5,1] + p[7,4]
    pAC[3,2] = p[5,3] + p[7,6]
    pAC[3,3] = p[5,5] + p[7,7]

    separate(pAB)

    separate(pBC)

    separate(pAC)

# Private method that calculates separate systems for 4 qubit systems
def __separate4(p):

    # pABC
    pABC = np.zeros((8,8))

    # pABD TODO (use for validation for A and B?)
    pABD = np.zeros((8,8))

    # pBCD TODO (use for validation?)
    pBCD= np.zeros((8,8))

    # pBCD TODO (use for validation?)
    pACD= np.zeros((8,8))

    __separate3(pABC)

    __separate3(pABD)
