import numpy as np
from numpy import linalg as LA
import math
import random
from utils import isMatrixSame
from utils import matrixInList
from utils import allocSub

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
        pA = allocSub(p)
        # dimensions of each sub matrix
        sub_dim = pA.shape[0]
        x_A = [0,2]
        y_A = [1,3]

        # pB
        pB = allocSub(p)

        x_B = [0,1]
        y_B = [2,3]

        for i in range(sub_dim):
            for j in range(sub_dim):
                pA[i,j] = p[x_A[i],x_A[j]] + p[y_A[i],y_A[j]]
                pB[i,j] = p[x_B[i],x_B[j]] + p[y_B[i],y_B[j]]

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

    # pAB
    pAB = allocSub(p)
    # dimensions of each sub matrix
    sub_dim = pAB.shape[0]

    x_AB = [0,2,3,6]
    y_AB = [1,4,5,7]

    # pBC
    pBC = allocSub(p)

    x_BC = [0,1,2,4]
    y_BC = [3,5,6,7]

    # pAC
    pAC = allocSub(p)

    x_AC = [0,1,3,5]
    y_AC = [2,4,6,7]

    for i in range(sub_dim):
        for j in range(sub_dim):
            pAB[i,j] = p[x_AB[i],x_AB[j]] + p[y_AB[i],y_AB[j]]
            pBC[i,j] = p[x_BC[i],x_BC[j]] + p[y_BC[i],y_BC[j]]
            pAC[i,j] = p[x_AC[i],x_AC[j]] + p[y_AC[i],y_AC[j]]


    separate(pAB)
    separate(pBC)
    separate(pAC)

# Private method that calculates separate systems for 4 qubit systems
def __separate4(p):

    # pABC
    pABC = allocSub(p)

    # dimensions of each sub matrix
    sub_dim = pABC.shape[0]

    x_ABC = [0,2,3,4,7,10,8,14]
    y_ABC = [1,5,6,9,11,12,13,15]

    # pABD
    pABD = allocSub(p)

    x_ABD = [0,1,3,4,6,9,8,13]
    y_ABD = [2,5,7,10,11,12,14,15]

    # pBCD
    pBCD = allocSub(p)

    x_BCD = [0,1,2,3,5,6,7,11]
    y_BCD = [4,9,10,8,12,13,14,15]

    # pBCD
    pACD= allocSub(p)

    x_ACD = [0,1,2,4,5,9,10,12]
    y_ACD = [3,6,7,8,11,13,14,15]

    for i in range(sub_dim):
        for j in range(sub_dim):
            pABC[i,j] = p[x_ABC[i],x_ABC[j]] + p[y_ABC[i],y_ABC[j]]
            pABD[i,j] = p[x_ABD[i],x_ABD[j]] + p[y_ABD[i],y_ABD[j]]
            pBCD[i,j] = p[x_BCD[i],x_BCD[j]] + p[y_BCD[i],y_BCD[j]]
            pACD[i,j] = p[x_ACD[i],x_ACD[j]] + p[y_ACD[i],y_ACD[j]]


    __separate3(pABC)
    __separate3(pABD)
    __separate3(pBCD)
    __separate3(pACD)



# Separate qutrit <0|, <1|, <2|
def separate_qutrit(p):

    dim = p.shape[0]

    # 3^q = dim, q is the number of qutrits
    q = math.log(dim) / math.log(3)

    # p_AB 2 qutrits
    if(q == 2):
        # pA
        pA = allocSub(p)
        # dimensions of each sub matrix
        sub_dim = pA.shape[0]
        x_A = [0,3,6]
        y_A = [1,4,7]
        z_A = [2,5,8]

        # pB
        pB = allocSub(p)

        x_B = [0,1,2]
        y_B = [3,4,5]
        z_B = [6,7,8]

        for i in range(sub_dim):
            for j in range(sub_dim):
                pA[i,j] = p[x_A[i],x_A[j]] + p[y_A[i],y_A[j]] + p[z_A[i],z_A[j]]
                pB[i,j] = p[x_B[i],x_B[j]] + p[y_B[i],y_B[j]] + p[z_B[i],z_B[j]]

        if(not matrixInList(pA, systems)):
            systems.append(pA)
        if(not matrixInList(pB, systems)):
            systems.append(pB)

    # p_ABC 3 qutrits
    # elif(q == 3):
    #     __separate3_qutrit(p)
    # elif(q == 4):
    #     __separate4_qutrit(p)

    return systems
