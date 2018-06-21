import numpy as np
from numpy import linalg as LA
import math
import random
import sys
from utils import *

# Class that computes partial trace in 2, 3 and 4 qubits
# and 2,3 and 4 qutrits systems

# Separates quantum state p by computing its partial trace
# Separated systems are saved in list systems iself.e. pA, pB ...
# systems = []

# Intermediate bi-partite systems are stored
# Order for pABC:  pAB, pBC, pAC
# Order for pABCD: pAB (0), pBC (1), pAC (2), pBD (3), pAD (4), pCD (5)
# Order for pABCDE: pAB, pAC, pAD, pBC, pBD, pAE, pBE, pCD, pCE
# joint_systems = []

# Intermediate  tri-partite systems are stored
# Order for pABCD: pABC (0), pABD (1), pBCD (2), pACD (3)
# Order for pABCDE: pABC (0), pABD (2), pACD (3) ,pBCD, pABE, pACE, pBCE, pADE,
#                   pBDE, pCDE (9)
# joint_systems3 = []

# Intermediate 4-partite systems are stored
# Order for pABCDE = pABCD (0), pABCE (1), pABDE (2), pBCDE (3), pACDE (4)
# joint_sysems4 = []

def separate(p,dim):
    s, j, j3, j4 = partial_trace(p, dim, [], [], [], [])
    return s, j, j3, j4


def partial_trace(p, dim, systems, joint_systems, joint_systems3, joint_systems4):
    """
    Top level function that calls separate_qubit and separate_qutrit
    to get single qubit and qutrit systems
    dim = 2 if qubit, dim = 3 if qutrit
    """
    # Check matrix is square
    check_square_matrix(p, "separate")


    n = p.shape[0]
    n2 = p.shape[1]
    func_str = "separate in partial_trace.py"
    q = check_power_of_dim(n, dim, func_str)

    if(isPowerof2(dim)):
        separate_qubit(p, systems, joint_systems, joint_systems3, joint_systems4)
    else:
        separate_qutrit(p, systems, joint_systems, joint_systems3, joint_systems4)

    return systems, joint_systems, joint_systems3, joint_systems4


def separate_qubit(p, systems, joint_systems, joint_systems3, joint_systems4):
    """
    Function that separates joint pure qubit quantum state p. The density matrix
    width (and length) MUST be written as 2^q where q is the number of qubits
    """

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
        __separate3(p, systems, joint_systems, joint_systems3, joint_systems4)
    elif(q == 4):
        __separate4(p, systems, joint_systems, joint_systems3, joint_systems4)
    elif(q == 5):
        __separate5(p, systems, joint_systems, joint_systems3, joint_systems4)


def __separate3(p, systems, joint_systems, joint_systems3, joint_systems4):
    """
    Private method that calculates partial trace of 3 qubit systems
    """

    # pAB
    pAB = allocSub(p)
    # dimensions of each sub matrix
    sub_dim = pAB.shape[0]

    x_AB = [2*x for x in range(sub_dim)]
    y_AB = [x+(2**0) for x in x_AB]

    # pAC
    pAC = allocSub(p)
    x_AC = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%2) == 0):
            x_AC[i] = x_AC[i-1] + (2 + 1)
        else:
            x_AC[i] = x_AC[i-1] + 1

    y_AC = [x+(2**1) for x in x_AC]

    # pBC
    pBC = allocSub(p)
    x_BC = [x for x in range(sub_dim)]
    y_BC = [x+(2**2) for x in x_BC]

    for i in range(sub_dim):
        for j in range(sub_dim):
            pAB[i,j] = p[x_AB[i],x_AB[j]] + p[y_AB[i],y_AB[j]]
            pBC[i,j] = p[x_BC[i],x_BC[j]] + p[y_BC[i],y_BC[j]]
            pAC[i,j] = p[x_AC[i],x_AC[j]] + p[y_AC[i],y_AC[j]]

    # Storing intermediiate joint systems
    if(not matrixInList(pAB, joint_systems)):
        joint_systems.append(pAB)
    if(not matrixInList(pBC, joint_systems)):
        joint_systems.append(pBC)
    if(not matrixInList(pAC, joint_systems)):
        joint_systems.append(pAC)

    # Separate joint systems into single systems
    separate_qubit(pAB, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qubit(pBC, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qubit(pAC, systems, joint_systems, joint_systems3, joint_systems4)


def __separate4(p, systems, joint_systems, joint_systems3, joint_systems4):
    """
    Private method that calculates separate systems for 4 qubit systems
    """

    # pABC
    pABC = allocSub(p)

    # dimensions of each sub matrix
    sub_dim = pABC.shape[0]

    x_ABC = [2*x for x in range(sub_dim)]
    y_ABC = [x+(2**0) for x in x_ABC]

    # pABD
    pABD = allocSub(p)
    x_ABD = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%2) == 0):
            x_ABD[i] = x_ABD[i-1] + (2 + 1)
        else:
            x_ABD[i] = x_ABD[i-1] + 1

    y_ABD = [x+(2**1) for x in x_ABD]

    # pACD
    pACD = allocSub(p)
    x_ACD = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%4) == 0):
            x_ACD[i] = x_ACD[i-1] + (2*2 + 1)
        else:
            x_ACD[i] = x_ACD[i-1] + 1

    y_ACD = [x+(2**2) for x in x_ACD]

    # pBCD
    pBCD = allocSub(p)
    x_BCD = [x for x in range(sub_dim)]
    y_BCD = [x+(2**3) for x in x_BCD]

    # Fill in matrix pABC, pABD, pBCD and pACD
    for i in range(sub_dim):
        for j in range(sub_dim):
            pABC[i,j] = p[x_ABC[i],x_ABC[j]] + p[y_ABC[i],y_ABC[j]]
            pABD[i,j] = p[x_ABD[i],x_ABD[j]] + p[y_ABD[i],y_ABD[j]]
            pBCD[i,j] = p[x_BCD[i],x_BCD[j]] + p[y_BCD[i],y_BCD[j]]
            pACD[i,j] = p[x_ACD[i],x_ACD[j]] + p[y_ACD[i],y_ACD[j]]

    # Storing intermediiate joint systems
    if(not matrixInList(pABC, joint_systems3)):
        joint_systems3.append(pABC)
    if(not matrixInList(pABD, joint_systems3)):
        joint_systems3.append(pABD)
    if(not matrixInList(pBCD, joint_systems3)):
        joint_systems3.append(pBCD)
    if(not matrixInList(pACD, joint_systems3)):
        joint_systems3.append(pACD)

    # Separate joint systems into single systems
    separate_qubit(pABC, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qubit(pABD, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qubit(pBCD, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qubit(pACD, systems, joint_systems, joint_systems3, joint_systems4)


def __separate5(p, systems, joint_systems, joint_systems3, joint_systems4):
    """
    Private method that calculates separate systems for 5 qubit systems
    """

    # pABCD
    pABCD = allocSub(p)

    # dimensions of each sub matrix
    sub_dim = pABCD.shape[0]

    x_ABCD = [2*x for x in range(sub_dim)]
    y_ABCD = [x+(2**0) for x in x_ABCD]

    # pABCE
    pABCE = allocSub(p)
    x_ABCE = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%2) == 0):
            x_ABCE[i] = x_ABCE[i-1] + (2 + 1)
        else:
            x_ABCE[i] = x_ABCE[i-1] + 1

    y_ABCE = [x+(2**1) for x in x_ABCE]

    # pABDE
    pABDE = allocSub(p)
    x_ABDE = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%4) == 0):
            x_ABDE[i] = x_ABDE[i-1] + (2*2 + 1)
        else:
            x_ABDE[i] = x_ABDE[i-1] + 1

    y_ABDE = [x+(2**2) for x in x_ABDE]

    # pACDE
    pACDE = allocSub(p)
    x_ACDE = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%8) == 0):
            x_ACDE[i] = x_ACDE[i-1] + (4*2 + 1)
        else:
            x_ACDE[i] = x_ACDE[i-1] + 1

    y_ACDE = [x+(2**3) for x in x_ACDE]

    # pBCD
    pBCDE = allocSub(p)
    x_BCDE = [x for x in range(sub_dim)]
    y_BCDE = [x+(2**4) for x in x_BCDE]

    # Fill in matrix pABC, pABD, pBCD and pACD
    for i in range(sub_dim):
        for j in range(sub_dim):
            pABCD[i,j] = p[x_ABCD[i],x_ABCD[j]] + p[y_ABCD[i],y_ABCD[j]]
            pABCE[i,j] = p[x_ABCE[i],x_ABCE[j]] + p[y_ABCE[i],y_ABCE[j]]
            pABDE[i,j] = p[x_ABDE[i],x_ABDE[j]] + p[y_ABDE[i],y_ABDE[j]]
            pBCDE[i,j] = p[x_BCDE[i],x_BCDE[j]] + p[y_BCDE[i],y_BCDE[j]]
            pACDE[i,j] = p[x_ACDE[i],x_ACDE[j]] + p[y_ACDE[i],y_ACDE[j]]

    # Storing intermediiate joint systems
    if(not matrixInList(pABCD, joint_systems4)):
        joint_systems4.append(pABCD)
    if(not matrixInList(pABCE, joint_systems4)):
        joint_systems4.append(pABCE)
    if(not matrixInList(pABDE, joint_systems4)):
        joint_systems4.append(pABDE)
    if(not matrixInList(pBCDE, joint_systems4)):
        joint_systems4.append(pBCDE)
    if(not matrixInList(pACDE, joint_systems4)):
        joint_systems4.append(pACDE)

    # Separate joint systems into single systems
    separate_qubit(pABCD, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qubit(pABCE, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qubit(pABDE, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qubit(pBCDE, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qubit(pACDE, systems, joint_systems, joint_systems3, joint_systems4)


# Separate qutrit <0|, <1|, <2|
def separate_qutrit(p, systems, joint_systems, joint_systems3, joint_systems4):
    """
    Function that separates joint pure qutrit quantum state p. The density
    matrix width (and length) MUST be written as 3^q where q is the number
    of qutrits
    """

    dim = p.shape[0]
    # 3^q = dim, q is the number of qutrits
    q = math.log(dim) / math.log(3)

    # p_AB 2 qutrits
    if(q == 2):
        # pA
        pA = allocSub(p)
        # dimensions of each sub matrix
        sub_dim = pA.shape[0]

        x_A = [3*x for x in range(sub_dim)]
        y_A = [x+1 for x in x_A]
        z_A = [x+1 for x in y_A]

        # pB
        pB = allocSub(p)
        x_B = [x for x in range(sub_dim)]
        y_B = [x+sub_dim for x in x_B]
        z_B = [x+sub_dim for x in y_B]

        # Fill in matrix pA and pB
        for i in range(sub_dim):
            for j in range(sub_dim):
                pA[i,j] = p[x_A[i],x_A[j]] + p[y_A[i],y_A[j]] + p[z_A[i],z_A[j]]
                pB[i,j] = p[x_B[i],x_B[j]] + p[y_B[i],y_B[j]] + p[z_B[i],z_B[j]]

        # Store in list
        if(not matrixInList(pA, systems)):
            systems.append(pA)
        if(not matrixInList(pB, systems)):
            systems.append(pB)

    # p_ABC 3 qutrits
    elif(q == 3):
        __separate3_qutrit(p, systems, joint_systems, joint_systems3, joint_systems4)
    elif(q == 4):
        __separate4_qutrit(p, systems, joint_systems, joint_systems3, joint_systems4)
    elif(q == 5 or isclose(q,5)):
        __separate5_qutrit(p, systems, joint_systems, joint_systems3, joint_systems4)


def __separate3_qutrit(p, systems, joint_systems, joint_systems3, joint_systems4):
    """
    Private method that calculates separate systems for 3 qutrit systems
    """

    #pAB
    pAB = allocSub(p)

    # dimensions of each sub matrix
    sub_dim = pAB.shape[0]
    x_AB = [3*x for x in range(sub_dim)]
    y_AB = [x+(3**0) for x in x_AB]
    z_AB = [x+(3**0) for x in y_AB]

    # pAC
    pAC = allocSub(p)
    x_AC = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%3) == 0):
            x_AC[i] = x_AC[i-1] + (3*2 + 1)
        else:
            x_AC[i] = x_AC[i-1] + 1

    y_AC = [x+(3**1) for x in x_AC]
    z_AC = [x+(3**1) for x in y_AC]

    # pBC
    pBC = allocSub(p)
    x_BC = [x for x in range(sub_dim)]
    y_BC = [x+(3**2) for x in x_BC]
    z_BC = [x+(3**2) for x in y_BC]

    # Fill in matrix pAB, pBC and pAC
    for i in range(sub_dim):
        for j in range(sub_dim):
            pAB[i,j] = p[x_AB[i],x_AB[j]] + p[y_AB[i],y_AB[j]] + p[z_AB[i],z_AB[j]]
            pBC[i,j] = p[x_BC[i],x_BC[j]] + p[y_BC[i],y_BC[j]] + p[z_BC[i],z_BC[j]]
            pAC[i,j] = p[x_AC[i],x_AC[j]] + p[y_AC[i],y_AC[j]] + p[z_AC[i],z_AC[j]]

    # Storing intermediiate joint systems
    if(not matrixInList(pAB, joint_systems)):
        joint_systems.append(pAB)
    if(not matrixInList(pBC, joint_systems)):
        joint_systems.append(pBC)
    if(not matrixInList(pAC, joint_systems)):
        joint_systems.append(pAC)

    # Recursively separate further
    separate_qutrit(pAB, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qutrit(pBC, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qutrit(pAC, systems, joint_systems, joint_systems3, joint_systems4)


def __separate4_qutrit(p, systems, joint_systems, joint_systems3, joint_systems4):
    """
    Private method that calculates separate systems for 4 qutrit systems
    """

    #pABC
    pABC = allocSub(p)
    sub_dim = pABC.shape[0]
    x_ABC = [3*x for x in range(sub_dim)]
    y_ABC = [x+(3**0) for x in x_ABC]
    z_ABC = [x+(3**0) for x in y_ABC]

    #pABD
    pABD = allocSub(p)
    x_ABD = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%3) == 0):
            x_ABD[i] = x_ABD[i-1] + (3*2 + 1)
        else:
            x_ABD[i] = x_ABD[i-1] + 1

    y_ABD = [x+(3**1) for x in x_ABD]
    z_ABD = [x+(3**1) for x in y_ABD]

    #pACD
    pACD = allocSub(p)
    x_ACD = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%9) == 0):
            x_ACD[i] = x_ACD[i-1] + (9*2 + 1)
        else:
            x_ACD[i] = x_ACD[i-1] + 1

    y_ACD = [x+(3**2) for x in x_ACD]
    z_ACD = [x+(3**2) for x in y_ACD]

    #pBCD
    pBCD = allocSub(p)
    x_BCD = [x for x in range(sub_dim)]
    y_BCD = [x+(3**3) for x in x_BCD]
    z_BCD = [x+(3**3) for x in y_BCD]

    # Fill in matrix pABC, pABD, pBCD and pACD
    for i in range(sub_dim):
        for j in range(sub_dim):
            pABC[i,j] = p[x_ABC[i],x_ABC[j]] + p[y_ABC[i],y_ABC[j]] + p[z_ABC[i],z_ABC[j]]
            pABD[i,j] = p[x_ABD[i],x_ABD[j]] + p[y_ABD[i],y_ABD[j]] + p[z_ABD[i],z_ABD[j]]
            pBCD[i,j] = p[x_BCD[i],x_BCD[j]] + p[y_BCD[i],y_BCD[j]] + p[z_BCD[i],z_BCD[j]]
            pACD[i,j] = p[x_ACD[i],x_ACD[j]] + p[y_ACD[i],y_ACD[j]] + p[z_ACD[i],z_ACD[j]]


    # Storing intermediiate joint systems
    if(not matrixInList(pABC, joint_systems3)):
        joint_systems3.append(pABC)
    if(not matrixInList(pABD, joint_systems3)):
        joint_systems3.append(pABD)
    if(not matrixInList(pBCD, joint_systems3)):
        joint_systems3.append(pBCD)
    if(not matrixInList(pACD, joint_systems3)):
        joint_systems3.append(pACD)


    # Recursively separate further
    separate_qutrit(pABC, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qutrit(pABD, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qutrit(pBCD, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qutrit(pACD, systems, joint_systems, joint_systems3, joint_systems4)


def __separate5_qutrit(p, systems, joint_systems, joint_systems3, joint_systems4):
    """
    Private method that calculates separate systems for 5 qutrit systems
    """

    #pABCD
    pABCD = allocSub(p)
    sub_dim = pABCD.shape[0]
    x_ABCD = [3*x for x in range(sub_dim)]
    y_ABCD = [x+(3**0) for x in x_ABCD]
    z_ABCD = [x+(3**0) for x in y_ABCD]

    #pABCE
    pABCE = allocSub(p)
    x_ABCE = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%3) == 0):
            x_ABCE[i] = x_ABCE[i-1] + (3*2 + 1)
        else:
            x_ABCE[i] = x_ABCE[i-1] + 1

    y_ABCE = [x+(3**1) for x in x_ABCE]
    z_ABCE = [x+(3**1) for x in y_ABCE]

    #pABDE
    pABDE = allocSub(p)
    x_ABDE = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%9) == 0):
            x_ABDE[i] = x_ABDE[i-1] + (9*2 + 1)
        else:
            x_ABDE[i] = x_ABDE[i-1] + 1

    y_ABDE = [x+(3**2) for x in x_ABDE]
    z_ABDE = [x+(3**2) for x in y_ABDE]

    #pACDE
    pACDE = allocSub(p)
    x_ACDE  = [0 for x in range(sub_dim)]
    for i in range(1,sub_dim):
        if((i%27) == 0):
            x_ACDE [i] = x_ACDE[i-1] + (27*2 + 1)
        else:
            x_ACDE [i] = x_ACDE[i-1] + 1

    y_ACDE = [x+(3**3) for x in x_ACDE]
    z_ACDE = [x+(3**3) for x in y_ACDE]

    #pBCDE
    pBCDE = allocSub(p)
    x_BCDE = [x for x in range(sub_dim)]
    y_BCDE = [x+(3**4) for x in x_BCDE]
    z_BCDE = [x+(3**4) for x in y_BCDE]

    # Fill in matrix pABCD, pABCE, pABDE, pBCDE and pACDE
    for i in range(sub_dim):
        for j in range(sub_dim):
            pABCD[i,j] = p[x_ABCD[i],x_ABCD[j]] + p[y_ABCD[i],y_ABCD[j]] + p[z_ABCD[i],z_ABCD[j]]
            pABCE[i,j] = p[x_ABCE[i],x_ABCE[j]] + p[y_ABCE[i],y_ABCE[j]] + p[z_ABCE[i],z_ABCE[j]]
            pABDE[i,j] = p[x_ABDE[i],x_ABDE[j]] + p[y_ABDE[i],y_ABDE[j]] + p[z_ABDE[i],z_ABDE[j]]
            pBCDE[i,j] = p[x_BCDE[i],x_BCDE[j]] + p[y_BCDE[i],y_BCDE[j]] + p[z_BCDE[i],z_BCDE[j]]
            pACDE[i,j] = p[x_ACDE[i],x_ACDE[j]] + p[y_ACDE[i],y_ACDE[j]] + p[z_ACDE[i],z_ACDE[j]]

    # Storing intermediiate joint systems
    if(not matrixInList(pABCD, joint_systems4)):
        joint_systems4.append(pABCD)
    if(not matrixInList(pABCE, joint_systems4)):
        joint_systems4.append(pABCE)
    if(not matrixInList(pABDE, joint_systems4)):
        joint_systems4.append(pABDE)
    if(not matrixInList(pBCDE, joint_systems4)):
        joint_systems4.append(pBCDE)
    if(not matrixInList(pACDE, joint_systems4)):
        joint_systems4.append(pACDE)

    # Recursively separate further
    separate_qutrit(pABCD, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qutrit(pABCE, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qutrit(pABDE, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qutrit(pBCDE, systems, joint_systems, joint_systems3, joint_systems4)
    separate_qutrit(pACDE, systems, joint_systems, joint_systems3, joint_systems4)
