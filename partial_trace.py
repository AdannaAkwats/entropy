import numpy as np
from numpy import linalg as LA
import math
import random
import sys
from utils import isMatrixSame
from utils import matrixInList
from utils import allocSub
from utils import isPowerof2
from utils import isPowerof3

# Class that cmputes partial trace in 2, 3 and 4 qubits
# and 2,3 and 4 qutrits systems

# Separates joint pure quantum state p by comouting its partial trace
# Separated Systems are saved in list systems i.e. pA, pB ...
systems = []

# Intermediate joint systems are stored i.e. pAB, pBC, pAC
joint_systems = []
print "sys"

def separate(p):
    """
    Function that separates joint pure qubit quantum state p. The density matrix
    width (and length) MUST be written as 2^q where q is the number of qubits
    """

    dim = p.shape[0]

    if(not isPowerof2(dim)):
        print "Error in Function 'separate in partial_trace.py':"
        print "Density matrix given is not a qubit system."
        print "i.e. Width/Length of matrix is not in form 2^q."
        sys.exit()

    # 2^q = dim, q is the number of qubits
    q = math.log(dim) / math.log(2)

    # p_AB 2 qubits
    if(q == 2):
    #    print "here"
        print "systems"
        print systems
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

    return systems, joint_systems


def __separate3(p):
    """
    Private method that calculates partial trace of 3 qubit systems
    """


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

    # Storing intermediiate joint systems
    # joint_systems.append(pAB)
    # joint_systems.append(pBC)
    # joint_systems.append(pAC)

    # Separate joint systems into single systems
    separate(pAB)
    separate(pBC)
    separate(pAC)


def __separate4(p):
    """
    Private method that calculates separate systems for 4 qubit systems
    """

    # pABC
    pABC = allocSub(p)

    # dimensions of each sub matrix
    sub_dim = pABC.shape[0]

    x_ABC = [2*x for x in range(sub_dim)]
    y_ABC = [2*x + 1 for x in range(sub_dim)]

    # pABD
    pABD = allocSub(p)

    x_ABD = [0 for x in range(sub_dim)]

    for i in range(sub_dim):
        if(i != 0 && ((i%2) == 0)):
            x_ABD[i] = x_ABD[i-1] + (2 + 1)
        else:
            x_ABD[i] = x_ABD[i-1] + 1

    # pBCD
    pBCD = allocSub(p)

    x_BCD = [x for x in range(sub_dim)]
    y_BCD = [x+sub_dim for x in range(sub_dim)]

    # pBCD
    pACD= allocSub(p)

    x_ACD = [0,1,2,4,5,9,10,12]
    y_ACD = [3,6,7,8,11,13,14,15]

    x_ACD = [0 for x in range(sub_dim)]

    for i in range(sub_dim):
        if(i != 0 && ((i%4) == 0)):
            x_ACD[i] = x_ACD[i-1] + (2*2 + 1)
        else:
            x_ACD[i] = x_ACD[i-1] + 1

    for i in range(sub_dim):
        for j in range(sub_dim):
            pABC[i,j] = p[x_ABC[i],x_ABC[j]] + p[y_ABC[i],y_ABC[j]]
            pABD[i,j] = p[x_ABD[i],x_ABD[j]] + p[y_ABD[i],y_ABD[j]]
            pBCD[i,j] = p[x_BCD[i],x_BCD[j]] + p[y_BCD[i],y_BCD[j]]
            pACD[i,j] = p[x_ACD[i],x_ACD[j]] + p[y_ACD[i],y_ACD[j]]

    # Storing intermediiate joint systems
    # joint_systems.append(pABC)
    # joint_systems.append(pABD)
    # joint_systems.append(pBCD)
    # joint_systems.append(pACD)


    # Separate joint systems into single systems
    __separate3(pABC)
    __separate3(pABD)
    __separate3(pBCD)
    __separate3(pACD)



# Separate qutrit <0|, <1|, <2|
def separate_qutrit(p):
    """
    Function that separates joint pure qutrit quantum state p. The density
    matrix width (and length) MUST be written as 3^q where q is the number
    of qutrits
    """

    dim = p.shape[0]

    if(not isPowerof3(dim)):
        print "Error in Function 'separate_qutrit in partial_trace.py':"
        print "Density matrix given is not a qutrit system."
        print "i.e. Width and Length of matrix is not in form 3^q."
        sys.exit()

    # 3^q = dim, q is the number of qutrits
    q = math.log(dim) / math.log(3)

    # p_AB 2 qutrits
    if(q == 2):
        # pA
        pA = allocSub(p)
        # dimensions of each sub matrix
        sub_dim = pA.shape[0]

        x_A = [3*x for x in range(sub_dim)]
        y_A = [3*x + 1 for x in range(sub_dim)]
        z_A = [3*x + 2 for x in range(sub_dim)]

        # pB
        pB = allocSub(p)

        x_B = [x for x in range(sub_dim)]
        y_B = [x+sub_dim for x in range(sub_dim)]
        z_B = [x+2*sub_dim for x in range(sub_dim)]

        for i in range(sub_dim):
            for j in range(sub_dim):
                pA[i,j] = p[x_A[i],x_A[j]] + p[y_A[i],y_A[j]] + p[z_A[i],z_A[j]]
                pB[i,j] = p[x_B[i],x_B[j]] + p[y_B[i],y_B[j]] + p[z_B[i],z_B[j]]

        if(not matrixInList(pA, systems)):
            systems.append(pA)
        if(not matrixInList(pB, systems)):
            systems.append(pB)

    # p_ABC 3 qutrits
    elif(q == 3):
        __separate3_qutrit(p)
    elif(q == 4):
        __separate4_qutrit(p)

    return systems


def __separate3_qutrit(p):
    """
    Private method that calculates separate systems for 3 qutrit systems
    """

    #pAB
    pAB = allocSub(p)

    # dimensions of each sub matrix
    sub_dim = pAB.shape[0]
    x_AB = [3*x for x in range(sub_dim)]
    y_AB = [3*x + 1 for x in range(sub_dim)]
    z_AB = [3*x + 2 for x in range(sub_dim)]

    # pBC
    pBC = allocSub(p)

    x_BC = [x for x in range(sub_dim)]
    y_BC = [x+sub_dim for x in range(sub_dim)]
    z_BC = [x+2*sub_dim for x in range(sub_dim)]

    # pAC
    pAC = allocSub(p)

    x_AC = [0 for x in range(sub_dim)]

    for i in range(sub_dim):
        if(i != 0 && ((i%3) == 0)):
            x_AC[i] = x_AC[i-1] + (3*2 + 1)
        else:
            x_AC[i] = x_AC[i-1] + 1

    y_AC = [x+3 for x in x_AC]
    z_AC = [x+3 for x in y_AC]

    for i in range(sub_dim):
        for j in range(sub_dim):
            pAB[i,j] = p[x_AB[i],x_AB[j]] + p[y_AB[i],y_AB[j]] + p[z_AB[i],z_AB[j]]
            pBC[i,j] = p[x_BC[i],x_BC[j]] + p[y_BC[i],y_BC[j]] + p[z_BC[i],z_BC[j]]
            pAC[i,j] = p[x_AC[i],x_AC[j]] + p[y_AC[i],y_AC[j]] + p[z_AC[i],z_AC[j]]

    separate(pAB)
    separate(pBC)
    separate(pAC)


def __separate4_qutrit(p):
    """
    Private method that calculates separate systems for 4 qutrit systems
    """

    #pABC
    pABC = allocSub(p)
    sub_dim = pABC.shape[0]
    x_ABC = [3*x for x in range(sub_dim)]
    y_ABC = [3*x + 1 for x in range(sub_dim)]
    z_ABC = [3*x + 2 for x in range(sub_dim)]

    #pABD
    pABD = allocSub(p)
    x_ABD = [0 for x in range(sub_dim)]

    for i in range(sub_dim):
        if(i != 0 && ((i%3) == 0)):
            x_ABD[i] = x_ABD[i-1] + (3*2 + 1)
        else:
            x_ABD[i] = x_ABD[i-1] + 1

    y_ABD = [x+3 for x in x_ABD]
    z_ABD = [x+3 for x in y_ABD]


    #pBCD
    pBCD = allocSub(p)
    x_BCD = [x for x in range(sub_dim)]
    y_BCD = [x+sub_dim for x in range(sub_dim)]
    z_BCD = [x+2*sub_dim for x in range(sub_dim)]

    #pACD
    pACD = allocSub(p)
    x_ACD = [0 for x in range(sub_dim)]

    for i in range(sub_dim):
        if(i != 0 && ((i%9) == 0)):
            x_ABD[i] = x_ABD[i-1] + (9*2 + 1)
        else:
            x_ABD[i] = x_ABD[i-1] + 1

    y_ACD = [x+9 for x in x_ACD]
    z_ACD = [x+9 for x in y_ACD]
