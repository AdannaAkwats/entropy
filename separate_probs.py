import math
import numpy as np
from utils import *
import sys

# Separates joint propability distribution
# Separated systems are saved in list systems iself.e. pA, pB ...
systems = []

# Intermediate joint distributions are stored
# Order for pABC:  pAB, pBC, pAC
# Order for pABCD: pAB (0), pBC (1), pAC (2), pBD (3), pAD (4), pCD (5)
joint_systems = []

# Intermediate joint distributions are stored
# Order for pABCD: pABC (0), pABD (1), pBCD (2), pACD (3)
joint_systems3 = []



def remove_dups_list(List):
    seen = []
    for l in List:
        if(not listInList(l, seen)):
            seen.append(l)
    return seen


def separate_probs(p):
    """
    Separate joint probability distribution p into marginal and smaller
    joint probabilities
    """

    q = len(p) ** (1. / 4)
    c = len(p) ** (1. / 3)
    s = len(p) ** (1. / 2)

    if(q.is_integer()):
        __separate_4(p)
    elif(c.is_integer()):
        __separate_3(p)
    elif(s.is_integer()):
        __separate_2(p)
    else:
        print "Error in Function 'separate_probs' in separate_probs.py':"
        print "Probability list length is not a square, cube or to the 4th power"
        sys.exit()

    s = remove_dups_list(systems)
    j = remove_dups_list(joint_systems)
    js = remove_dups_list(joint_systems3)
    return s, j, js


def __separate_2(pAB):
    """
    Separate list pAB to pA and pB
    """
    # length of pA, pB square root of the length pAB
    n = len(pAB) ** (1. / 2)
    if(not n.is_integer()):
        print "Error in Function 'separate_2' in separate_probs.py':"
        print "Probability list length is not a square"
        sys.exit()

    n = int(n)
    pA = np.zeros(n)
    pB = np.zeros(n)

    k = 0
    for i in range(n):
        indices = range(i, i+n*n, n)
        pB[i] = pAB[indices].sum()

        if((k % n) == 1):
            k += (n-1)
        indices = range(k, k+n)
        pA[i] = pAB[indices].sum()
        k += 1

        # Store in list
        if(not listInList(pA, systems)):
            systems.append(pA)
        if(not listInList(pB, systems)):
            systems.append(pB)


def __separate_3(pABC):
    """
    Separate list pABC to get pAB, pBC and pAC
    """

    # length of pAB is a cube
    # length of pAB, pBC, pAC is the square of the cube root of pABC
    n = len(pABC) ** (1. / 3)
    if(not n.is_integer()):
        print "Error in Function 'separate_3' in separate_probs.py':"
        print "Probability list length is not a cube"
        sys.exit()

    lp = int(n ** 2)
    pAB = np.zeros(lp)
    pAC = np.zeros(lp)
    pBC = np.zeros(lp)

    n = int(n)

    j = 0
    k = 0
    for i in range(lp):
        indices = range(i, i+n*lp, lp)
        pBC[i] = pABC[indices].sum() # pABC[i] + pABC[i + lp] + pABC[i + 2*lp]

        if((j%lp) == n): # j == (n%lp)
            j += (lp-n)
        indices = range(j, j+n*n, n)
        pAC[i] = pABC[indices].sum() # pABC[j] + pABC[j + n] + pABC[j + 2*n]
        j += 1

        if((k % n) == 1):#k == (1%n)):
            k += (n-1)
        indices = range(k, k+n)
        pAB[i] = pABC[indices].sum() #pABC[k] + pABC[k + 1] + pABC[k + 2]
        k += 1

    # Storing intermediiate joint systems
    if(not listInList(pAB, joint_systems)):
        joint_systems.append(pAB)
    if(not listInList(pBC, joint_systems)):
        joint_systems.append(pBC)
    if(not listInList(pAC, joint_systems)):
        joint_systems.append(pAC)

    # Recursively separate further
    separate_probs(pAB)
    separate_probs(pBC)
    separate_probs(pAC)


def __separate_4(pABCD):
    """
    Separate list pABCD to get pABC, pBCD, pACD and pABD
    """

    # length of pABC is a to the poDer of 4
    # length ofpABC, pBCD, pACD and pABD is the cube of the 4th  root of pABCD
    n = len(pABCD) ** (1. / 4)
    if(not n.is_integer()):
        print "Error in Function 'separate_4' in separate_probs.py':"
        print "Probability list length is not written to the 4th power"
        sys.exit()

    lp = int(n ** 3)
    ln = int(n ** 2)

    pABC = np.zeros(lp)
    pABD = np.zeros(lp)
    pBCD = np.zeros(lp)
    pACD = np.zeros(lp)

    n = int(n)

    j = 0
    k = 0
    m = 0
    for i in range(lp):
        indices = range(i, i+n*lp, lp)
        pBCD[i] = pABCD[indices].sum()

        if((j % lp) == ln):
            j += (lp-ln)
        indices = range(j, j+n*ln, ln)
        pACD[i] = pABCD[indices].sum()
        j += 1

        if((m % ln) == n):
            m += (ln-n)
        indices = range(m, m+n*n, n)
        pABD[i] = pABCD[indices].sum()
        m += 1

        if((k % n) == 1):
            k += (n-1)
        indices = range(k, k+n)
        pABC[i] = pABCD[indices].sum()
        k += 1


    # Storing intermediiate joint systems
    if(not listInList(pABC, joint_systems3)):
        joint_systems3.append(pABC)
    if(not listInList(pABD, joint_systems3)):
        joint_systems3.append(pABD)
    if(not listInList(pBCD, joint_systems3)):
        joint_systems3.append(pBCD)
    if(not listInList(pACD, joint_systems3)):
        joint_systems3.append(pACD)

    # Recursively separate further
    separate_probs(pABC)
    separate_probs(pABD)
    separate_probs(pBCD)
    separate_probs(pACD)
