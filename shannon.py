import math
import numpy as np
from numpy import linalg as LA
import random
from utils import *
from separate_probs import *

def shannon(probs):
    """
    Returns Shannon entropy H(x) = -sum(P(x)log(P(x)))
    """
    # Check that probabilities given add up to one and are > 0 and < 1
    checkSum = np.sum(probs)

    if(np.any(probs > 1) or np.any(probs < 0)) : # probabilities are between 0 and 1
        print("Error in Function 'shannon in shannon.py':")
        print("Error: Probabilities are not > 1 or < 0")
        sys.exit()

    if(not isclose(checkSum,1)) :
        print("Error in Function 'shannon in shannon.py':")
        print("Error: Probabilities do not add to one")
        sys.exit()

    # If probabilities are valid,
    # 2 represents bits
    v = probs*np.log2(probs)
    return -np.sum(v)

def binary_entropy(p):
    """
    Returns binary entropy H(p) = -plogp - (1-p)log(1-p)
    """
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def randomProbabilityDist(n):
    """
    Generate a random probability distribution of n numbers - from continous
    uniform distribution
    """
    probs = np.random.random(n)
    probs /= probs.sum()
    return probs


def subadditivity(Pxy):
    """
    Returns true if H(X,Y) <= H(X) + H(Y)
    """
    s, _, _ = separate_probs(Pxy)
    px, py = s[0], s[1]
    H_X = shannon(px)
    H_Y = shannon(py)
    H_XY = shannon(Pxy)
    return H_XY <= H_X + H_Y


# def shannon_leq_log(px):
#     """
#     Returns true if H(X) <= log|X|
#     """
#     H_X = shannon(px)
#     print(H_X)
#     abs_X = vector_abs(px)
#     l = np.log2(abs_X)
#     print(l)
#
#     return H_X <= l


def conditionXY(Pxy):
    """
    Returns entropy of X conditional on knowing Y: H(X|Y)
    """

    s, _, _ = separate_probs(Pxy)
    px, py = s[0], s[1]
    H_X = shannon(px)
    H_Y = shannon(py)
    H_XY = shannon(Pxy)

    return H_XY - H_Y

def mutual_information_s(Pxy):
    """
    Returns the mutual information content of X and Y: I(X:Y)
    """

    s, _, _ = separate_probs(Pxy)
    px, py = s[0], s[1]
    H_X = shannon(px)
    H_Y = shannon(py)
    H_XY = shannon(Pxy)

    return H_X + H_Y - H_XY


def mutualInfo_leq_HY(Pxy):
    """
    Returns true if I(X:Y) <= H(Y)
    """
    I_XY = mutual_information_s(Pxy)
    s, _, _ = separate_probs(Pxy)
    _, py = s[0], s[1]
    H_Y = shannon(py)
    return I_XY <= H_Y


def mutualInfo_leqMin(Pxy):
    """
    Returns true if I(X:Y) <= min(H(X),H(Y))
    """

    I_XY = mutual_information_s(Pxy)
    s, _, _ = separate_probs(Pxy)
    px, py = s[0], s[1]
    H_X = shannon(px)
    H_Y = shannon(py)

    return I_XY <= np.minimum(H_X,H_Y)


def mutualInfo_leq_log(Pxy):
    """
    Returns true if I(X:Y) <= log|X| and log|Y|
    """

    I_XY = mutual_information_s(Pxy)
    s, _, _ = separate_probs(Pxy)
    px, py = s[0], s[1]

    upper_x = -np.log2(vector_abs(px))
    upper_y = -np.log2(vector_abs(py))

    return (I_XY <= upper_x) and (I_XY <= upper_y)


def cond_leq_HY(Pxy):
    """
    Returns true if H(X|Y) <= H(X)
    """

    cond_XY = conditionXY(Pxy)
    s, _, _ = separate_probs(Pxy)
    px, _ = s[0], s[1]
    H_X = shannon(px)

    return cond_XY <= H_X


def HXY_geq_max(Pxy) :
    """
    Returns true if H(X,Y) >= max[H(x), H(Y)]
    """
    s, _, _ = separate_probs(Pxy)
    px, py = s[0], s[1]
    H_X = shannon(px)
    H_Y = shannon(py)
    H_XY = shannon(Pxy)

    return H_XY >= max(H_X, H_Y)


def strongSubadditivity(Pxyz):
    """
    Returns true if H(X,Y,Z) + H(Y) <= H(X,Y) + H(Y,Z)
    """

    s, js, _ = separate_probs(Pxyz)
    py = s[1]
    pxy, pyz = js[0], js[1]

    H_XYZ = shannon(Pxyz)
    H_XY = shannon(pxy)
    H_YZ = shannon(pyz)
    H_Y = shannon(py)

    return H_XYZ + H_Y <= H_XY + H_YZ


def cond_mutual_information_s(pAC, pC, pABC, pBC):
    """
    calculates conditional mutual information
    I(A:B|C) = H(A,C) - H(C) - H(A,B,C) + H(B,C)
    """

    H_AC = shannon(pAC)
    H_C = shannon(pC)
    H_ABC = shannon(pABC)
    H_BC = shannon(pBC)

    return H_AC + H_BC - H_C - H_ABC


def and_mutual_information_s(pABC):
    """
    calculates I(A:B,C) = H(A) + H(B,C) - H(A,B,C)
    """

    # Ensure that length of pABC is a cube
    check_power(len(pABC), 3, "and_mutual_information_s in shannon.py")

    s,j,_ = separate_probs(pABC)

    pA = s[0]
    pBC = j[1]

    H_BC = shannon(pBC)
    H_A = shannon(pA)
    H_ABC = shannon(pABC)

    return H_A + H_BC - H_ABC


def print_seps(p):
    """
    Prints all elements of separate_probs
    """
    s, j, j3 = separate_probs(p)

    print("s")
    for i in s:
        print(i)
    print("")
    print("j")
    for i in j:
        print(i)
    print("")
    print("j3")
    for i in j3:
        print(i)


# Non shannon-type entropies from paper
# http://www.cnd.mcgill.ca/~ivan/it_ineq_script/Raymond%20Yeung%20papers/04035957.pdf
# Theorem II.2
def new_eq1_s(pABCD):
    """
    Returns true if:
    2I(C:D) <= I(A:B) + I(A:C,D) + 3I(C:D|A) + I(C:D|B)
    """

    # Ensure that length of pABCD is to the 4th power
    check_power(len(pABCD), 4, "new_eq1_s in shannon.py")

    s,j,j3 = separate_probs(pABCD)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_C_D = mutual_information_s(pCD)        # I(C:D)
    I_A_B = mutual_information_s(pAB)        # I(A:B)
    I_ACD = and_mutual_information_s(pACD)   # I(A:C,D)

    I_CD_A = cond_mutual_information_s(pAC, pA, pACD, pAD) # I(C:D|A)
    I_CD_B = cond_mutual_information_s(pBC, pB, pBCD, pBD) # I(C:D|B)

    return 2*I_C_D <= (I_A_B + I_ACD + 3*I_CD_A + I_CD_B)


# Theorem III.1
def new_eq2_s(pABCD):
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 3I(A:C|B) + 3I(B:C|A) + 2I(A:D) +2I(B:C|D)
    """

    # Ensure that length of pABCD is to the 4th power
    check_power(len(pABCD), 4, "new_eq2_s in shannon.py")

    s,j,j3 = separate_probs(pABCD)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information_s(pAB)        # I(A:B)
    I_A_D = mutual_information_s(pAD)        # I(A:D)

    I_AB_C = cond_mutual_information_s(pAC, pC, pABC, pBC) # I(A:B|C)
    I_AC_B = cond_mutual_information_s(pAB, pB, pABC, pBC) # I(A:D|B)
    I_BC_A = cond_mutual_information_s(pAB, pA, pABC, pAC) # I(B:C|A)
    I_BC_D = cond_mutual_information_s(pBD, pD, pBCD, pCD) # I(B:C|D)

    return 2*I_A_B <= 3*I_AB_C + 3*I_AC_B + 3*I_BC_A + 2*I_A_D + 2*I_BC_D


# Theorem III.2
def new_eq3_s(pABCD):
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + I(A:C|B) + 2I(B:C|A) + 3I(A:B|D) + I(B:D|A) + 2I(C:D)
    """

    # Ensure that length of pABCD is to the 4th power
    check_power(len(pABCD), 4, "new_eq3_s in shannon.py")

    s,j,j3 = separate_probs(pABCD)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information_s(pAB)        # I(A:B)
    I_C_D = mutual_information_s(pCD)        # I(C:D)

    I_AB_C = cond_mutual_information_s(pAC, pC, pABC, pBC) # I(A:B|C)
    I_AC_B = cond_mutual_information_s(pAB, pB, pABC, pBC) # I(A:C|B)
    I_BC_A = cond_mutual_information_s(pAB, pA, pABC, pAC) # I(B:C|A)
    I_AB_D = cond_mutual_information_s(pAD, pD, pABD, pBD) # I(A:B|D)
    I_BD_A = cond_mutual_information_s(pAB, pA, pABD, pAD) # I(B:D|A)

    return 2*I_A_B <= 4*I_AB_C + I_AC_B + 2*I_BC_A +3*I_AB_D + I_BD_A + 2*I_C_D


# Theorem III.3
def new_eq4_s(pABCD):
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 4I(B:C|A) + 2I(A:C|D) + I(A:D|C) +
    2I(B:D) + I(C:D|A)
    """

    # Ensure that length of pABCD is to the 4th power
    check_power(len(pABCD), 4, "new_eq4_s in shannon.py")

    s,j,j3 = separate_probs(pABCD)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information_s(pAB)        # I(A:B)
    I_B_D = mutual_information_s(pBD)        # I(B:D)

    I_AB_C = cond_mutual_information_s(pAC, pC, pABC, pBC) # I(A:B|C)
    I_AC_B = cond_mutual_information_s(pAB, pB, pABC, pBC) # I(A:C|B)
    I_BC_A = cond_mutual_information_s(pAB, pA, pABC, pAC) # I(B:C|A)
    I_AC_D = cond_mutual_information_s(pAD, pD, pACD, pCD) # I(A:C|D)
    I_AD_C = cond_mutual_information_s(pAC, pC, pACD, pCD) # I(A:D|C)
    I_CD_A = cond_mutual_information_s(pAC, pA, pACD, pAD) #(C:D|A)

    return 2*I_A_B <= 3*I_AB_C + 2*I_AC_B + 4*I_BC_A + 2*I_AC_D + I_AD_C + 2*I_B_D + I_CD_A

# Theorem III.4
def new_eq5_s(pABCD):
    """
    Returns true if:
    2I(A:B) <= 5I(A:B|C) + 3I(A:C|B) + I(B:C|A) + 2I(A:D) + 2I(B:C|D)
    """

    # Ensure that length of pABCD is to the 4th power
    check_power(len(pABCD), 4, "new_eq5_s in shannon.py")

    s,j,j3 = separate_probs(pABCD)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information_s(pAB)        # I(A:B)
    I_A_D = mutual_information_s(pAD)        # I(A:D)

    I_AB_C = cond_mutual_information_s(pAC, pC, pABC, pBC) # I(A:B|C)
    I_AC_B = cond_mutual_information_s(pAB, pB, pABC, pBC) # I(A:C|B)
    I_BC_A = cond_mutual_information_s(pAB, pA, pABC, pAC) # I(B:C|A)
    I_BC_D = cond_mutual_information_s(pBD, pD, pBCD, pCD) # I(B:C|D)

    return 2*I_A_B <= 5*I_AB_C + 3*I_AC_B + I_BC_A + 2*I_A_D + 2*I_BC_D

# Theorem III.5
def new_eq6_s(pABCD):
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + 4I(A:C|B) + I(B:C|A) + 2I(A:D) + 3I(B:C|D) + I(C:D|B)
    """

    # Ensure that length of pABCD is to the 4th power
    check_power(len(pABCD), 4, "new_eq6_s in shannon.py")

    s,j,j3 = separate_probs(pABCD)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information_s(pAB)        # I(A:B)
    I_A_D = mutual_information_s(pAD)        # I(A:D)

    I_AB_C = cond_mutual_information_s(pAC, pC, pABC, pBC) # I(A:B|C)
    I_AC_B = cond_mutual_information_s(pAB, pB, pABC, pBC) # I(A:C|B)
    I_BC_A = cond_mutual_information_s(pAB, pA, pABC, pAC) # I(B:C|A)
    I_BC_D = cond_mutual_information_s(pBD, pD, pBCD, pCD) # I(B:C|D)
    I_CD_B = cond_mutual_information_s(pBC, pB, pBCD, pBD) # I(C:D|B)

    return 2*I_A_B <= 4*I_AB_C + 4*I_AC_B + I_BC_A + 2*I_A_D + 2*I_BC_D + I_CD_B

# Theorem III.6
def new_eq7_s(pABCD):
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 2I(B:C|A) + 2I(A:B|D) + I(A:D|B) + I(B:D|A) + 2I(C:D)
    """

    # Ensure that length of pABCD is to the 4th power
    check_power(len(pABCD), 4, "new_eq7_s in shannon.py")

    s,j,j3 = separate_probs(pABCD)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information_s(pAB)        # I(A:B)
    I_C_D = mutual_information_s(pCD)        # I(C:D)

    I_AB_C = cond_mutual_information_s(pAC, pC, pABC, pBC) # I(A:B|C)
    I_AC_B = cond_mutual_information_s(pAB, pB, pABC, pBC) # I(A:C|B)
    I_BC_A = cond_mutual_information_s(pAB, pA, pABC, pAC) # I(B:C|A)
    I_AB_D = cond_mutual_information_s(pAD, pD, pABD, pBD) # I(A:B|D)
    I_AD_B = cond_mutual_information_s(pAB, pB, pABD, pBD) # I(A:D|B)
    I_BD_A = cond_mutual_information_s(pAB, pA, pABD, pAD) # I(B:D|A)

    return 2*I_A_B <= 3*I_AB_C + 2*I_AC_B + 2*I_BC_A + 2*I_AB_D + I_AD_B + I_BD_A + 2*I_C_D


def non_shannon_eqs(pABCD, eq_no):
    """
    Returns true if all the non shannon type inequalities hold.
    """

    # Ensure that length of pABCD is to the 4th power
    check_power(len(pABCD), 4, "non_shannon_eqs in shannon.py")

    s,j,j3 = separate_probs(pABCD)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    result = []
    res = False

    I_A_B = mutual_information_s(pAB)        # I(A:B)
    I_A_D = mutual_information_s(pAD)        # I(A:D)
    I_B_D = mutual_information_s(pBD)        # I(B:D)
    I_C_D = mutual_information_s(pCD)        # I(C:D)

    I_ACD = and_mutual_information_s(pACD)   # I(A:C,D)

    I_AB_C = cond_mutual_information_s(pAC, pC, pABC, pBC) # I(A:B|C)
    I_AB_D = cond_mutual_information_s(pAD, pD, pABD, pBD) # I(A:B|D)

    I_AC_B = cond_mutual_information_s(pAB, pB, pABC, pBC) # I(A:C|B)
    I_AC_D = cond_mutual_information_s(pAD, pD, pACD, pCD) # I(A:C|D)

    I_AD_B = cond_mutual_information_s(pAB, pB, pABD, pBD) # I(A:D|B)
    I_AD_C = cond_mutual_information_s(pAC, pC, pACD, pCD) # I(A:D|C)

    I_BC_A = cond_mutual_information_s(pAB, pA, pABC, pAC) # I(B:C|A)
    I_BC_D = cond_mutual_information_s(pBD, pD, pBCD, pCD) # I(B:C|D)

    I_BD_A = cond_mutual_information_s(pAB, pA, pABD, pAD) # I(B:D|A)

    I_CD_A = cond_mutual_information_s(pAC, pA, pACD, pAD) # I(C:D|A)
    I_CD_B = cond_mutual_information_s(pBC, pB, pBCD, pBD) # I(C:D|B)


    # 2I(C:D) <= I(A:B) + I(A:C,D) + 3I(C:D|A) + I(C:D|B)
    if(eq_no == 0 or eq_no == 1):
        #res = 2*I_C_D <= I_A_B + I_ACD + 3*I_CD_A + I_CD_B
        res = 2*I_A_B <= I_C_D + I_ACD + 3*I_AB_C + I_AB_D
        result.append(res)

    # 2I(A:B) <= 3I(A:B|C) + 3I(A:C|B) + 3I(B:C|A) + 2I(A:D) +2I(B:C|D)
    if(eq_no == 0 or eq_no == 2):
        res = 2*I_A_B <= 3*I_AB_C + 3*I_AC_B + 3*I_BC_A + 2*I_A_D + 2*I_BC_D
        result.append(res)

    # 2I(A:B) <= 4I(A:B|C) + I(A:C|B) + 2I(B:C|A) + 3I(A:B|D) + I(B:D|A) + 2I(C:D)
    if(eq_no == 0 or eq_no == 3):
        res = 2*I_A_B <= 4*I_AB_C + I_AC_B + 2*I_BC_A +3*I_AB_D + I_BD_A + 2*I_C_D
        result.append(res)

    # 2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 4I(B:C|A) + 2I(A:C|D) + I(A:D|C) + ...
    # 2I(B:D) + I(C:D|A)
    if(eq_no == 0 or eq_no == 4):
        res =  2*I_A_B <= 3*I_AB_C + 2*I_AC_B + 4*I_BC_A + 2*I_AC_D + I_AD_C + 2*I_B_D + I_CD_A
        result.append(res)

    # 2I(A:B) <= 5I(A:B|C) + 3I(A:C|B) + I(B:C|A) + 2I(A:D) + 2I(B:C|D)
    if(eq_no == 0 or eq_no == 5):
        res = 2*I_A_B <= 5*I_AB_C + 3*I_AC_B + I_BC_A + 2*I_A_D + 2*I_BC_D
        result.append(res)

    # 2I(A:B) <= 4I(A:B|C) + 4I(A:C|B) + I(B:C|A) + 2I(A:D) + 3I(B:C|D) + I(C:D|B)
    if(eq_no == 0 or eq_no == 6):
        res = 2*I_A_B <= 4*I_AB_C + 4*I_AC_B + I_BC_A + 2*I_A_D + 2*I_BC_D + I_CD_B
        result.append(res)

    # 2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 2I(B:C|A) + 2I(A:B|D) + I(A:D|B) + ...
    # I(B:D|A) + 2I(C:D)
    if(eq_no == 0 or eq_no == 7):
        res =  2*I_A_B <= 3*I_AB_C + 2*I_AC_B + 2*I_BC_A + 2*I_AB_D + I_AD_B + I_BD_A + 2*I_C_D
        result.append(res)

    if(eq_no == 0):
        return result

    return res
