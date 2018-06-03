import math
import numpy as np
from numpy import linalg as LA
import random
from utils import *
from separate_probs import *
from shannon import *


def mutual_information_s(pAB, pA, pB):
    """
    Returns the mutual information content of X and Y: I(X:Y)
    """

    H_A = shannon(pA)
    H_B = shannon(pB)
    H_AB = shannon(pAB)

    return H_A + H_B - H_AB


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


def and_mutual_information_s(pABC, pA, pBC):
    """
    calculates I(A:B,C) = H(A) + H(B,C) - H(A,B,C)
    """

    # Ensure that length of pABC is a cube
    check_power(len(pABC), 3, "and_mutual_information_s in shannon.py")

    H_BC = shannon(pBC)
    H_A = shannon(pA)
    H_ABC = shannon(pABC)

    return H_A + H_BC - H_ABC



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
    res = 0

    I_A_B = mutual_information_s(pAB, pA, pB)        # I(A:B)
    I_A_D = mutual_information_s(pAD, pA, pD)        # I(A:D)
    I_B_D = mutual_information_s(pBD, pB, pD)        # I(B:D)
    I_C_D = mutual_information_s(pCD, pC, pD)        # I(C:D)

    I_ACD = and_mutual_information_s(pACD, pA, pCD)   # I(A:C,D)

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
