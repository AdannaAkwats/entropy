import numpy as np
import math
import random

from numpy import linalg as LA
from shannon import randomProbabilityDist
from partial_trace import separate
from entropy import vonNeumann
from utils import *


def mutual_information(pAB, pA, pB, dim):
    """
    calculates the mutual information defined by: I(A:B) = H(A) + H(B) - H(A,B)
    """
    # Ensure that system is a 2 qubit/qutrit quantum system
    check_n_q(pAB, dim, 2, "mutual_information in entropy.py")

    H_A = vonNeumann(pA)
    H_B = vonNeumann(pB)
    H_AB = vonNeumann(pAB)

    return H_A + H_B - H_AB


def cond_mutual_information(pAC, pC, pABC, pBC, dim):
    """
    calculates conditional mutual information
    I(A:B|C) = H(A,C) - H(C) - H(A,B,C) + H(B,C)
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "cond_mutual_information in entropy.py")

    H_AC = vonNeumann(pAC)
    H_C = vonNeumann(pC)
    H_ABC = vonNeumann(pABC)
    H_BC = vonNeumann(pBC)

    return H_AC + H_BC - H_C - H_ABC


def and_mutual_information(pABC, pA, pBC, dim):
    """
    calculates I(A:B,C) = H(A) + H(B,C) - H(A,B,C)
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "and_mutual_information in entropy.py")

    H_BC = vonNeumann(pBC)
    H_A = vonNeumann(pA)
    H_ABC = vonNeumann(pABC)

    return H_A + H_BC - H_ABC


def non_shannon_eqs_q(pABCD, dim, eq_no):
    """
    Returns true if all the non shannon type inequalities hold.
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "non_shannon_eqs_q in entropy.py")

    s,j,j3 = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    result = []
    res = False

    I_A_B = mutual_information(pAB,pA,pB,dim)        # I(A:B)
    I_A_D = mutual_information(pAD,pA,pD,dim)        # I(A:D)
    I_B_D = mutual_information(pBD,pB,pD,dim)        # I(B:D)
    I_C_D = mutual_information(pCD,pC,pD,dim)        # I(C:D)

    I_ACD = and_mutual_information(pACD,pA,pCD,dim)   # I(A:C,D)

    I_AB_C = cond_mutual_information(pAC, pC, pABC, pBC, dim) # I(A:B|C)
    I_AB_D = cond_mutual_information(pAD, pD, pABD, pBD, dim) # I(A:B|D)

    I_AC_B = cond_mutual_information(pAB, pB, pABC, pBC, dim) # I(A:C|B)
    I_AC_D = cond_mutual_information(pAD, pD, pACD, pCD, dim) # I(A:C|D)

    I_AD_B = cond_mutual_information(pAB, pB, pABD, pBD, dim) # I(A:D|B)
    I_AD_C = cond_mutual_information(pAC, pC, pACD, pCD, dim) # I(A:D|C)

    I_BC_A = cond_mutual_information(pAB, pA, pABC, pAC, dim) # I(B:C|A)
    I_BC_D = cond_mutual_information(pBD, pD, pBCD, pCD, dim) # I(B:C|D)

    I_BD_A = cond_mutual_information(pAB, pA, pABD, pAD, dim) # I(B:D|A)

    I_CD_A = cond_mutual_information(pAC, pA, pACD, pAD, dim) # I(C:D|A)
    I_CD_B = cond_mutual_information(pBC, pB, pBCD, pBD, dim) # I(C:D|B)


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
