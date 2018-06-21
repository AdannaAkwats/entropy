import numpy as np
import math
import random

from numpy import linalg as LA
from shannon import randomProbabilityDist
from partial_trace import separate
from entropy import *
from utils import *

# Non shannon-type entropies from paper
# http://www.cnd.mcgill.ca/~ivan/it_ineq_script/Raymond%20Yeung%20papers/04035957.pdf
# Theorem II.2
def non_shannon_1(pABCD,dim):
    """
    Returns true if:
    Zhang-Yeung inequality
    2I(C:D) <= I(A:B) + I(A:C,D) + 3I(C:D|A) + I(C:D|B)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq2 in entropy.py")

    s,j,j3,_ = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_C_D = mutual_information(pCD,dim)        # I(C:D)
    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_ACD = and_mutual_information(pACD,dim)   # I(A:C,D)

    I_CD_A = cond_mutual_information(pAC, pA, pACD, pAD, dim) # I(C:D|A)
    I_CD_B = cond_mutual_information(pBC, pB, pBCD, pBD, dim) # I(C:D|B)

    LHS = 2*I_C_D
    RHS = I_A_B + I_ACD + 3*I_CD_A + I_CD_B

    # How close is it to 0?
    diff = RHS - LHS

    return LHS <= RHS, diff

def non_shannon_1_ghz(pABCD,dim):
    """
    Returns true if:
    Zhang-Yeung inequality
    2I(C:D) <= I(A:B) + I(A:C,D) + 3I(C:D|A) + I(C:D|B)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq2 in entropy.py")

    s,j,j3,_ = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[0], j3[0], j3[0]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[0], j[0], j[0], j[0], j[0]
    pA, pB, pC, pD = s[0], s[0], s[0], s[0]

    H_C = vonNeumann(pC)
    H_D = vonNeumann(pD)
    H_CD = vonNeumann(pCD)
    I_C_D = H_C + H_D - H_CD        # I(C:D)
    H_A = vonNeumann(pA)
    H_B = vonNeumann(pB)
    H_AB = vonNeumann(pAB)
    I_A_B = H_A + H_B - H_AB         # I(A:B)
    H_ABC = vonNeumann(pABC)
    I_ACD = H_A + H_AB - H_ABC   # I(A:C,D)

    I_CD_A = H_AB + H_AB - H_A - H_ABC # I(C:D|A)
    I_CD_B = H_AB + H_AB - H_A - H_ABC# I(C:D|B)

    LHS = 2*I_C_D
    RHS = I_A_B + I_ACD + 3*I_CD_A + I_CD_B

    # How close is it to 0?
    diff = RHS - LHS

    return LHS <= RHS, diff




# Theorem III.1
def non_shannon_2(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 3I(A:C|B) + 3I(B:C|A) + 2I(A:D) +2I(B:C|D)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq3 in entropy.py")

    s,j,j3,_ = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_A_D = mutual_information(pAD,dim)        # I(A:D)

    I_AB_C = cond_mutual_information(pAC, pC, pABC, pBC, dim) # I(A:B|C)
    I_AC_B = cond_mutual_information(pAB, pB, pABC, pBC, dim) # I(A:D|B)
    I_BC_A = cond_mutual_information(pAB, pA, pABC, pAC, dim) # I(B:C|A)
    I_BC_D = cond_mutual_information(pBD, pD, pBCD, pCD, dim) # I(B:C|D)

    LHS = 2*I_A_B
    RHS = 3*I_AB_C + 3*I_AC_B + 3*I_BC_A + 2*I_A_D + 2*I_BC_D

    # How close is it to 0?
    diff = RHS - LHS

    return LHS <= RHS, diff


# Theorem III.2
def non_shannon_3(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + I(A:C|B) + 2I(B:C|A) + 3I(A:B|D) + I(B:D|A) + 2I(C:D)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq4 in entropy.py")

    s,j,j3,_ = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_C_D = mutual_information(pCD,dim)        # I(C:D)

    I_AB_C = cond_mutual_information(pAC, pC, pABC, pBC, dim) # I(A:B|C)
    I_AC_B = cond_mutual_information(pAB, pB, pABC, pBC, dim) # I(A:C|B)
    I_BC_A = cond_mutual_information(pAB, pA, pABC, pAC, dim) # I(B:C|A)
    I_AB_D = cond_mutual_information(pAD, pD, pABD, pBD, dim) # I(A:B|D)
    I_BD_A = cond_mutual_information(pAB, pA, pABD, pAD, dim) # I(B:D|A)

    LHS = 2*I_A_B
    RHS = 4*I_AB_C + I_AC_B + 2*I_BC_A +3*I_AB_D + I_BD_A + 2*I_C_D

    # How close is it to 0?
    diff = RHS - LHS

    return LHS <= RHS, diff


# III.3
def non_shannon_4(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 4I(B:C|A) + 2I(A:C|D) + I(A:D|C) +
    2I(B:D) + I(C:D|A)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq5 in entropy.py")

    s,j,j3,_ = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_B_D = mutual_information(pBD,dim)        # I(B:D)

    I_AB_C = cond_mutual_information(pAC, pC, pABC, pBC, dim) # I(A:B|C)
    I_AC_B = cond_mutual_information(pAB, pB, pABC, pBC, dim) # I(A:C|B)
    I_BC_A = cond_mutual_information(pAB, pA, pABC, pAC, dim) # I(B:C|A)
    I_AC_D = cond_mutual_information(pAD, pD, pACD, pCD, dim) #(A:C|D)
    I_AD_C = cond_mutual_information(pAC, pC, pACD, pCD, dim) #(A:D|C)
    I_CD_A = cond_mutual_information(pAC, pA, pACD, pAD, dim) #(C:D|A)

    LHS = 2*I_A_B
    RHS = 3*I_AB_C + 2*I_AC_B + 4*I_BC_A + 2*I_AC_D + I_AD_C + 2*I_B_D + I_CD_A

    # How close is it to 0?
    diff = RHS - LHS

    return  LHS <= RHS, diff

# Theorem III.4
def non_shannon_5(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 5I(A:B|C) + 3I(A:C|B) + I(B:C|A) + 2I(A:D) + 2I(B:C|D)
    """
    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq6 in entropy.py")

    s,j,j3,_ = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_A_D = mutual_information(pAD,dim)        # I(A:D)

    I_AB_C = cond_mutual_information(pAC, pC, pABC, pBC, dim) # I(A:B|C)
    I_AC_B = cond_mutual_information(pAB, pB, pABC, pBC, dim) # I(A:C|B)
    I_BC_A = cond_mutual_information(pAB, pA, pABC, pAC, dim) # I(B:C|A)
    I_BC_D = cond_mutual_information(pBD, pD, pBCD, pCD, dim) # I(B:C|D)

    LHS = 2*I_A_B
    RHS = 5*I_AB_C + 3*I_AC_B + I_BC_A + 2*I_A_D + 2*I_BC_D

    # How close is it to 0?
    diff = RHS - LHS

    return  LHS <= RHS, diff

# Theorem III.5
def non_shannon_6(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + 4I(A:C|B) + I(B:C|A) + 2I(A:D) + 3I(B:C|D) + I(C:D|B)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq7 in entropy.py")

    s,j,j3,_ = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_A_D = mutual_information(pAD,dim)        # I(A:D)

    I_AB_C = cond_mutual_information(pAC, pC, pABC, pBC, dim) # I(A:B|C)
    I_AC_B = cond_mutual_information(pAB, pB, pABC, pBC, dim) # I(A:C|B)
    I_BC_A = cond_mutual_information(pAB, pA, pABC, pAC, dim) # I(B:C|A)
    I_BC_D = cond_mutual_information(pBD, pD, pBCD, pCD, dim) # I(B:C|D)
    I_CD_B = cond_mutual_information(pBC, pB, pBCD, pBD, dim) # I(C:D|B)

    LHS = 2*I_A_B
    RHS = 4*I_AB_C + 4*I_AC_B + I_BC_A + 2*I_A_D + 2*I_BC_D + I_CD_B

    # How close is it to 0?
    diff = RHS - LHS

    return LHS <= RHS, diff

# Theorem III.6
def non_shannon_7(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 2I(B:C|A) + 2I(A:B|D) + I(A:D|B) + I(B:D|A) + 2I(C:D)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq8 in entropy.py")

    s,j,j3,_ = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_C_D = mutual_information(pCD,dim)        # I(C:D)

    I_AB_C = cond_mutual_information(pAC, pC, pABC, pBC, dim) # I(A:B|C)
    I_AC_B = cond_mutual_information(pAB, pB, pABC, pBC, dim) # I(A:C|B)
    I_BC_A = cond_mutual_information(pAB, pA, pABC, pAC, dim) # I(B:C|A)
    I_AB_D = cond_mutual_information(pAD, pD, pABD, pBD, dim) # I(A:B|D)
    I_AD_B = cond_mutual_information(pAB, pB, pABD, pBD, dim) # I(A:D|B)
    I_BD_A = cond_mutual_information(pAB, pA, pABD, pAD, dim) # I(B:D|A)

    LHS = 2*I_A_B
    RHS = 3*I_AB_C + 2*I_AC_B + 2*I_BC_A + 2*I_AB_D + I_AD_B + I_BD_A + 2*I_C_D

    # How close is it to 0?
    diff = RHS - LHS

    return LHS <= RHS, diff
