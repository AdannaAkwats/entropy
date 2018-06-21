import numpy as np
import math
import random
import sys

from numpy import linalg as LA
from shannon import randomProbabilityDist
from partial_trace import separate
from utils import *
from generate_random_quantum import *


def vonNeumann(A):
    """
    Calculate the Von Neumann Entropy of a quantum state
    """

    # Check matrix is square
    check_square_matrix(A, "vonNeumann in entropy.py")

    # Get eigenvalues of A
    values, _ = LA.eig(A)
    values = values.real

    # Take the logarithm of the eigenvalues and sum them
    x = 0
    for v in values:
        if not is_close_to_zero(v):
            x += v*np.log2(v)
    return -np.sum(x)


def is_non_neg_VN(A):
    """
    Returns true if vonNeumann entropy >= 0
    """

    S = vonNeumann(A)
    return S > 0 or is_close_to_zero(S)


def is_vn_leq_log(A):
    """
    Returns true if in a d-dim Hibert space the entropy is at most log(d)
    """
    dim = A.shape[0]
    S = vonNeumann(A)
    l = np.log2(dim)

    return S < l or np.isclose(S, l)


def H_X_leq_H_XY(p):
    """
    Returns true if shannon inequality H(X) <= H(XY) holds
    Note: This should fail to hold for von Neumann entropy if state entangled
    """
    sys, _, _, _ = separate(p, 2)
    pA = sys[0]
    S_A = vonNeumann(pA)
    S_AB = vonNeumann(p)

    return S_A <= S_AB


def is_non_neg_relative_entropy(p,r):
    """
    Validates Klein's inequality: The quantum relative entropy is non-negative,
    with equality if and only if p = r
    """
    return relative_entropy(p,r) >= 0


def conditional_entropy(pAB,dim):
    """
    calculates the conditional entropy: S(A|B) = S(A,B) - S(B)
    """

    # Ensure that system is a 2 qubit/qutrit quantum system
    check_n_q(pAB,dim, 2, "conditional_entropy in entropy.py")

    s, _,_, _ = separate(pAB,dim)
    pB = s[1]
    S_B = vonNeumann(pB)
    S_AB = vonNeumann(pAB)

    return S_AB - S_B


def relative_entropy(p,r):
    """
    Calculates the relative entropy of quantum state p to quantum state r
    S(p||r) = tr(p log p) - tr(p log r)
    """

    # Checks that p and r have same size and are both square
    check_same_size(p,r,"relative_entropy in entropy.py")

    #  tr(p log p)
    A = -vonNeumann(p)

    # tr(p log r)
    eign_p,_ = LA.eig(p)
    eign_p = eign_p.real
    eign_r,_ = LA.eig(r)
    eign_r = eign_r.real
    l = np.log2(eign_r)
    B = eign_p*l
    B = np.sum(B)

    # S(p||r)
    return A - B


def monotocity_relative_entropy(pAB, rAB, dim):
    """
    Returns true if relative entropy is monotonic i.e. S(pA || rA) <= S(pAB || rAB)
    """

    # Checks that pAB and rAB have same size and are both square
    check_same_size(pAB,rAB,"monotocity_relative_entropy in entropy.py")

    S_AB = relative_entropy(pAB,rAB)
    s_p,_,_,_ = separate(pAB,dim)
    s_r,_,_,_= separate(rAB,dim)

    pA = s_p[0]
    rA = s_r[0]

    S_A = relative_entropy(pA, rA)

    return (S_A < S_AB) or np.isclose(S_A, S_AB)


def mutual_information(pAB,dim):
    """
    calculates the mutual information defined by: I(A:B) = S(A) + S(B) - S(A,B)
    """
    # Ensure that system is a 2 qubit/qutrit quantum system
    check_n_q(pAB, dim, 2, "mutual_information in entropy.py")

    systems, _, _,_ = separate(pAB,dim)
    pA = systems[0]
    pB = systems[1]
    S_A = vonNeumann(pA)
    S_B = vonNeumann(pB)
    S_AB = vonNeumann(pAB)

    return S_A + S_B - S_AB

def bound_mutual_information(pAB,dim):
    """
    ensures that mutual information is within bound 0 <= I(A:B) <= 2min(S(A),S(B))
    """
    I_AB = mutual_information(pAB,dim)
    s,_,_,_ = separate(pAB,dim)
    lower = (I_AB >= 0)
    S_A = vonNeumann(s[0])
    S_B = vonNeumann(s[1])
    upper = (I_AB <= 2 * np.minimum(S_A, S_B))
    return lower and upper

def bound_mutual_information_log(pAB,dim):
    """
    ensures that mutual information is within bound I(A:B) <= 2log|A| and 2log|B|
    """
    I_AB = mutual_information(pAB,dim)
    s,_,_,_ = separate(pAB,dim)

    # d-dim hilbert space
    a_dim = s[0].shape[0]
    b_dim = s[1].shape[0]

    upper = (2*np.log2(a_dim)) or (2*np.log2(b_dim))
    return I_AB <= upper


def cond_mutual_information(pAC, pC, pABC, pBC, dim):
    """
    calculates conditional mutual information
    I(A:B|C) = S(A,C) - S(C) - S(A,B,C) + S(B,C)
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "cond_mutual_information in entropy.py")

    S_AC = vonNeumann(pAC)
    S_C = vonNeumann(pC)
    S_ABC = vonNeumann(pABC)
    S_BC = vonNeumann(pBC)

    return S_AC + S_BC - S_C - S_ABC


def and_mutual_information(pABC,dim):
    """
    calculates I(A:B,C) = S(A) + S(B,C) - S(A,B,C)
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "and_mutual_information in entropy.py")

    s,j,j3,_ = separate(pABC,dim)
    pA = s[0]
    pBC = j[1]

    S_BC = vonNeumann(pBC)
    S_A = vonNeumann(pA)
    S_ABC = vonNeumann(pABC)

    return S_A + S_BC - S_ABC

def weak_subadditivity(pAB,dim):
    """
    Checks that weak subadditivity holds: S(A,B) <= S(A) + S(B)
    (2 qubit system)
    """

    # Ensure that system is a 2 qubit/qutrit quantum system
    check_n_q(pAB, dim, 2, "weak_subadditivity in entropy.py")

    systems, _, _,_ = separate(pAB,dim)
    pA = systems[0]
    pB = systems[1]
    S_A = vonNeumann(pA)
    S_B = vonNeumann(pB)
    S_AB = vonNeumann(pAB)

    return S_AB <= S_A + S_B


def strong_subadditivity_q(pABC,dim):
    """
    Checks that strong subadditivity holds: S(A,B,C) + S(B) <= S(A, B)
    + H(B,C) (3 qubit system)
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "strong_subadditivity_q in entropy.py")

    systems, joint_systems,_,_ = separate(pABC,dim)
    pAB = joint_systems[0]
    pBC = joint_systems[1]
    pB = systems[1]

    S_ABC = vonNeumann(pABC)
    S_AB = vonNeumann(pAB)
    S_BC = vonNeumann(pBC)
    S_B = vonNeumann(pB)

    return S_ABC + S_B <= S_AB + S_BC

def triangle_inequality(pAB,dim):
    """
    Returns true if S(AB) >= |S(A) - S(B)|
    """

    # Ensure that system is a 2 qubit/qutrit quantum system
    check_n_q(pAB, dim, 2, "triangle_inequality in entropy.py")

    systems, _, _,_ = separate(pAB,dim)
    pA = systems[0]
    pB = systems[1]
    S_A = vonNeumann(pA)
    S_B = vonNeumann(pB)
    S_AB = vonNeumann(pAB)

    abs = np.absolute(S_A - S_B)

    return (S_AB > abs) or np.isclose(S_AB, abs)

def cond_triangle_inequality(pABC, dim):
    """
    Returns true if S(A|BC) >= S(A|C) - S(B|C)
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "cond_triangle_inequality in entropy.py")

    systems, joint_systems,_,_ = separate(pABC,dim)
    pBC = joint_systems[1]
    pAC = joint_systems[1]
    pC = systems[2]

    S_ABC = vonNeumann(pABC)
    S_AC = vonNeumann(pAC)
    S_BC = vonNeumann(pBC)
    S_C = vonNeumann(pC)

    S_AB_C = S_ABC - S_BC
    S_A_C = S_AC - S_C
    S_B_C = S_BC - S_C

    return S_AB_C >= S_A_C - S_B_C

def cond_reduce_entropy(pABC, dim):
    """
    Returns true if S(A|BC) <= S(A|B)
    """

    systems, joint_systems,_,_ = separate(pABC,dim)
    pA = systems[0]
    pB = systems[1]
    pAB = joint_systems[0]
    pBC = joint_systems[1]

    S_A = vonNeumann(pA)
    S_B = vonNeumann(pB)
    S_AB = vonNeumann(pAB)
    S_BC = vonNeumann(pBC)
    S_ABC = vonNeumann(pABC)

    S_A_BC = S_ABC - S_BC
    S_A_B = S_AB - S_B

    return S_A_BC <= S_A_B


def mutual_info_not_increase(pABC, dim):
    """
    Returns true if I(A:B) <= I(A:BC)
    """
    systems, joint_systems,_,_ = separate(pABC,dim)
    pA = systems[0]
    pB = systems[1]
    pAB = joint_systems[0]
    pBC = joint_systems[1]

    S_A = vonNeumann(pA)
    S_B = vonNeumann(pB)
    S_AB = vonNeumann(pAB)
    S_BC = vonNeumann(pBC)
    S_ABC = vonNeumann(pABC)

    S_A_BC = S_A + S_BC - S_ABC
    S_A_B = S_A + S_B - S_AB

    return S_A_B <= S_A_BC


def subadditivity_of_cond_1(pABCD, dim):
    """
    Returns true if S(AB|CD) <= S(A|C) + S(B|D)
    """

    systems, joint_systems,_,_ = separate(pABCD,dim)
    pC = systems[2]
    pD = systems[3]
    pAC = joint_systems[2]
    pBD = joint_systems[3]
    pCD = joint_systems[5]

    S_C = vonNeumann(pC)
    S_D = vonNeumann(pD)
    S_AC = vonNeumann(pAC)
    S_BD = vonNeumann(pBD)
    S_CD = vonNeumann(pCD)
    S_ABCD = vonNeumann(pABCD)

    S_B_D = S_BD - S_D
    S_A_C = S_AC - S_C
    S_AB_CD = S_ABCD - S_CD

    return S_AB_CD <= S_A_C + S_B_D


def subadditivity_of_cond_2(pABC, dim):
    """
    Returns S(AB|C) <= S(A|C) + S(B|C)
    """
    s,j,_,_ = separate(pABC, dim)
    pC = s[2]
    pBC = j[1]
    pAC = j[2]

    S_C = vonNeumann(pC)
    S_BC = vonNeumann(pBC)
    S_AC= vonNeumann(pAC)
    S_ABC = vonNeumann(pABC)

    S_B_C = S_BC - S_C
    S_A_C = S_AC - S_C
    S_AB_C = S_ABC - S_C

    return S_AB_C <= S_A_C + S_B_C


def subadditivity_of_cond_3(pABC, dim):
    """
    Returns S(A|BC) <= S(A|B) + S(A|C)
    """
    s,j,_,_ = separate(pABC, dim)
    pB = s[1]
    pC = s[2]
    pAB = j[0]
    pBC = j[1]
    pAC = j[2]

    S_B = vonNeumann(pB)
    S_C = vonNeumann(pC)
    S_AB = vonNeumann(pAB)
    S_BC = vonNeumann(pBC)
    S_AC = vonNeumann(pAC)
    S_ABC = vonNeumann(pABC)

    S_A_B = S_AB - S_B
    S_A_C = S_AC - S_C
    S_A_BC = S_ABC - S_BC

    return S_A_BC <= S_A_B + S_A_C


def cond_strong_subadditivity(pABCD, dim):
    """
    Returns S(ABC|D) + S(B|D) <= S(AB|D) + S(BC|D)
    """

    systems, joint_systems,j3,_ = separate(pABCD,dim)
    pD = systems[3]
    pBD = joint_systems[3]
    pABD = j3[1]
    pBCD = j3[2]

    S_BCD = vonNeumann(pBCD)
    S_ABD = vonNeumann(pABD)
    S_BD = vonNeumann(pBD)
    S_D = vonNeumann(pD)
    S_ABCD = vonNeumann(pABCD)

    S_BC_D = S_BCD - S_D
    S_AB_D = S_ABD - S_D
    S_B_D = S_BD - S_D
    S_ABC_D = S_ABCD - S_D

    return S_ABC_D + S_B_D <= S_AB_D + S_BC_D
