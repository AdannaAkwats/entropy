import numpy as np
import math
import random

from numpy import linalg as LA
from shannon import randomProbabilityDist
from partial_trace import separate
from utils import *


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
    v = values*np.log2(values)
    return -np.sum(v)


def is_non_neg_VN(A):
    """
    Returns true if vonNeumann entropy >= 0
    """

    return vonNeumann(A) >= 0


def is_non_neg_relative_entropy(p,r):
    """
    Validates Klein's inequality: The quantum relative entropy is non-negative,
    with equality if and only if p = r
    """
    return relative_entropy(p,r) >= 0


def conditional_entropy(pAB,dim):
    """
    calculates the conditional entropy: H(A|B) = H(A,B) - H(B)
    """

    # Ensure that system is a 2 qubit/qutrit quantum system
    check_n_q(pAB,dim, 2, "conditional_entropy in entropy.py")

    systems, _,_ = separate(pAB,dim)
    pB = systems[1]
    H_B = vonNeumann(B)
    H_AB = vonNeumann(pAB)

    return H_AB - H_B


def relative_entropy(p,r):
    """
    Calculates the relative entropy of quantum state p to quantum state r
    H(p||r) = tr(p log p) - tr(p log r)
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

    # H(p||r)
    return A - B


def monotocity_relative_entropy(pAB, rAB,dim):
    """
    Returns true if relative entropy is monotonic i.e. H(pA || rA) <= H(pAB || rAB)
    """

    # Checks that pAB and rAB have same size and are both square
    check_same_size(pAB,rAB,"monotocity_relative_entropy in entropy.py")

    H_AB = relative_entropy(pAB,rAB)
    s_p,_,_ = separate(pAB,dim)
    pA = s_p[0]
    s_r,_,_ = separate(rAB,dim)
    rA = s_p[0]
    H_A = relative_entropy(pA, rA)
    return H_AB <= H_A


def mutual_information(pAB,dim):
    """
    calculates the mutual information defined by: I(A:B) = H(A) + H(B) - H(A,B)
    """
    # Ensure that system is a 2 qubit/qutrit quantum system
    check_n_q(pAB, dim, 2, "mutual_information in entropy.py")

    systems, _, _ = separate(pAB,dim)
    pA = systems[0]
    pB = systems[1]
    H_A = vonNeumann(pA)
    H_B = vonNeumann(pB)
    H_AB = vonNeumann(pAB)

    return H_A + H_B - H_AB

def bound_mutual_information(pAB,dim):
    """
    ensures that mutual information is within bound 0 <= I(A:B) <= 2min(H(A),H(B))
    """
    I_AB = mutual_information(pAB,dim)
    s,_,_ = separate(pAB,dim)
    lower = (I_AB >= 0)
    H_A = vonNeumann(s[0])
    H_B = vonNeumann(s[1])
    upper = (I_AB <= 2 * np.minimum(H_A, H_B))
    return lower and upper

def bound_mutual_information_log(pAB,dim):
    """
    ensures that mutual information is within bound I(A:B) <= 2log|A| and 2log|B|
    """
    I_AB = mutual_information(pAB,dim)
    s,_,_ = separate(pAB,dim)

    H_A = vonNeumann(s[0])
    H_B = vonNeumann(s[1])

    values, _ = LA.eig(pAB)
    values = values.real

    upper = np.log2(vector_abs(values))
    return I_AB <= upper


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


def and_mutual_information(pABC,dim):
    """
    calculates I(A:B,C) = H(A) + H(B,C) - H(A,B,C)
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "and_mutual_information in entropy.py")

    s,j,_ = separate(pABC,dim)
    pA = s[0]
    pBC = j[1]

    H_BC = vonNeumann(pBC)
    H_A = vonNeumann(pA)
    H_ABC = vonNeumann(pABC)

    return H_A + H_BC - H_ABC

def weak_subadditivity(pAB,dim):
    """
    Checks that weak subadditivity holds: H(A,B) <= H(A) + H(B)
    (2 qubit system)
    """

    # Ensure that system is a 2 qubit/qutrit quantum system
    check_n_q(pAB, dim, 2, "weak_subadditivity in entropy.py")

    systems, _, _ = separate(pAB,dim)
    pA = systems[0]
    pB = systems[1]
    H_A = vonNeumann(pA)
    H_B = vonNeumann(pB)
    H_AB = vonNeumann(pAB)

    return H_AB <= H_A + H_B


def strong_subadditivity_q(pABC,dim):
    """
    Checks that strong subadditivity holds: H(A,B,C) + H(B) <= H(A, B)
    + H(B,C) (3 qubit system)
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "strong_subadditivity_q in entropy.py")

    systems, joint_systems,_ = separate(pABC,dim)
    pAB = joint_systems[0]
    pBC = joint_systems[1]
    pB = systems[1]

    H_ABC = vonNeumann(pABC)
    H_AB = vonNeumann(pAB)
    H_BC = vonNeumann(pBC)
    H_B = vonNeumann(pB)

    return H_ABC + H_B <= H_AB + H_BC
