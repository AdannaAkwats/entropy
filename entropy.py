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
    #v = values*np.log2(values)
    return -np.sum(x)


def is_non_neg_VN(A):
    """
    Returns true if vonNeumann entropy >= 0
    """

    H = vonNeumann(A)
    return H > 0 or is_close_to_zero(H)


def is_vn_leq_log(A):
    """
    Returns true if in a d-dim Hibert space the entropy is at most log(d)
    """
    dim = A.shape[0]
    H = vonNeumann(A)
    l = np.log2(dim)

    return H < l or np.isclose(H, l)


def H_X_leq_H_XY(p):
    """
    Returns true if shannon equation H(X) <= H(XY)
    Note: This should fail to hold for von Neumann entropy by page 541 in book:
    Quantum Compuation and Quantum Information
    """
    sys, _, _, _ = separate(p, 2)
    pA = sys[0]
    H_A = vonNeumann(pA)
    H_AB = vonNeumann(p)

    return H_A <= H_AB


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

    s, _,_, _ = separate(pAB,dim)
    pB = s[1]
    H_B = vonNeumann(pB)
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


def monotocity_relative_entropy(pAB, rAB, dim):
    """
    Returns true if relative entropy is monotonic i.e. H(pA || rA) <= H(pAB || rAB)
    """

    # Checks that pAB and rAB have same size and are both square
    check_same_size(pAB,rAB,"monotocity_relative_entropy in entropy.py")

    H_AB = relative_entropy(pAB,rAB)
    s_p,_,_,_ = separate(pAB,dim)
    s_r,_,_,_= separate(rAB,dim)

    pA = s_p[0]
    rA = s_r[0]

    H_A = relative_entropy(pA, rA)

    return H_A <= H_AB


# def monotocity_relative_entropy(pAB, rAB, dim):
#     """
#     Returns true if relative entropy is monotonic i.e. H(pA || rA) <= H(pAB || rAB)
#     """
#
#     # Checks that pAB and rAB have same size and are both square
#     check_same_size(pAB,rAB,"monotocity_relative_entropy in entropy.py")
#
#     H_AB = relative_entropy(pAB,rAB)
#     s_p,j_p,j3_p = separate(pAB,dim)
#     s_r,j_r,j3_r = separate(rAB,dim)
#
#     pA = []
#     rA = []
#     if(len(j3_p) == 0):
#         if(len(j_p) == 0):
#             pA = s_p[0]
#             rA = s_r[0]
#         else:
#             k = np.random.randint(len(j_p))
#             pA = j_p[k]
#             rA = j_r[k]
#     else:
#         h = np.random.randint(len(j3_p))
#         pA = j3_p[h]
#         rA = j3_r[h]
#
#     H_A = relative_entropy(pA, rA)
#
#     return H_A <= H_AB


def mutual_information(pAB,dim):
    """
    calculates the mutual information defined by: I(A:B) = H(A) + H(B) - H(A,B)
    """
    # Ensure that system is a 2 qubit/qutrit quantum system
    check_n_q(pAB, dim, 2, "mutual_information in entropy.py")

    systems, _, _,_ = separate(pAB,dim)
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
    s,_,_,_ = separate(pAB,dim)
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
    s,_,_,_ = separate(pAB,dim)

    # d-dim hilbert space
    a_dim = s[0].shape[0]
    b_dim = s[1].shape[0]

    upper = (2*np.log2(a_dim)) or (2*np.log2(b_dim))
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

    s,j,j3,_ = separate(pABC,dim)
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

    systems, _, _,_ = separate(pAB,dim)
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

    systems, joint_systems,_,_ = separate(pABC,dim)
    pAB = joint_systems[0]
    pBC = joint_systems[1]
    pB = systems[1]

    H_ABC = vonNeumann(pABC)
    H_AB = vonNeumann(pAB)
    H_BC = vonNeumann(pBC)
    H_B = vonNeumann(pB)

    return H_ABC + H_B <= H_AB + H_BC

def bell_states(n):
    """
    Returns Bell's states
    n = 1: Return (|00> + |11>)/sqrt(2)
    n = 2: Return (|00> - |11>)/sqrt(2)
    n = 3: Return (|01> + |10>)/sqrt(2)
    n = 4: Return (|01> - |10>)/sqrt(2)
    """
    p = np.zeros((4,4))
    if(n == 1):
        p[0,0] = 0.5
        p[0,3] = 0.5
        p[3,0] = 0.5
        p[3,3] = 0.5
    elif(n == 2):
        p[0,0] = 0.5
        p[0,3] = -0.5
        p[3,0] = -0.5
        p[3,3] = 0.5
    elif(n == 3):
        p[1,1] = 0.5
        p[1,2] = 0.5
        p[2,1] = 0.5
        p[2,2] = 0.5
    elif(n == 4):
        p[1,1] = 0.5
        p[1,2] = -0.5
        p[2,1] = -0.5
        p[2,2] = 0.5
    else:
        print("Error in function 'bell_states' in entropy.py")
        print("bell state number given is not valid.")
        sys.exit()
    return p


def is_entangled(pAB, pB):
    """
    Returns true if p is entangled in A : B. If H(AB) - H(B) < 0
    """
    H_AB = vonNeumann(pAB)
    H_B = vonNeumann(pB)

    return (H_AB - H_B) < 0


def is_bell_state_max_entangled(n):
    """
    Returns true if bell states are maximally entangled
    0 < n <= 4 represents bell states; go to function bell_states in entropy.py
    """
    p = bell_states(n)
    V = vonNeumann(p)
    if(not np.isclose(V,0)): # Bell states are pure states
        return False

    s,_,_,_ = separate(p,2)

    # In bell state, separated pA = pB
    if(not is_entangled(p, s[0])):
        return False

    # Maximally entangled if H = log d
    H = vonNeumann(s[0])
    if(H == np.log2(s[0].shape[0])):
        return True
    return False


def mixed_entangled_bipartite(gen_func, lim, dim):
    """
    Return number of mixed states and number of mixed entangled states out of
    the n bipartite states generated
    """

    ent = 0
    for i in range(lim):
        p = gen_func(dim**2)
        s,_,_,_ = separate(p, dim)

        if(is_entangled(p, s[1])): #p and pB
            ent = ent + 1

    return lim, ent, lim-ent


def mixed_entangled_joint(gen_func, size, dim, cut, lim):
    """
    size: 3 (tri- partite), 4, 5 partite system
    Return number of mixed states and number of mixed entangled states out of
    the lim 3 4 or 5 partite states generated.
    cut = 1 -> entanglement between pAB|CDE
    cut = 2 -> entanglement between pABC|DE
    """
    ent = 0
    func = is_entangled_ABC
    if (size == 4):
        func = is_entangled_ABCD
    elif(size == 5):
        func = is_entangled_5
    elif(size != 3):
        print("Error in function 'mixed_entangled_joint' in entropy.py")
        print("size given is not valid.")
        sys.exit()

    for i in range(lim):
        p = gen_func(dim**size)
        if(func(p, dim, cut)):
            ent = ent + 1

    return lim, ent, lim-ent



def is_entangled_ABC(pABC, dim, cut):
    """
    Returns true if pABCD is entangled with cut s.t
    cut 1 - pA|BC : H(ABC) - H(A) < 0
    cut 2 - pAB|C : H(ABC) - H(AB) < 0
    """
    s, j, _,_ = separate(pABC, dim)
    if(cut == 1):
        pA = s[0]
        return is_entangled(pABC, pA)
    elif(cut == 2):
        pAB = j[0]
        return is_entangled(pABC, pAB)
    else:
        print("Error in function 'is_entangled_ABC'")
        print("Cut given is not valid.")
        sys.exit()


def is_entangled_ABCD(pABCD, dim, cut):
    """
    Returns true if pABCD is entangled with cut s.t
    cut 1 - pA|BCD : H(ABCD) - H(A) < 0
    cut 2 - pAB|CD : H(ABCD) - H(AB) < 0
    """
    s, j, j3,_ = separate(pABCD, dim)
    if(cut == 1):
        pA = s[0]
        return is_entangled(pABCD, pA)
    elif(cut == 2):
        pAB = j[0]
        return is_entangled(pABCD, pAB)
    else:
        print("Error in function 'is_entangled_ABC'")
        print("Cut given is not valid.")
        sys.exit()


def is_entangled_5(pABCDE, dim, cut):
    """
    Returns true if pABCD is entangled with cut s.t
    cut 1 - pAB|CDE : H(ABCDE) - H(AB) < 0
    cut 2 - pABC|DE : H(ABCDE) - H(ABC) < 0
    """
    s, j, j3,j4 = separate(pABCDE, dim)
    if(cut == 1):
        pA = j[0]
        return is_entangled(pABCDE, pA)
    elif(cut == 2):
        pAB = j3[0]
        return is_entangled(pABCDE, pAB)

    else:
        print("Error in function 'is_entangled_ABC' in entropy.py")
        print("Cut given is not valid.")
        sys.exit()
