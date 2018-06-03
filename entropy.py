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
    return H_A <= H_B


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

    values, _ = LA.eig(A)
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

def new_eq1(pABC,dim):
    """
    Returns true if: I(A : C|B)  >= -log F(pABC, R_pBC,trC x I_A(pAB))
    where F is the Uhlmann's fidelity F(p,r) = ||sqrt(p) sqrt(r)||^2 _1
    from https://arxiv.org/pdf/1604.03023.pdf
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "new_eq1 in entropy.py")

    # I(A:C|B) = H(A,B)-H(B)-H(A,B,C)-H(B,C)
    seps,joint,_ = separate(pABC,dim)
    pB = seps[1]
    pAB = joint[0]
    pBC = joint[1]
    H_ABC = vonNeumann(pABC)
    H_AB = vonNeumann(pAB)
    H_BC = vonNeumann(pBC)
    H_B = vonNeumann(pB)

    lhs = H_AB - H_B - H_ABC - H_BC

    # TODO: define R
    fidelity = fidelity(pABC, R)

    rhs = -np.log(fidelity)

    return lhs >= rhs

def fidelity(p,r):
    """
    Calculates fidelity between quantum states p and r
    F = ||sqrt(pr)||^2 _1
    """

    # Checks that p and r have same size and are both square
    check_same_size(p,r,"fidelity in entropy.py")

    sqrt_p = np.sqrt(p)
    p_r_p = sqrt_p*r*sqrt_p
    sqrt_prp = np.sqrt(p_r_p)

    values, _ = LA.eig(sqrt_prp)
    values = values.real

    f = np.sum(values)**2

    return f


# Non shannon-type entropies from paper
# http://www.cnd.mcgill.ca/~ivan/it_ineq_script/Raymond%20Yeung%20papers/04035957.pdf
# Theorem II.2
def new_eq2(pABCD,dim):
    """
    Returns true if:
    2I(C:D) <= I(A:B) + I(A:C,D) + 3I(C:D|A) + I(C:D|B)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq2 in entropy.py")

    s,j,j3 = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_C_D = mutual_information(pCD,dim)  # I(C:D)
    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_ACD = and_mutual_information(pACD,dim)   # I(A:C,D)


    I_CD_A = cond_mutual_information(pAC, pA, pACD, pAD, dim) # I(C:D|A)
    I_CD_B = cond_mutual_information(pBC, pB, pBCD, pBD, dim) # I(C:D|B)

    return 2*I_C_D <= I_A_B + I_ACD + 3*I_CD_A + I_CD_B


# Theorem III.1
def new_eq3(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 3I(A:C|B) + 3I(B:C|A) + 2I(A:D) +2I(B:C|D)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq3 in entropy.py")

    s,j,j3 = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_A_D = mutual_information(pAD,dim)        # I(A:D)

    I_AB_C = cond_mutual_information(pAC, pC, pABC, pBC, dim) # I(A:B|C)
    I_AC_B = cond_mutual_information(pAB, pB, pABC, pBC, dim) # I(A:D|B)
    I_BC_A = cond_mutual_information(pAB, pA, pABC, pAC, dim) # I(B:C|A)
    I_BC_D = cond_mutual_information(pBD, pD, pBCD, pCD, dim) # I(B:C|D)

    return 2*I_A_B <= 3*I_AB_C + 3*I_AC_B + 3*I_BC_A + 2*I_A_D + 2*I_BC_D


# Theorem III.2
def new_eq4(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + I(A:C|B) + 2I(B:C|A) + 3I(A:B|D) + I(B:D|A) + 2I(C:D)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq4 in entropy.py")

    s,j,j3 = separate(pABCD,dim)
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

    return 2*I_A_B <= 4*I_AB_C + I_AC_B + 2*I_BC_A +3*I_AB_D + I_BD_A + 2*I_C_D


# III.3
def new_eq5(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 4I(B:C|A) + 2I(A:C|D) + I(A:D|C) +
    2I(B:D) + I(C:D|A)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq5 in entropy.py")

    s,j,j3 = separate(pABCD,dim)
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

    return 2*I_A_B <= 3*I_AB_C + 2*I_AC_B + 4*I_BC_A + 2*I_AC_D + I_AD_C + 2*I_B_D + I_CD_A

# Theorem III.4
def new_eq6(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 5I(A:B|C) + 3I(A:C|B) + I(B:C|A) + 2I(A:D) + 2I(B:C|D)
    """
    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq6 in entropy.py")

    s,j,j3 = separate(pABCD,dim)
    pABC, pABD, pBCD, pACD = j3[0], j3[1], j3[2], j3[3]
    pAB, pBC, pAC, pBD, pAD, pCD = j[0], j[1], j[2], j[3], j[4], j[5]
    pA, pB, pC, pD = s[0], s[1], s[2], s[3]

    I_A_B = mutual_information(pAB,dim)        # I(A:B)
    I_A_D = mutual_information(pAD,dim)        # I(A:D)

    I_AB_C = cond_mutual_information(pAC, pC, pABC, pBC, dim) # I(A:B|C)
    I_AC_B = cond_mutual_information(pAB, pB, pABC, pBC, dim) # I(A:C|B)
    I_BC_A = cond_mutual_information(pAB, pA, pABC, pAC, dim) # I(B:C|A)
    I_BC_D = cond_mutual_information(pBD, pD, pBCD, pCD, dim) # I(B:C|D)

    return 2*I_A_B <= 5*I_AB_C + 3*I_AC_B + I_BC_A + 2*I_A_D + 2*I_BC_D

# Theorem III.5
def new_eq7(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + 4I(A:C|B) + I(B:C|A) + 2I(A:D) + 3I(B:C|D) + I(C:D|B)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq7 in entropy.py")

    s,j,j3 = separate(pABCD,dim)
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

    return 2*I_A_B <= 4*I_AB_C + 4*I_AC_B + I_BC_A + 2*I_A_D + 2*I_BC_D + I_CD_B

# Theorem III.6
def new_eq8(pABCD,dim):
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 2I(B:C|A) + 2I(A:B|D) + I(A:D|B) + I(B:D|A) + 2I(C:D)
    """

    # Ensure that system is a 4 qubit/qutrit quantum system
    check_n_q(pABCD, dim, 4, "new_eq8 in entropy.py")

    s,j,j3 = separate(pABCD,dim)
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

    return 2*I_A_B <= 3*I_AB_C + 2*I_AC_B + 2*I_BC_A + 2*I_AB_D + I_AD_B + I_BD_A + 2*I_C_D


def unitary(n):
    """
    Generate random nxn unitary matrix
    """

    # generate a random complex matrix
    temp = np.zeros((n,n))
    u_rand = np.random.randn(2 * n * n).view(np.complex128)
    X = np.array(temp, dtype=np.complex128)
    k = 0
    for i in range(n):
        for j in range(n):
            X[i,j] = u_rand[k]
            k += 1

    X /= math.sqrt(2)

    # factorize the matrix
    Q, R = LA.qr(X)
    test = np.allclose(X, np.dot(Q,R))

    # For a complex square matrix U, Q should be unitary, so
    # verify that Q is unitary
    Q_mat = np.matrix(Q)
    # Q conjugate
    Q_conj = Q_mat.getH()
    I = np.matmul(Q, Q_conj)

    # unitary matrix Q
    return Q, Q_conj, I


def is_unitary(U):
    """
    Returns true if the matrix A is unitary
    """

    # Complex conjuate of U
    U_conj = np.matrix(U)
    U_conj = U_conj.getH()

    print np.multiply(U,U_conj)


# generates multipartite states
def generate(n):
    """
    Generate random nxn matrix A s.t A = UDU*, where D is diagonal,
    U is unitary matrix and U* is conplex conjugate transpose of U
    -- A is a multipartite quantum state
    """

    # Unitary matrix and its complex conjugate transpose
    U, U_conj, I = unitary(n)

    # D: diagonal matrix filled with prob distribution so all entries add to 1
    D = np.zeros((n,n))
    diag = randomProbabilityDist(n)
    for i in range(n):
        D[i,i] = diag[i]

    # A = UDU*
    D_mat = np.matrix(D)
    UD = np.matmul(U, D)
    # A = UDU*
    A = np.matmul(UD, U_conj)

    A = A / np.trace(A)

    return A

# TODO : COMPLETE
def generate_unitary(n):
    """
    Generates nxn unitary matrix disributed with Haar Measure
    according to article: https://arxiv.org/pdf/math-ph/0609050.pdf pg11
    """
    # Z ares i.i.d. standard complex normal random variables
    # belongs to Ginibre ensemble
    N = (np.matlib.randn(n,n) + 1j*np.matlib.randn(n,n))/np.sqrt(2.0)
    Q,R = LA.qr(N)
    # D = np.diagonal(R)
    # R = np.diag(D/np.absolute(D))
    # U = Q.dot(R)
    D = np.diagonal(R)
    R = D/np.absolute(D)
    print N
    U = np.multiply(Q,R,np.matrix(Q).getH())

#    print U

    return U

# TODO
def generate_pure_state(n,dim):
    """
    Generate random pure quantum state of dim
    qubit: dim = 2, qutrit: dim = 3, ...
    """

    n = n*2

    func_str = "generate_pure_state in entropy.py"
    check_power_of_dim(n,dim,func_str)

    # p_AB = |u><u|_AB
    # |u>_AB = U_AB |0>_AB
    O = np.zeros(n)
    O[0] = 1

    U = generate_unitary(n)

    # |u>AB = U |0>AB
    u = U.dot(O)

    # <u|AB
    u_mat = np.matrix(u)
    u_conj = u_mat.getH()

    # pAB = |u> <u|
    p = (u_mat.T).dot(u_conj.T)

    seps, joint_systems,_ = separate(p,dim)
    pA = seps[0]
    # if not 2-qubit system
    if(not (n == 4 and dim == 2)):
        pA = joint_systems[0]

    return pA


def test_random_density_matrix(n):

    gin = np.matlib.randn(n,n);
    gin = gin + 1j*np.matlib.randn(n,n);
    rho = gin*(gin.T)
    rho = rho/np.trace(rho);

    return rho

def test_random_pure_state(n):
    v = np.matlib.randn(n,1) + 1j*np.matlib.randn(n,1)
    v = v/LA.norm(v)

    print v

def test_generate_unitary(n):
    gin = np.matlib.randn(n,n)
    gin = gin + 1j*np.matlib.randn(n,n)
    Q,R = LA.qr(gin)
    R = np.sign(np.diag(R))
    #R(R==0) = 1; % protect against potentially zero diagonal entries
    U = Q*np.diag(R)

    # This gives a unitary matrix!!!
    # print U*(np.matrix(U).getH())

    return U

# TODO
def generate2(n):
    """
    Generate random nxn matrix A s.t A = UDU*, where D is diagonal,
    U is unitary matrix and U* is conplex conjugate transpose of U
    -- A is a multipartite quantum state
    """

    # Unitary matrix and its complex conjugate transpose
    U = test_generate_unitary(n)

    # D: diagonal matrix filled with prob distribution so all entries add to 1
    D = np.zeros((n,n))
    diag = randomProbabilityDist(n)
    D = np.diag(diag)

    # A = UDU*
    D_mat = np.matrix(D)
    UD = np.matmul(U, D)
    U_conj = np.matrix(U).getH()
    # A = UDU*
    A = np.matmul(UD, U_conj)

    #A = A / np.trace(A)

    return A
