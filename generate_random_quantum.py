import numpy as np
import numpy.matlib
import math
import random

from numpy import linalg as LA
from shannon import randomProbabilityDist
from partial_trace import separate
from utils import *

def unitary(n):
    """
    Generate random nxn unitary matrix -NOT USING HAAR MEASURE
    """
    assert n != 0

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


def generate_unitary(n):
    """
    Generates nxn unitary matrix disributed with Haar Measure
    according to article: https://arxiv.org/pdf/math-ph/0609050.pdf pg11
    """
    assert n != 0
    # Z ares i.i.d. standard complex normal random variables
    # belongs to Ginibre ensemble
    N = (np.matlib.randn(n,n) + 1j*np.matlib.randn(n,n))/np.sqrt(2.0)
    Q,R = LA.qr(N)
    D = np.diagonal(R)
    P = [(d/np.absolute(d)) for d in D]
    U = np.matmul(Q, np.diag(P))

    if(not is_unitary(U)):
        print("Error in function 'generate_unitary in generate_random_quantum.py':")
        print("Matrix generated is not unitary.")
        sys.exit()

    return U


def is_unitary(U):
    """
    Returns true if the matrix U is unitary i.e UU* = I
    """

    # Complex conjuate of U
    U_conj = np.matrix(U)
    U_conj = U_conj.getH()

    I = np.matmul(U,U_conj)

    n = U.shape[0]
    expect_I = np.eye(n)

    return np.allclose(np.diag(I), np.diag(expect_I))


def generate_hermitian(n):
    """
    Generates nxn hermitian matrix
    """
    assert n != 0

    H = (np.matlib.randn(n,n) + 1j*np.matlib.randn(n,n))
    H_conj = np.matrix(H)
    H_conj = H_conj.getH()

    herm = H + H_conj

    if(not is_hermitian(herm)):
        print("Error in function 'generate_hermitian in generate_random_quantum.py':")
        print("Matrix generated is not hermitian.")
        sys.exit()

    return herm

def is_hermitian(H):
    """
    Returns true if the matrix H is hermitian i.e H = H*
    """

    # Complex conjugate of H
    H_conj = np.matrix(H)
    H_conj = H_conj.getH()

    return np.allclose(H, H_conj)


# generates multipartite states
def generate(n):
    """
    Generate random nxn density matrix A s.t A = UDU*, where D is diagonal,
    U is unitary matrix and U* is conplex conjugate transpose of U
    -- A is a multipartite quantum state
    """
    assert n != 0

    # Unitary matrix and its complex conjugate transpose
    U = generate_unitary(n)
    U_conj = np.matrix(U).getH()

    # D: diagonal matrix filled with prob distribution so all entries add to 1
    D = np.zeros((n,n))
    diag = randomProbabilityDist(n)
    for i in range(n):
        D[i,i] = diag[i]

    # A = UDU*
    D_mat = np.matrix(D)
    UD = np.matmul(U, D)
    A = np.matmul(UD, U_conj)

    # For trace of 1
    A = A / np.trace(A)

    return A


def generate_2(n):
    """
    Generate random nxn density matrix A.
    Note: This is a faster implementation than generate_1(n)
    Motivated by: article https://arxiv.org/pdf/math-ph/0609050.pdf page 3
    """
    assert n != 0

    # Generate random matrix from gaussian distribution
    R = np.matlib.randn(n,n) + 1j*np.matlib.randn(n,n)

    # Calculate A x A* to get positive semi definite matrix
    A = R.dot(R.getH())

    # For trace of 1
    A = A / np.trace(A)

    return A


# def generate_pure_state_2(n,dim):
#     """
#     Generate random pure quantum state of dim
#     qubit: dim = 2, qutrit: dim = 3
#     Note: Can only generate up to 3-qubit and qutrit states
#     """
#
#     func_str = "generate_pure_state in generate_random_quantum.py"
#     check_power_of_dim(n,dim,func_str)
#
#     # p_AB = |u><u|_AB
#     # |u>_AB = U_AB |0>_AB
#     O = np.zeros(n)
#     O[0] = 1
#
#     U = generate_unitary(n)
#
#     # |u>AB = U |0>AB
#     u = U.dot(O)
#
#     # <u|AB
#     u_mat = np.matrix(u)
#     u_conj = u_mat.getH()
#
#     # pAB = |u> <u|
#     p = (u_mat.T).dot(u_conj.T)
#
#     seps, joint_systems, js3 = separate(p,dim)
#     pA = seps[0]
#
#     # if 3-qubit or 3-qutrit system
#     if((n == 2**3 and dim == 2) or (n == 3**3 and dim == 3)):
#         pA = joint_systems[0]
#     elif((n == 2**4 and dim == 2) or (n == 3**4 and dim == 3)): # 4-qubit/qutrit
#         pA = js3[0]
#
#     return pA


def generate_pure_state(n):
    """
    Generate random pure quantum state
    Note: Can generate up any qubit and qutrit state as suggested on
    page 119 of:
    https://www.iitis.pl/~miszczak/files/papers/miszczak12generating.pdf
    """
    assert n != 0

    # Generate a random unitary matrix and use the columns as random
    # pure states
    U = generate_unitary(n)

    # Choose random column number and column
    k = np.random.randint(n)
    temp = np.zeros(n)
    # |u>     columns: U[0..n, k]
    u = np.array(temp, dtype=np.complex128)
    for i in range(U.shape[0]):
        u[i] = U[i, k]

    # <u|
    u_mat = np.matrix(u)
    u_conj = u_mat.getH()

    # p = |u> <u|
    p = (u_mat.T).dot(u_conj.T)

    return p

def generate_4part(n, dim):
    """
    Returns mixed entangled dim^4 x dim^4 state.
    dim = 2 qubit, dim = 3 qutrit, ...
    """
    assert n != 0

    # Generate pure state of dim*n
    P = generate_pure_state(n*dim)

    # Take partial trace over one system to get mixed state
    _,_,_,j = separate(P, dim)
    # Return a random mixed state
    # k = np.random.randint(len(j))
    return j[0]

def generate_3(n):
    """
    Assumes that its either qubit or qutrit state
    Returns mixed nxn state.
    """
    assert n != 0

    dim = 0
    if(isPowerof2(n)):
        dim = 2
    elif(isPowerof3(n)):
        dim = 3

    # Generate pure state of dim*n
    P = generate_pure_state(n*dim)

    # Take partial trace over one system to get mixed state
    s,j,j3,j4 = separate(P, dim)
    if(len(j) == 0):
        return s[0]
    if(len(j3) == 0):
        return j[0]
    if(len(j4) == 0):
        return j3[0]
    return j4[0]
