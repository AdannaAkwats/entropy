import numpy as np
from numpy import linalg as LA
import math
import random
from shannon import randomProbabilityDist
from partial_trace import separate

def vonNeumann(A):
    """
    Calculate the Von Neumann Entropy of a quantum state
    """

    # Get eigenvalues of A
    values, _ = LA.eig(A)

    # Take the natural logarithm of the eigenvalues and sum them
    sum = 0
    for v in values:
        l = 0
        if(v == 0) :
            l = 0
        else:
            try:
                l = math.log(v)
            except ValueError:
                print "(The matrix is not a quantum state!)"
        sum += l

    return -sum


def isNonNegVN(A):
    """
    Returns true if vonNeumann entropy >= 0
    """
    return vonNeumann(A) >= 0


def strongSubadditivity_q(pABC):
    """
    Checks that strong subadditivity holds: H(A,B,C) + H(B) <= H(A, B) + H(B,c)
    """

    systems, joint_systems = separate(pABC)
    pAB = joint_systems[0]
    pBC = joint_systems[1]

    H_ABC = vonNeumann(pABC)
    H_AB = vonNeumann(pAB)
    H_BC = vonNeumann(pBC)
    H_B = vonNeumann(pB)

    return H_ABC + H_B <= H_AB + H_BC


def Unitary(n):
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


def generate(n):
    """
    Generate random nxn matrix A s.t A = UDU*, where D is diagonal,
    U is unitary matrix and U* is conplex conjugate transpose of U
    """

    # Unitary matrix and its complex conjugate transpose
    U, U_conj, I = Unitary(n)

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
