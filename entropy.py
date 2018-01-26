import numpy as np
from numpy import linalg as LA
import math
import random
from shannon import randomProbabilityDist

# Calculate the Von Neumann Entropy of a quantum state
def vonNeumann(A):
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


# Generate random unitary matrix
def Unitary(n):

    # generate a random complex matrix
    temp = np.zeros((n,n))
    u_rand = np.random.randn(2 * n * n).view(np.complex128)
    X = np.array(temp, dtype=np.complex128)
    k = 0
    for i in range(n):
        for j in range(n):
            X[i][j] = u_rand[k]
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


# Generate random A s.t A = UDU*, where D is diagonal, U is unitary matrix and
# U* is conplex conjugate transpose of U
def generate(n):

    # Unitary matrix and its complex conjugate transpose
    U, U_conj, I = Unitary(n)

    # D: diagonal matrix filled with prob distribution so all entries add to 1
    D = np.zeros((n,n))
    diag = randomProbabilityDist(n)
    for i in range(n):
        D[i][i] = diag[i]

    # A = UDU*
    D_mat = np.matrix(D)
    UD = np.matmul(U, D)
    # A = UDU*
    A = np.matmul(UD, U_conj)

    return A
