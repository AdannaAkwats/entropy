import numpy as np
from numpy import linalg as LA
import math
import random

# Calculate the Von Neumann Entropy of a quantum state
def vonNeumann(A):
    # Get eigenvalues of A
    values, _ = LA.eig(A)

    print(values)

    # Take the natural logarithm of the eigenvalues and sum them
    sum = 0
    for v in values:
        l = 0
        if(v == 0) :
            l = 0
        else:
            l = math.log(v)
        sum += l

    return -sum

# Generate random quantum state of dimension n
def generateRandomState(n):
    # D: diagonal matrix
    D = np.zeros((n,n))
    diag = np.random.random(n)
    diag /= diag.sum()
    for i in range(n):
        D[i][i] = diag[i]

    # U: unitary matrix
    # Generate n*n random complex numbers
    U  = np.random.rand(2 * n * n).view(np.complex128)

    # Fill in U
    U = np.zeros((n,n))
    U_a = np.array(U, dtype=np.complex128)
    for i in range(n):
        for j in range(n):
            U_a[i][j] = u_rand[i]

    # Turn Array into Matrix
    U_mat = np.matrix(U_a)

    # U conjugate
    U_conj = U_mat.getH()

    # Get UDU* = A
    D_mat = np.matrix(D)
    UD = np.matmul(U_mat, D_mat)
    # A = UDU*
    A = np.matmul(UD, U_conj)

    return A


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

    # Divide diagonal of R with its absolute value
    diag_R = np.diag(R)
    R_temp_diag = np.zeros(len(diag_R))
    R_diag = X = np.array(R_temp_diag, dtype=np.complex128)
    for i in range(len(diag_R)):
        R_diag[i] = diag_R[i] / abs(diag_R[i])

    # Make matrix with new diagonal
    #R = np.diag(R_diag)
    R = np.diag(diag_R)
    #print(R)

    # Calculate U = Q * R
    U = np.matmul(Q, R)

    # Verify that U is unitary
    U_mat = np.matrix(U)

    # U conjugate
    U_conj = U_mat.getH()
    I = np.matmul(U, U_conj)

    # unitary matrix
    #U = Q * R
    return U, U_conj, I

# Generate random A s.t A = UDU*, where D is diagonal, U is unitary matrix and
# U* is conplex conjugate transpose of U 
def generate(n):

    # Unitary matrix and its complex conjugate transpose
    U, U_conj, I = Unitary(n)

    # D: diagonal matrix
    D = np.zeros((n,n))
    diag = np.random.random(n)
    diag /= diag.sum()
    for i in range(n):
        D[i][i] = diag[i]

    # A = UDU*
    D_mat = np.matrix(D)
    UD = np.matmul(U, D)
    # A = UDU*
    A = np.matmul(UD, U_conj)

    return A


# Testing...

g = generate(2)
print(g)

v = vonNeumann(g)
print(v)
