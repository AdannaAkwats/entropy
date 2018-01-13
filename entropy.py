import numpy as np
from numpy import linalg as LA
import math
import random

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


# -- Generate random unitary matrix
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

# Fill matrix with nunbers from standard input
def fill_matrix_with_input(matrix, n):
    rows = {}
    row_num = 0
    for line in fileinput.input():
        rows[row_num] = line
        row_num = row_num + 1
        if(row_num == n) :
            break
    for i in range(n) :
        each_row = rows[i].split(" ");
        for j in range(n) :
             matrix[i][j] = complex(each_row[j])

# Intro questions
def intro_input() :
    print("Press N : Type new matrix")
    print("Press G : Generate random matrix")
    print("Press Q : Quit")
    print(" ")
    typ = raw_input("[N/G]: ")
    print(" ")
    return typ

# Get dimension of mstrix from input
def get_n_from_input():
    n = input("What is the dimension (n) of the matrix? (eg. 3x3 matrix dimension = 3): ")
    print(" ")
    return n

# Initialise matrix
def init_matrix(n) :
    temp = np.zeros((n,n))
    matrix = np.array(temp, dtype=np.complex128)
    return matrix

# Testing...

# Read input
import fileinput
import sys

# Generating matrix to use
print(" ")
while True:
    typ = intro_input()

    # Choices
    if typ in ['N', 'n', 'G', 'g'] :
        n = get_n_from_input()
        matrix = init_matrix(n)

        if typ == 'N'or typ == 'n':
            print "Input the %dx%d matrix, pressing enter after each row and having a space between each element." %(n,n,)
            print(" ")
            fill_matrix_with_input(matrix, n)
        elif typ == 'G'or typ == 'g':
            matrix = generate(n)
        break
    elif typ in ['Q', 'q']:
        sys.exit()
    else :
        print "ERROR!: Input is not N, G or Q"
        print(" ")

print(" ")

# Print original matrix to console
print "Original Matrix: "
print(" ")
print matrix

# For testing, remove warnings TODO
import warnings
warnings.filterwarnings("ignore")

# Get and print Von Neumann value
print(" ")
print "Von Neumann transformation of Matrix: "
print vonNeumann(matrix)

# Pause at python exe
raw_input()
