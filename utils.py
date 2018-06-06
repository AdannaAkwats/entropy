import numpy as np
from numpy import linalg as LA
import math
import random
import sys


def is_close_to_zero(n):
    """
    Returns true if n is close to 0
    """
    return abs(n) <= 0.0001


def isclose(a, b, rel_tol=1e-14, abs_tol=0.0):
    """
    Compares floating point numbers
    """
    close = abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    close_to_zero = is_close_to_zero(a) and is_close_to_zero(b)
    return close or close_to_zero

def allclose(A, B):
    """
    Compares matrices A and B and returns true if they are equal
    """
    dimA = A.shape[0]
    dimB = B.shape[0]
    if(dimA != dimB):
        return False

    for i in range(dimA):
        for j in range(dimA):
            A_r = A[i,j].real
            A_c = A[i,j].imag
            B_r = A[i,j].real
            B_c = A[i,j].imag
            if(not(isclose(A_r, B_r) and isclose(A_c, B_c))):
                return False
    return True


def isMatrixSame(A, B):
    """
    Returns true if matrices A and B are equal
    """
    dimA = A.shape
    dimB = B.shape

    if(dimA != dimB):
        return False

    for i in range(dimA[0]):
        for j in range(dimA[0]):
            c = isclose(A[i,j],B[i,j])
            if(not c):
                return False
    return True


def matrixInList(A, L):
    """
    Returns true if matrix A is the list L
    """
    # if list empty
    if(not L):
        return False

    for l in L:
        if(isMatrixSame(A, l)):
            return True
    return False


def isListSame(A, B):
    """
    Returns true if lists A and B are equal
    """
    dimA = len(A)
    dimB = len(B)

    if(dimA != dimB):
        return False

    for i in range(dimA):
        if(not np.isclose(A[i],B[i])):
            return False
    return True


def listInList(ls, searchList):
    """
    Returns true if list "ls" is the list "searchList"
    """

    # if list empty
    if(not searchList):
        return False

    for l in searchList:
        if(isListSame(ls, l)):
            return True
    return False

def to_list(L):
    """
    Converts list of arrays to list of lists
    """
    newList = []
    for arr in L:
        aList = arr.tolist()
        newList.append(aList)
    return newList


def is_positive_semi_def(p):
    """
    Returns true if demsity matrix p is positive semi definite i.e all its
    eigenvalues are non-negative
    """
    values, _ = LA.eig(p)
    values = values.real
    # number of elements in values that are 0
    s = sum(is_close_to_zero(v) for v in values)
    n = sum(np.isclose(v,1) for v in values)

    # If s == len(values) - 1 and n = 1, then p is a pure state
    pure_state = (s == len(values) -1) and (n == 1)

    return np.all(values >= 0) or pure_state


def allocSub(p):
    """
    Allocates space needed to store each separated density matrix of p
    """
    dim = p.shape[0]
    sub_dim = 0

    if(isPowerof2(dim)): # qubit
        sub_dim = dim / 2
    elif(isPowerof3(dim)): # qutrit
        sub_dim = dim / 3

    # Error if sub_dim is still = 0
    if sub_dim == 0:
        print("Error in Function 'allocSub in utils.py':")
        print("Density matrix dimension not power of 2 (qubit) or 3 (qutrit)")

    temp = np.zeros((sub_dim, sub_dim))
    sub_p = np.matrix(temp, dtype=np.complex128)

    return sub_p

def isPowerof2(n):
    """
    Returns true if n is a power of 2 i.e can be written as 2^q = n
    """
    return (n and not (n & (n-1)))


def isPowerof3(n):
    """
    Returns true if n is a power of 3 i.e can be written as 3^q = n
    """
    # 3^19 = 1162261467
    return 1162261467 % n == 0;


def check_n_q(p, dim, n, func_str):
    """
    Ensures quantum system p is a n qubit/qutrit quantum state
    """
    d = p.shape[0]
    if(d != dim**n):
        print("Error in Function '" + func_str + "':")
        print("Quantum system is not a " + str(n) + "-(" + str(dim) + "-dimensional)" + " quantum system")
        sys.exit()


def check_power(n, pow, func_str):
    """
    Checks if n is written to the pow'th power e.g if pow = 2, then checks if n
    is a square
    """

    p = n ** (1. / pow)
    if(not p.is_integer()):
        print("Error in Function '" + func_str +"':")
        print("n is not to the specified power")
        sys.exit()


def check_power_of_dim(n,dim,func_str):
    """
    n = side of matrix, dim = dimension of quantum state
    Checks that density matrix dimension n = dim^q
    If so, return number of qubits/qutrits/4-dim...
    If not, exit with error
    """

    m = math.log(n)
    k = math.log(dim)
    q = m / k

    five_part_state = isclose(q,5)

    # Checks that q is an integer
    if(q.is_integer() or five_part_state):
        return q
    else:
        print("Error in Function '" + func_str +"':")
        print("Density matrix given is not a " + str(dim) +"-dim state")
        print("i.e. Width and Length of matrix is not in form dim^q")
        sys.exit()

def check_square_matrix(p,func_str):
    """
    Checks that the matrix p is square
    """
    p1 = p.shape[0]
    p2 = p.shape[1]

    if(p1 != p2):
        print("Error in Function '" + func_str +"':")
        print("Density matrix given is not square")
        sys.exit()


def check_same_size(p,r,func_str):
    """
    Checks that matrix p and r are both square and the same size
    If it's not, exit with error
    """
    p1 = p.shape[0]
    p2 = p.shape[1]
    r1 = r.shape[0]
    r2 = r.shape[1]

    # Check that p and r are square
    check_square_matrix(p,func_str)
    check_square_matrix(r,func_str)

    if((p1 != r1) or (p2 != r2)):
        print("Error in Function '" + func_str +"':")
        print("Density matrices size are not equal")
        sys.exit()

def vector_abs(vector):
    """
    Calculates absolute value of a vector
    """
    vector = vector ** 2
    sum = np.sum(vector)
    return sum ** (1. / 2)


def check_true(f, *args):
    """
    *args list contains func arguments and last element of args is "lim"
    Function that runs function f many times and returns true if func returns
    true "lim" amount of times
    """
    l = len(args)
    lim = args[l-1] # Last element of *args is the number of times to run func
    func_args = args[0:l-1] # rest of arguments are the function arguments

    for i in range(lim):
        if not f(*func_args) :
            return False
    return True
