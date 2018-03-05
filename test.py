import numpy as np
from numpy import linalg as LA
import math
import random

from shannon import randomProbabilityDist
from shannon import pxy
from shannon import shannon
from shannon import subadditivity
from shannon import getPxPy
from shannon import getPxPyPz
from shannon import strongSubadditivity
from entropy import Unitary
from entropy import generate
from evolution import isEntropyConstant
from evolution import u_p_u
from partial_trace import separate
from partial_trace import separate_qutrit
from utils import isMatrixSame

# Testing output...
pxy = randomProbabilityDist(3**2)
# print(p)
# px, py = getPxPy(p)
# print(px)
# print(py)

# Function that runs functions many times
def testTrue(func, args, lim):
    for i in range(lim):
        print "test " + str(i)
        if not func(args) :
            return False
    return True


# Check whether tensor product of separated systems gives original density matrix
def testSeparate(seps):
    dim = seps[0].shape
    product = seps[0]

    for i in range(len(seps)):
        product = np.tensordot(product, seps[1], 0)

    print product

    # temp = np.zeros((dim * 2, dim * 2))
    # mat = np.matrix(temp, dtype=np.complex128)

    return isMatrixSame(p, product)


# print testTrue(subadditivity, pxy, 10000)

# print testTrue(subadditivity, pxy, 10000)

# Generate random unitary matrix
U,_,_ = Unitary(4)
# print U
# Generate random density matrix
p = generate(9)
print p
print ""

# Separate density matrix into several systems
seps = separate_qutrit(p)
for s in seps:
    print s
    print np.trace(s)
    print ""

# print isEntropyConstant(u_p_u, p, U)
