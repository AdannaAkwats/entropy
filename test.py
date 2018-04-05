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
from entropy import strongSubadditivity_q
from evolution import isEntropyConstant
from evolution import u_p_u
from evolution import isCPTPEntropyMore
from evolution import depolarising_channel
from evolution import bit_flip_channel
from evolution import phase_flip_channel
from evolution import bit_phase_flip_channel
from partial_trace import separate
from partial_trace import separate_qutrit
from utils import isMatrixSame



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


# Generate random unitary matrix
U,_,_ = Unitary(4)
# print U
# Generate random density matrix
p = generate(2)
print p
print ""
print np.trace(p)
print ""
E = phase_flip_channel(p, 0.5)
print E
print np.trace(E)
I =  np.zeros((2,2))
I[0,0] = 1
I[1,1] = 1
E_I = phase_flip_channel(I, 0.5)
print E_I
print isCPTPEntropyMore(phase_flip_channel, p, 0.5)



# Separate density matrix into several systems
#seps,_ = separate(p)

#testTrue(strongSubadditivity_q, p, 1)

#j,seps = separate(p)
# for s in seps:
#     print s
#     print ""
#     sep,_ = separate(s)
#     print sep
#     #print np.trace(s)
#     print ""
#
# print j
# print isEntropyConstant(u_p_u, p, U)
