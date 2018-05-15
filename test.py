import numpy as np
from numpy import linalg as LA
import math
import random

from shannon import *
from entropy import *
from evolution import *
from partial_trace import *
from utils import  *



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

########### Generate matrices #########################
# Generate random unitary matrix
#U,_,_ = unitary(2)
# #p = generate(2)
# U = generate_unitary(2)
# print U
# print ""
# is_unitary(U)
# print ""

# Generate random density matrix
# p = generate_pure_state(4,2)
# print p
# print ""
# print np.trace(p)
# print ""
u = test_generate_unitary(2)
print u
print ""
gg = generate2(2)
print gg
print ""
print "fid " + str(fidelity(gg,gg))



p = test_random_density_matrix(2)
print p
q =  generate(2)
print q

#################### Entropy inequalities ##############

# 2 qubit systems
# g = generate_pure_state(2,2)
# r = generate(2)
print "fidelity " + str(fidelity(q,q))
# print vonNeumann(g)
# print is_non_neg_VN(g)
# print relative_entropy(g,r)
# print is_non_neg_relative_entropy(g,r)
# print conditional_entropy(g)
# print monotocity_relative_entropy(g,r)
# print mutual_information(g)
# print weak_subadditivity(g)
#
# # 3 qubit system
# y = generate(8)
# print strongSubadditivity_q(y)
#

print ""

#################### Evolution #########################
#p = generate_pure_state(2,2)
# is_CPTP_entropy_more(bit_flip_channel, p, 0.1)
# is_CPTP_entropy_more(phase_flip_channel, p, 0.1)
is_CPTP_entropy_more(bit_phase_flip_channel, p, 0.1)
is_CPTP_entropy_more(depolarising_channel, p, 0.1)
# print bit_flip_channel(p, 0.5)
# print depolarising_channel(p,0.5)
# print bit_phase_flip_channel(p,0.5)

# E = phase_flip_channel(p, 0.5)
# print E
# print np.trace(E)
# I =  np.eye(2)
# E_I = phase_flip_channel(I, 0.78)
# print E_I
# print isCPTPEntropyMore(phase_flip_channel, p, 0.5)
#
# I =  np.eye(2)
# print is_unital(bit_phase_flip_channel, 2)
# E_I = depolarising_channel(I, 0.5)
# print E_I
# print is_CPTP_entropy_more(bit_flip_channel, p, 1)
# print is_PTP(depolarising_channel,p)
# print is_unital(depolarising_channel, 2)
# print is_entropy_constant(u_p_u, p, u)



#################### Separation #########################
# Separate density matrix into several systems
# seps, j= separate(p)
# for s in seps:
#     print s
#     print ""
#     print np.trace(s)

#j,seps = separate(p)
# for s in seps:
#     print s
#     print ""
#     sep,_ = separate(s)
#     print sep
#     #print np.trace(s)
#     print ""


################## Test function multiple times ###########

#testTrue(strongSubadditivity_q, p, 1)
