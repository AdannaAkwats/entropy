import numpy as np
from numpy import linalg as LA
import math
import random
import timeit

from shannon import *
from separate_probs import *
from entropy import *
from evolution import *
from partial_trace import *
from utils import  *
from generate_random_quantum import *



########### Shannon ###################################
p = randomProbabilityDist(2**2)
# print(p)
#
# print subadditivity(p)
# print mutual_information_s(p)
# print conditionXY(p)
# print mutualInfo_leq_HY(p)
# print mutualInfo_leqMin(p)
# print cond_leq_HY(p)
# print HXY_geq_max(p)
# print ""

pp = randomProbabilityDist(2**4)
# s, j, _ = separate_probs(pp)
# print pp
# print ""
# print j[0]
# print j[1]
# print j[2]
# print ""
# print s[0]
# print s[1]
# print s[2]

# print ""
# print strongSubadditivity(pp)
# #print test_true(strongSubadditivity,pp,100)
# print and_mutual_information_s(pp)
# #print cond_mutual_information_s
# print ""
# q = randomProbabilityDist(2**3)
# print_seps(pp)


# s, j, j3 = separate_probs(q)
# print j3[0].sum()#pABC
# print j3[1] #pABD
# print ""
# s1, j1, j31 = separate_probs(j3[0])
# print s1
# print ""
# s2, j2, j32 = separate_probs(j3[1])
# print s2
# print ""
# print_seps(q)

# print new_eq1_s(pp)
# print new_eq2_s(pp)
# print new_eq3_s(pp)
# print new_eq4_s(pp)
# print new_eq5_s(pp)
# print new_eq6_s(pp)
# print new_eq7_s(pp)
# print ""
# print non_shannon_eqs(pp,0)


########### Generate matrices #########################
t = timeit.Timer("generate(4)", "from generate_random_quantum import generate")
print("Time taken for generate_1: " + str(t.timeit()))
s = timeit.Timer("generate_2(4)", "from generate_random_quantum import generate_2")
print("Time taken for generate_2 :" + str(s.timeit()))


# q = test_random_density_matrix(4)
# s, _, _ = separate(q,2)
#
#
# for i in range(2000):
#     p2 = generate(4)
#     res = weak_subadditivity(p2, 2)
#     if(not res):
#         print("False")
#     #print(res)

# Generate random unitary matrix
#U,_,_ = unitary(2)
# #p = generate(2)
#U = generate_unitary(2)
h = generate_hermitian(2)
# print "h"
# print h
# print ""
#
# print time_evol_U(0,h)
# time_evol_is_unitary(2, h)

# is_unitary(U)
# print ""

# Generate random density matrix
# p = generate_pure_state(4,2)
# print p
# print ""
# print np.trace(p)
# print ""
# u = test_generate_unitary(2)
# print u
# print ""
# gg = generate2(2)
# print gg
# print ""
# print "fid " + str(fidelity(gg,gg))
#
#
#
# p = test_random_density_matrix(2)
# print p
# q =  generate(2)
# print q

# p = generate(16)
# print test_true(non_shannon_2,p,2,100)
# print non_shannon_eqs_q(p,2,0)


#################### Entropy inequalities ##############

# 2 qubit systems
# g = generate_pure_state(2,2)
# r = generate(2)
# print "fidelity " + str(fidelity(q,q))
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


#################### Evolution #########################
#p = generate_pure_state(2,2)
# is_CPTP_entropy_more(bit_flip_channel, p, 0.1)
# is_CPTP_entropy_more(phase_flip_channel, p, 0.1)
# is_CPTP_entropy_more(bit_phase_flip_channel, p, 0.1)
# is_CPTP_entropy_more(depolarising_channel, p, 0.1)
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
# seps, j,_= separate(p)
# for s in seps:
#     print s
#     print ""
#     print np.trace(s)

#j,seps,_ = separate(p)
# for s in seps:
#     print s
#     print ""
#     sep,_,_ = separate(s)
#     print sep
#     #print np.trace(s)
#     print ""


################## Test function multiple times ###########

#testTrue(strongSubadditivity_q, p, 1)
