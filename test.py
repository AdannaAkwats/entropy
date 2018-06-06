import numpy as np
from numpy import linalg as LA
import math
import random
import time

from shannon import *
from separate_probs import *
from entropy import *
from evolution import *
from partial_trace import *
from utils import  *
from generate_random_quantum import *
from non_shannon_quantum import *


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


#p = generate(2**5)
# # print(p)
# # print("")
# s,j,j3,j4 = separate(p,2)
# print(len(j4))
# print(j4[0].shape)
# print(len(j3))
# print(j3[0].shape)
# print(len(j))
# print(vonNeumann(j3[0]))
# print(vonNeumann(j3[1]))
# print(vonNeumann(j3[2]))
# print(vonNeumann(j3[3]))
# print(vonNeumann(j3[4]))
# print(vonNeumann(j3[5]))
# print(vonNeumann(j3[6]))
# print(vonNeumann(j3[7]))
# print(vonNeumann(j3[8]))
# print(vonNeumann(j3[9]))
# print("")
# print(j[0].shape)
# print(len(s))
# print(s[0].shape)

# print(s)
# print(j)


#
# print(p.shape)
# print(is_hermitian(p))
# print(is_positive_semi_def(p))
# print(np.trace(p))
# print(vonNeumann(p))
print("entangled:")
# ent = 0
# for i in range(100):
#     p = generate_3(2**4, 2)
#     if(is_entangled_ABCD(p, 2,2)):
#         ent = ent + 1
#
# print(ent)

p = generate_3(2**3)
print(p.shape)
# print (5.0).is_integer()
#
print mixed_entangled_bipartite(generate_3, 10, 2)
#print mixed_entangled_joint(3, 2, 1, 20)




########### Generate matrices #########################
# t = timeit.Timer("generate(4)", "from generate_random_quantum import generate")
# print("--- %s seconds ---" % (t.timeit()))
# s = timeit.Timer("generate_2(4)", "from generate_random_quantum import generate_2")
# print("--- %s seconds ---" % (s.timeit()))



# for i in range(3000):
#     p = generate_2(2**2)
#     s,_,_ = separate(p,2)
#     pure = vonNeumann(p)
#     entangled = is_entangled(p, s[1])
#     if(entangled):
#         print(i)
#         print(entangled)
#         print("Less than 0")

# g = generate_pure_state_2(4)
# print(vonNeumann(g))


# p = generate(2**4)
# print is_entangled_ABCD(p, 2, 3)
# s,_,_ = separate()

# print(is_entangled(g))

# print non_shannon_1(p, 3)
#print and_mutual_information(p,3)
# start_time = time.clock()
# p = generate(4)
# print("generate(4): --- %s seconds ---" % (time.clock() - start_time))
#
# start_time = time.clock()
# p = generate_2(4)
# print("generate_2(4): --- %s seconds ---" % (time.clock() - start_time))

#print non_shannon_1(p, 3)


#
# for i in range(10):
#     p = generate(4)
#     r = generate(4)
#     res = monotocity_relative_entropy(p, r, 2)
#     if(not res):
#         print("False")
        # print("HX < HXY ?")
        # print(H_X_leq_H_XY(p))
        # print("cond")
        # print(conditional_entropy(p,2))

#print check_true(H_X_leq_H_XY, p, 1000)


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
