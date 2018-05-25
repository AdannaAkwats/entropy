import numpy as np
from numpy import linalg as LA
import math
import random

from shannon import *
from separate_probs import *
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


########### Shannon ###################################
p = randomProbabilityDist(2**2)
print(p)

print subadditivity(p)
print mutual_information_s(p)
print conditionXY(p)
print mutualInfo_leq_HY(p)
print mutualInfo_leqMin(p)
print cond_leq_HY(p)
print HXY_geq_max(p)

pp = randomProbabilityDist(2**3)
print strongSubadditivity(pp)
print and_mutual_information_s(pp)

q = randomProbabilityDist(2**4)
print new_eq1_s(q)
print new_eq2_s(q)
print new_eq3_s(q)
print new_eq4_s(q)
print new_eq5_s(q)
print new_eq6_s(q)
print new_eq7_s(q)


# s, js, js3 = separate_probs(p)
# print js3
# print ""
# print js
# print ""
# print js3[0].sum()
# print(pxyz)
# print(pxyw)
# print(pyzw)
# print(pxzw)
# print np.sum(pxyz), np.sum(pxyw), np.sum(pyzw), np.sum(pxzw)
#
# pxy, pyz, pxz = separate_3(pxyz)
# pxy1, pyw, pxw = separate_3(pxyw)
# pyz1, pzw, pyw1 = separate_3(pyzw)
# pxz1, pzw1, pxw1 = separate_3(pxzw)
# print pxy
# print pxy1
# print ""
# print pyz
# print pyz1
# print ""
# print pyw
# print pyw1
# print ""
# print pxz
# print pxz1
# print ""
# print pzw
# print pzw1
# print ""
# print pxw
# print pxw1



# pxy, pyz, pxz = separate_3(p)
# print(pxy)
# print(pyz)
# print(pxz)
#
# print np.sum(pxy), np.sum(pyz), np.sum(pxz)
#
# px, py = separate_2(pxy)
# py1, pz = separate_2(pyz)
# px1, pz1 = separate_2(pxz)
#
# print px
# print px1
# print ""
# print py
# print py1
# print ""
# print pz
# print pz1


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
# s,j,j3 = separate(p,2)
# for i in s:
#     print i
#     print ""
# print ""
#
# # print new_eq2(p, 2)
# print new_eq7(p,2)
# print test_true(new_eq8,p,2,100)

# print cond_mutual_information(pACD,2) # I(A:C|D)
#
# # I(C:D|A)
# pA = s[0]
# pAC, pAD = j[2], j[4]
#
#
# H_AC = vonNeumann(pAC)
# H_A = vonNeumann(pA)
# H_ACD = vonNeumann(pACD)
# H_AD = vonNeumann(pAD)
#
# print H_AC - H_A - H_ACD + H_AD

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

print ""

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
