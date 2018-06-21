import numpy as np
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import random
import time

from shannon import *
from separate_probs import *
from entropy import *
from entangle import *
from evolution import *
from partial_trace import *
from utils import  *
from generate_random_quantum import *
from non_shannon_quantum import *

# TESTING MIXED STATES FOR ENTANGLEMENT
p = generate(2**2)
print "2"
start_time = time.clock()
print mixed_entangled_bipartite(generate, 1000, 2)
print time.clock() - start_time
print mixed_entangled_bipartite(generate_2, 1000, 2)
print time.clock() - start_time
print mixed_entangled_bipartite(generate_3, 1000, 2)
print time.clock() - start_time

print "3"
print mixed_entangled_joint(generate, 3, 2, 1, 1000)
print mixed_entangled_joint(generate_2, 3, 2, 1, 1000)
print mixed_entangled_joint(generate_3, 3, 2, 1, 1000)

print "4"
print mixed_entangled_joint(generate, 4, 2, 1, 1000)
print mixed_entangled_joint(generate_2, 4, 2, 1, 1000)
print mixed_entangled_joint(generate_3, 4, 2, 1, 1000)
print "cut2"
print mixed_entangled_joint(generate, 4, 2, 2, 1000)
print mixed_entangled_joint(generate_2, 4, 2, 2, 1000)
print mixed_entangled_joint(generate_3, 4, 2, 2, 1000)

# Bell states
print is_bell_state_max_entangled(1)
print is_bell_state_max_entangled(2)
print is_bell_state_max_entangled(3)
print is_bell_state_max_entangled(4)

# SPLIT TESTS RUN FOR NON SHANNON INEQUALITIES
diff_list = []
def test_non_shannon1(gen_func, test_func, lim):
    start_time = time.clock()
    for i in range(lim):
        p4 = gen_func(2**4) # 4-qubit state
        res, diff = test_func(p4, 2)
        assert res == True
        diff_list.append(diff)

    print("%d of 4 qubit states tested" % (lim))
    print("%s: --- PASSED in %s seconds ---" % (test_func, time.clock() - start_time))

def test_non_shannon2(gen_func, test_func, lim):
    start_time = time.clock()
    for i in range(lim):
        q4 = gen_func(3**4) # 4-qutrit state
        res, diff = test_func(q4, 3)
        assert res == True
        diff_list.append(diff)

    print("%d of 4 qutrit states tested" % (lim))
    print("%s: --- PASSED in %s seconds ---" % (test_func, time.clock() - start_time))


# PRINT AVG., MIN AND MAX DIFF
def print_result(diff_list):
    print("Avg. diff")
    print(sum(diff_list)/100)
    print("max")
    print(max(diff_list))
    print("min")
    print(min(diff_list))
    diff_list[:] = []


# RUN TESTS
# qubit
test_non_shannon1(generate, non_shannon_1 ,100000)
print_result(diff_list)
test_non_shannon1(generate, non_shannon_2 ,100000)
print_result(diff_list)
test_non_shannon1(generate, non_shannon_3 ,100000)
print_result(diff_list)
test_non_shannon1(generate, non_shannon_4 ,100000)
print_result(diff_list)
test_non_shannon1(generate, non_shannon_5 ,100000)
print_result(diff_list)
test_non_shannon1(generate, non_shannon_6 ,100000)
print_result(diff_list)
test_non_shannon1(generate, non_shannon_7 ,100000)
print_result(diff_list)

# qutrit
test_non_shannon2(generate, non_shannon_1 ,100000)
print_result(diff_list)
test_non_shannon2(generate, non_shannon_2 ,100000)
print_result(diff_list)
test_non_shannon2(generate, non_shannon_3 ,100000)
print_result(diff_list)
test_non_shannon2(generate, non_shannon_4 ,100000)
print_result(diff_list)
test_non_shannon2(generate, non_shannon_5 ,100000)
print_result(diff_list)
test_non_shannon2(generate, non_shannon_6 ,100000)
print_result(diff_list)
test_non_shannon2(generate, non_shannon_7 ,100000)


# qubit
test_non_shannon1(generate_2, non_shannon_1 ,100000)
print_result(diff_list)
test_non_shannon1(generate_2, non_shannon_2 ,100000)
print_result(diff_list)
test_non_shannon1(generate_2, non_shannon_3 ,100000)
print_result(diff_list)
test_non_shannon1(generate_2, non_shannon_4 ,100000)
print_result(diff_list)
test_non_shannon1(generate_2, non_shannon_5 ,100000)
print_result(diff_list)
test_non_shannon1(generate_2, non_shannon_6 ,100000)
print_result(diff_list)
test_non_shannon1(generate_2, non_shannon_7 ,100000)
print_result(diff_list)

# qutrit
test_non_shannon2(generate_2, non_shannon_1 ,100000)
print_result(diff_list)
test_non_shannon2(generate_2, non_shannon_2 ,100000)
print_result(diff_list)
test_non_shannon2(generate_2, non_shannon_3 ,100000)
print_result(diff_list)
test_non_shannon2(generate_2, non_shannon_4 ,100000)
print_result(diff_list)
test_non_shannon2(generate_2, non_shannon_5 ,100000)
print_result(diff_list)
test_non_shannon2(generate_2, non_shannon_6 ,100000)
print_result(diff_list)
test_non_shannon2(generate_2, non_shannon_7 ,100000)


# qubit
test_non_shannon1(generate_3, non_shannon_1 ,100000)
print_result(diff_list)
test_non_shannon1(generate_3, non_shannon_2 ,100000)
print_result(diff_list)
test_non_shannon1(generate_3, non_shannon_3 ,100000)
print_result(diff_list)
test_non_shannon1(generate_3, non_shannon_4 ,100000)
print_result(diff_list)
test_non_shannon1(generate_3, non_shannon_5 ,100000)
print_result(diff_list)
test_non_shannon1(generate_3, non_shannon_6 ,100000)
print_result(diff_list)
test_non_shannon1(generate_3, non_shannon_7 ,100000)
print_result(diff_list)

# qutrit
test_non_shannon2(generate_3, non_shannon_1 ,100000)
print_result(diff_list)
test_non_shannon2(generate_3, non_shannon_2 ,100000)
print_result(diff_list)
test_non_shannon2(generate_3, non_shannon_3 ,100000)
print_result(diff_list)
test_non_shannon2(generate_3, non_shannon_4 ,100000)
print_result(diff_list)
test_non_shannon2(generate_3, non_shannon_5 ,100000)
print_result(diff_list)
test_non_shannon2(generate_3, non_shannon_6 ,100000)
print_result(diff_list)
test_non_shannon2(generate_3, non_shannon_7 ,100000)

# qubit
test_non_shannon1(generate_pure_state, non_shannon_1 ,100000)
print_result(diff_list)
test_non_shannon1(generate_pure_state, non_shannon_2 ,100000)
print_result(diff_list)
test_non_shannon1(generate_pure_state, non_shannon_3 ,100000)
print_result(diff_list)
test_non_shannon1(generate_pure_state, non_shannon_4 ,100000)
print_result(diff_list)
test_non_shannon1(generate_pure_state, non_shannon_5 ,100000)
print_result(diff_list)
test_non_shannon1(generate_pure_state, non_shannon_6 ,100000)
print_result(diff_list)
test_non_shannon1(generate_pure_state, non_shannon_7 ,100000)
print_result(diff_list)

# qutrit
test_non_shannon2(generate_pure_state, non_shannon_1 ,100000)
print_result(diff_list)
test_non_shannon2(generate_pure_state, non_shannon_2 ,100000)
print_result(diff_list)
test_non_shannon2(generate_pure_state, non_shannon_3 ,100000)
print_result(diff_list)
test_non_shannon2(generate_pure_state, non_shannon_4 ,100000)
print_result(diff_list)
test_non_shannon2(generate_pure_state, non_shannon_5 ,100000)
print_result(diff_list)
test_non_shannon2(generate_pure_state, non_shannon_6 ,100000)
print_result(diff_list)
test_non_shannon2(generate_pure_state, non_shannon_7 ,100000)



# TESTING GHZ STATES
p = GHZ_states(4,2)
print non_shannon_1_ghz(p,2)

# GRAPH
# x = [i for i in range(1, 10001)]
# y = diff_list
#
# plt.plot(x, y, color='purple', linewidth = 2,
#          marker='o', markerfacecolor='black', markersize=3)
#
# plt.xlabel('States generated from mixed state method 1')
# plt.ylabel('Difference')
# plt.show()
