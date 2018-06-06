from matrix_generation_tests import *
from quantum_shannon_tests import *
from quantum_non_shannon_tests import *
from non_shannon_quantum import *
from evolution import *

# This script extensively tests functions i.e with all 2,3,4 qubit and qutrit
# states and a large amount of randomly generated states

# Tests n random unitary matrices are unitary
test_defs(generate_unitary, is_unitary, 1)

# Tests n random hermitian matrices are hermitian
test_defs(generate_hermitian, test_hermitian, 1)

# Tests n random density matrices are density matrices using different
# ways of generating mixed and pure states
test_defs(generate, test_density_matrix, 1)
test_defs(generate_2, test_density_matrix, 1)
#test_defs(generate_3, test_density_matrix, 1)
test_defs(generate_pure_state, test_density_matrix, 1)

# Testing von neumann entropy is non-negative
test_defs_true(generate, is_non_neg_VN, 1)
test_defs_true(generate_2, is_non_neg_VN, 1)
test_defs_true(generate_pure_state, is_non_neg_VN, 1)

# Testing von neumann entropy is less than log|X|
test_defs_true(generate, is_vn_leq_log, 1)
test_defs_true(generate_2, is_vn_leq_log, 1)
test_defs_true(generate_pure_state, is_vn_leq_log, 1)

# Test if random pure states have entropy of zero
test_defs(generate_pure_state, test_pure_state_entropy_is_zero, 1)

#Test if relative entropy is non negative
test_re(generate, test_relative_entropy_non_negative, 1)

# Test if relative entropy is monotonic
test_monotonic_re(generate, monotocity_relative_entropy, 1)

# Test if mutual information is less than 2xmin
test_bi_partite(generate, bound_mutual_information, 1)
test_bi_partite(generate_2, bound_mutual_information, 1)
test_bi_partite(generate_3, bound_mutual_information, 1)


# Test if mutual information is less than 2 log
test_bi_partite(generate, bound_mutual_information_log, 1)
test_bi_partite(generate_2, bound_mutual_information_log, 1)
test_bi_partite(generate_3, bound_mutual_information_log, 1)


# Test weak subadditivity holds
test_bi_partite(generate, weak_subadditivity, 1)
test_bi_partite(generate_2, weak_subadditivity, 1)
test_bi_partite(generate_3, weak_subadditivity, 1)
test_bi_partite(generate_pure_state, weak_subadditivity, 1)

# Test strong subadditivity
test_tri_partite(generate, strong_subadditivity_q, 1)
test_tri_partite(generate_2, strong_subadditivity_q, 1)
test_tri_partite(generate_3, strong_subadditivity_q, 1)
test_tri_partite(generate_pure_state, strong_subadditivity_q, 1)

# Test non shannpn entropies hold for extensive amount of quantum states
test_non_shannon(generate, non_shannon_1, 1)
test_non_shannon(generate_2, non_shannon_1, 1)
test_non_shannon(generate_3, non_shannon_1, 1)
test_non_shannon(generate_pure_state, non_shannon_1, 1)

test_non_shannon(generate, non_shannon_2, 1)
test_non_shannon(generate_2, non_shannon_2, 1)
test_non_shannon(generate_3, non_shannon_2, 1)
test_non_shannon(generate_pure_state, non_shannon_2, 1)

test_non_shannon(generate, non_shannon_3, 1)
test_non_shannon(generate_2, non_shannon_3, 1)
test_non_shannon(generate_3, non_shannon_3, 1)
test_non_shannon(generate_pure_state, non_shannon_3, 1)

test_non_shannon(generate, non_shannon_4, 1)
test_non_shannon(generate_2, non_shannon_4, 1)
test_non_shannon(generate_3, non_shannon_4, 1)
test_non_shannon(generate_pure_state, non_shannon_4, 1)

test_non_shannon(generate, non_shannon_5, 1)
test_non_shannon(generate_2, non_shannon_5, 1)
test_non_shannon(generate_3, non_shannon_5, 1)
test_non_shannon(generate_pure_state, non_shannon_5, 1)

test_non_shannon(generate, non_shannon_6, 1)
test_non_shannon(generate_2, non_shannon_6, 1)
test_non_shannon(generate_3, non_shannon_6, 1)
test_non_shannon(generate_pure_state, non_shannon_6, 1)

test_non_shannon(generate, non_shannon_7, 1)
test_non_shannon(generate_2, non_shannon_7, 1)
test_non_shannon(generate_3, non_shannon_7, 1)
test_non_shannon(generate_pure_state, non_shannon_7, 1)


# Test bit flip channel is positve trace preserving
test_PTP(generate, bit_flip_channel,1)
test_PTP(generate_2, bit_flip_channel,1)
test_PTP(generate_3, bit_flip_channel,1)
#test_PTP(generate_pure_state, bit_flip_channel,1)

# Test phase flip channel is positve trace preserving
test_PTP(generate, phase_flip_channel,1)
test_PTP(generate_2, phase_flip_channel,1)
test_PTP(generate_3, phase_flip_channel,1)
#test_PTP(generate_pure_state, phase_flip_channel ,1)

# Test bit phase flip channel is positve trace preserving
test_PTP(generate, bit_phase_flip_channel,1)
test_PTP(generate_2, bit_phase_flip_channel,1)
test_PTP(generate_3, phase_flip_channel,1)
#test_PTP(generate_pure_state, bit_phase_flip_channel ,1)

# Test bit flip channel is unital
test_unital(generate, bit_flip_channel, 1)
test_unital(generate_2, bit_flip_channel, 1)
test_unital(generate_3, bit_flip_channel, 1)
test_unital(generate_pure_state, bit_flip_channel, 1)

# Test phase flip channel is unital
test_unital(generate, phase_flip_channel, 1)
test_unital(generate_2, phase_flip_channel, 1)
test_unital(generate_3, phase_flip_channel, 1)
test_unital(generate_pure_state, phase_flip_channel, 1)

# Test bit phase flip channel is unital
test_unital(generate, bit_phase_flip_channel, 1)
test_unital(generate_2, bit_phase_flip_channel, 1)
test_unital(generate_3, bit_phase_flip_channel, 1)
test_unital(generate_pure_state, bit_phase_flip_channel, 1)

# Test depolarising_channel not unital
test_not_unital(generate, depolarising_channel, 1)
test_not_unital(generate_2, depolarising_channel, 1)
test_not_unital(generate_3, depolarising_channel, 1)
test_not_unital(generate_pure_state, depolarising_channel, 1)

# Test bit flip channel gives a higher entropy
test_entropy_more(generate, bit_flip_channel, 1)
test_entropy_more(generate_2, bit_flip_channel, 1)
test_entropy_more(generate_3, bit_flip_channel, 1)
#test_entropy_more(generate_pure_state, bit_flip_channel, 1)

# Test phase flip channel gives a higher entropy
test_entropy_more(generate, phase_flip_channel, 1)
test_entropy_more(generate_2, phase_flip_channel, 1)
test_entropy_more(generate_3, phase_flip_channel, 1)
#test_entropy_more(generate_pure_state, phase_flip_channel, 1)

# Test bit phase flip channel gives a higher entropy
test_entropy_more(generate, bit_phase_flip_channel, 1)
test_entropy_more(generate_2, bit_phase_flip_channel, 1)
test_entropy_more(generate_3, bit_phase_flip_channel, 1)
#test_entropy_more(generate_pure_state, bit_phase_flip_channel, 1)

# Test depolarising_channel gives higher entropy (may or may not)
test_entropy_more(generate, depolarising_channel, 1)
test_entropy_more(generate_2, depolarising_channel, 1)
test_entropy_more(generate_3, depolarising_channel, 1)
test_entropy_more(generate_pure_state, depolarising_channel, 1)

# Test unitary evolution keeps entropy constant
test_constant_entropy(generate, u_p_u, 1)
test_constant_entropy(generate_2, u_p_u, 1)
test_constant_entropy(generate_pure_state, u_p_u, 1)
