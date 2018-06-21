from quantum_tests import *
from non_shannon_quantum import *
from evolution import *

# This script extensively tests functions i.e with all 2,3,4 qubit and qutrit
# states and a large amount of randomly generated states

lim = 1000
# Tests n random unitary matrices are unitary
test_defs(generate_unitary, is_unitary, lim)

# Tests n random hermitian matrices are hermitian
test_defs(generate_hermitian, test_hermitian, lim)

# Tests n random density matrices are density matrices using different
# ways of generating mixed and pure states
test_defs(generate, test_density_matrix, lim)
test_defs(generate_2, test_density_matrix, lim)
#test_defs(generate_3, test_density_matrix, lim)
test_defs(generate_pure_state, test_density_matrix, lim)

# Testing von neumann entropy is non-negative
test_defs_true(generate, is_non_neg_VN, lim)
test_defs_true(generate_2, is_non_neg_VN, lim)
test_defs_true(generate_pure_state, is_non_neg_VN, lim)

# Testing von neumann entropy is less than log|X|
test_defs_true(generate, is_vn_leq_log, lim)
test_defs_true(generate_2, is_vn_leq_log, lim)
test_defs_true(generate_pure_state, is_vn_leq_log, lim)

# Test if random pure states have entropy of zero
test_defs(generate_pure_state, test_pure_state_entropy_is_zero, lim)

#Test if relative entropy is non negative
test_re(generate, test_relative_entropy_non_negative, lim)

# Test if relative entropy is monotonic
test_monotonic_re(generate, monotocity_relative_entropy, lim)

# Test if mutual information is less than 2xmin
test_bi_partite(generate, bound_mutual_information, lim)
test_bi_partite(generate_2, bound_mutual_information, lim)
test_bi_partite(generate_3, bound_mutual_information, lim)


# Test if mutual information is less than 2 log
test_bi_partite(generate, bound_mutual_information_log, lim)
test_bi_partite(generate_2, bound_mutual_information_log, lim)
test_bi_partite(generate_3, bound_mutual_information_log, lim)


# Test weak subadditivity holds
test_bi_partite(generate, weak_subadditivity, lim)
test_bi_partite(generate_2, weak_subadditivity, lim)
test_bi_partite(generate_3, weak_subadditivity, lim)
test_bi_partite(generate_pure_state, weak_subadditivity, lim)


# Test triangle_inequality holds
test_bi_partite(generate, triangle_inequality, lim)
test_bi_partite(generate_2, triangle_inequality, lim)
test_bi_partite(generate_3, triangle_inequality, lim)
test_bi_partite(generate_pure_state, triangle_inequality, lim)

# Test conditional triangle_inequality holds
test_tri_partite(generate, cond_triangle_inequality, lim)
test_tri_partite(generate_2, cond_triangle_inequality, lim)

# Test conditional reduces entropy holds
test_tri_partite(generate, cond_reduce_entropy, lim)
test_tri_partite(generate_2, cond_reduce_entropy, lim)
test_tri_partite(generate_3, cond_reduce_entropy, lim)
test_tri_partite(generate_pure_state, cond_reduce_entropy, lim)

# Test mutual_information does not increase
test_tri_partite(generate, mutual_info_not_increase, lim)
test_tri_partite(generate_2, mutual_info_not_increase, lim)
test_tri_partite(generate_3, mutual_info_not_increase, lim)
test_tri_partite(generate_pure_state, mutual_info_not_increase, lim)

# Test strong subadditivity
test_tri_partite(generate, strong_subadditivity_q, lim)
test_tri_partite(generate_2, strong_subadditivity_q, lim)
test_tri_partite(generate_3, strong_subadditivity_q, lim)
test_tri_partite(generate_pure_state, strong_subadditivity_q, lim)

# Test subadditivity of conditional entropy lim
test_4_partite(generate, subadditivity_of_cond_1, lim)
test_4_partite(generate_2, subadditivity_of_cond_1, lim)
test_4_partite(generate_3, subadditivity_of_cond_1, lim)
test_4_partite(generate_pure_state, subadditivity_of_cond_1, lim)

# Test subadditivity of conditional entropy 2
test_tri_partite(generate, subadditivity_of_cond_2, lim)
test_tri_partite(generate_2, subadditivity_of_cond_2, lim)
test_tri_partite(generate_3, subadditivity_of_cond_2, lim)
test_tri_partite(generate_pure_state, subadditivity_of_cond_2, lim)

# Test subadditivity of conditional entropy 2
test_tri_partite(generate, subadditivity_of_cond_3, lim)
test_tri_partite(generate_2, subadditivity_of_cond_3, lim)
test_tri_partite(generate_3, subadditivity_of_cond_3, lim)
test_tri_partite(generate_pure_state, subadditivity_of_cond_3, lim)

# Test conditonal strong subadditivity
test_4_partite(generate, cond_strong_subadditivity, lim)
test_4_partite(generate_2, cond_strong_subadditivity, lim)
test_4_partite(generate_3, cond_strong_subadditivity, lim)
test_4_partite(generate_pure_state, cond_strong_subadditivity, lim)

lim = 100000
# Test non shannpn entropies hold for extensive amount of quantum states
test_non_shannon(generate, non_shannon_1, lim)
test_non_shannon(generate_2, non_shannon_1, lim)
test_non_shannon(generate_3, non_shannon_1, lim)
test_non_shannon(generate_pure_state, non_shannon_1, lim)

test_non_shannon(generate, non_shannon_2, lim)
test_non_shannon(generate_2, non_shannon_2, lim)
test_non_shannon(generate_3, non_shannon_2, lim)
test_non_shannon(generate_pure_state, non_shannon_2, lim)

test_non_shannon(generate, non_shannon_3, lim)
test_non_shannon(generate_2, non_shannon_3, lim)
test_non_shannon(generate_3, non_shannon_3, lim)
test_non_shannon(generate_pure_state, non_shannon_3, lim)

test_non_shannon(generate, non_shannon_4, lim)
test_non_shannon(generate_2, non_shannon_4, lim)
test_non_shannon(generate_3, non_shannon_4, lim)
test_non_shannon(generate_pure_state, non_shannon_4, lim)

test_non_shannon(generate, non_shannon_5, lim)
test_non_shannon(generate_2, non_shannon_5, lim)
test_non_shannon(generate_3, non_shannon_5, lim)
test_non_shannon(generate_pure_state, non_shannon_5, lim)

test_non_shannon(generate, non_shannon_6, lim)
test_non_shannon(generate_2, non_shannon_6, lim)
test_non_shannon(generate_3, non_shannon_6, lim)
test_non_shannon(generate_pure_state, non_shannon_6, lim)

test_non_shannon(generate, non_shannon_7, lim)
test_non_shannon(generate_2, non_shannon_7, lim)
test_non_shannon(generate_3, non_shannon_7, lim)
test_non_shannon(generate_pure_state, non_shannon_7, lim)


lim = 1000
# Test bit flip channel is positve trace preserving
test_PTP(generate, bit_flip_channel,lim)
test_PTP(generate_2, bit_flip_channel,lim)
test_PTP(generate_3, bit_flip_channel,lim)
#test_PTP(generate_pure_state, bit_flip_channel,lim)

# Test phase flip channel is positve trace preserving
test_PTP(generate, phase_flip_channel,lim)
test_PTP(generate_2, phase_flip_channel,lim)
test_PTP(generate_3, phase_flip_channel,lim)
#test_PTP(generate_pure_state, phase_flip_channel ,lim)

# Test bit phase flip channel is positve trace preserving
test_PTP(generate, bit_phase_flip_channel,lim)
test_PTP(generate_2, bit_phase_flip_channel,lim)
test_PTP(generate_3, phase_flip_channel,lim)
#test_PTP(generate_pure_state, bit_phase_flip_channel ,lim)

# Test bit flip channel is unital
test_unital(generate, bit_flip_channel, lim)
test_unital(generate_2, bit_flip_channel, lim)
test_unital(generate_3, bit_flip_channel, lim)
test_unital(generate_pure_state, bit_flip_channel, lim)

# Test phase flip channel is unital
test_unital(generate, phase_flip_channel, lim)
test_unital(generate_2, phase_flip_channel, lim)
test_unital(generate_3, phase_flip_channel, lim)
test_unital(generate_pure_state, phase_flip_channel, lim)

# Test bit phase flip channel is unital
test_unital(generate, bit_phase_flip_channel, lim)
test_unital(generate_2, bit_phase_flip_channel, lim)
test_unital(generate_3, bit_phase_flip_channel, lim)
test_unital(generate_pure_state, bit_phase_flip_channel, lim)

# Test depolarising_channel not unital
test_not_unital(generate, depolarising_channel, lim)
test_not_unital(generate_2, depolarising_channel, lim)
test_not_unital(generate_3, depolarising_channel, lim)
test_not_unital(generate_pure_state, depolarising_channel, lim)

# Test bit flip channel gives a higher entropy
test_entropy_more(generate, bit_flip_channel, lim)
test_entropy_more(generate_2, bit_flip_channel, lim)
test_entropy_more(generate_3, bit_flip_channel, lim)

# Test phase flip channel gives a higher entropy
test_entropy_more(generate, phase_flip_channel, lim)
test_entropy_more(generate_2, phase_flip_channel, lim)
test_entropy_more(generate_3, phase_flip_channel, lim)

# Test bit phase flip channel gives a higher entropy
test_entropy_more(generate, bit_phase_flip_channel, lim)
test_entropy_more(generate_2, bit_phase_flip_channel, lim)
test_entropy_more(generate_3, bit_phase_flip_channel, lim)


# Test depolarising_channel gives higher entropy (may or may not)
test_entropy_more(generate, depolarising_channel, lim)
test_entropy_more(generate_2, depolarising_channel, lim)
test_entropy_more(generate_3, depolarising_channel, lim)
test_entropy_more(generate_pure_state, depolarising_channel, lim)

# Test unitary evolution keeps entropy constant
test_constant_entropy(generate, u_p_u, lim)
test_constant_entropy(generate_2, u_p_u, lim)
test_constant_entropy(generate_pure_state, u_p_u, lim)
