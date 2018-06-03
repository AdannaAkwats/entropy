from entropy import *
from generate_random_quantum import *
from test_quantum_non_shannon import *
from evolution import *

p = generate(8)

######### INEQUALITIES
def test_random_unitary_matrix_is_unitary():
    u = generate_unitary(4)
    assert is_unitary(u) == True

def test_random_hermitian_matrix_is_hermitian():
    h = generate_hermitian(6)
    assert is_hermitian(h) == True

def test_vonneumann_non_negative():
    # Returns true if vonNeumann entropy >= 0
    assert is_non_neg_VN(p) == True

def test_relative_entropy_non_negative():
    # Return true if relative entropy is non-negative, with equality
    # if and only if p = r
    r = generate(8)
    assert is_non_neg_relative_entropy(p,r) == True

def test_monotonicity_of_relative_entropy():
    # Returns true if relative entropy is monotonic i.e.
    # H(pA || rA) <= H(pAB || rAB)
    p2 = generate(4)
    r = generate(4)
    assert monotocity_relative_entropy(p2, r, 2)

def test_mutual_information_less_than_min():
    # Returns true if mutual information is within bound
    # 0 <= I(A:B) <= 2min(H(A),H(B))
    p2 = generate(4)
    assert bound_mutual_information(p2, 2)

# def test_mutual_information_less_than_log():
#     # Returns true if mutual information is within bound
#     # I(A:B) <= 2log|A| and 2log|B|
#     p2 = generate(4)
#     assert bound_mutual_information_log(p2, 2)

def test_weak_subadditivity():
    # Returns true if weak subadditivity holds: H(A,B) <= H(A) + H(B)
    p2 = generate(4)
    assert weak_subadditivity(p2,2)

def test_strong_subadditivity():
    # Returns true if strong subadditivity holds: H(A,B,C) + H(B) <= H(A, B)
    assert strong_subadditivity_q(p,2)


########### NON SHANNON INEQUALITIES
def test_non_shannon_eqs():
    # Returns true if:
    # EQ1: 2I(C:D) <= I(A:B) + I(A:C,D) + 3I(C:D|A) + I(C:D|B)
    # EQ2: 2I(A:B) <= 3I(A:B|C) + 3I(A:C|B) + 3I(B:C|A) + 2I(A:D) +2I(B:C|D)
    # EQ3: 2I(A:B) <= 4I(A:B|C) + I(A:C|B) + 2I(B:C|A) + 3I(A:B|D) + I(B:D|A) + 2I(C:D)
    # EQ4: 2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 4I(B:C|A) + 2I(A:C|D) + I(A:D|C) + ...
    # 2I(B:D) + I(C:D|A)
    # EQ5: 2I(A:B) <= 5I(A:B|C) + 3I(A:C|B) + I(B:C|A) + 2I(A:D) + 2I(B:C|D)
    # EQ6: 2I(A:B) <= 4I(A:B|C) + 4I(A:C|B) + I(B:C|A) + 2I(A:D) + 3I(B:C|D) + I(C:D|B)
    # EQ7: 2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 2I(B:C|A) + 2I(A:B|D) + I(A:D|B) + ...
    # I(B:D|A) + 2I(C:D)

    p4 = generate(16)
    assert non_shannon_eqs_q(p4,2,0) == [True, True, True, True, True, True, True]


####### EVOLUTION
q = generate(2)

def test_bit_phase_flip_channel_is_PTP():
    # Returns true if bit_phase_flip_channel is positive and trace preserving
    # w.r.t p
    assert is_PTP(bit_phase_flip_channel, q) == True

def test_phase_flip_channel_is_PTP():
    # Returns true if phase_flip_channel is positive and trace preserving
    # w.r.t p
    assert is_PTP(phase_flip_channel, q) == True

def test_bit_flip_channel_is_PTP():
    # Returns true if bit flip channel is positive and trace preserving
    # w.r.t p
    assert is_PTP(bit_flip_channel, q) == True

def test_bit_flip_channel_is_unital():
    # Returns true if bit_flip_channel is unital i.e op(I) = I,
    # I being the identity
    assert is_unital(bit_flip_channel, 2) == True

def test_phase_flip_channel_is_unital():
    # Returns true if phase_flip_channel is unital i.e op(I) = I,
    # I being the identity
    assert is_unital(phase_flip_channel, 2) == True

def test_bit_phase_flip_channel_is_unital():
    # Returns true if bit_phase_flip_channel is unital i.e op(I) = I,
    # I being the identity
    assert is_unital(bit_phase_flip_channel, 2) == True

def test_depolarising_channel_not_unital():
    # Returns true if depolarising_channel is not unital
    assert is_unital(depolarising_channel, 2) == False

def test_bit_flip_channel_entropy_more():
    # Returns true if quantum operator bit_flip_channel increases entropy
    prob = 0
    for i in range(11):
        assert is_CPTP_entropy_more(bit_flip_channel, q, prob) == True
        prob = prob + 0.1

def test_phase_flip_channel_entropy_more():
    # Returns true if quantum operator phase_flip_channel increases entropy
    prob = 0
    for i in range(11):
        assert is_CPTP_entropy_more(phase_flip_channel, q, prob) == True
        prob = prob + 0.1


def test_bit_phase_flip_channel_entropy_more():
    # Returns true if quantum operator bit_phase_flip_channel increases entropy
    prob = 0
    for i in range(11):
        assert is_CPTP_entropy_more(bit_phase_flip_channel, q, prob) == True
        prob = prob + 0.1

def test_is_entropy_constant():
    # Returns true if unitary evolution keeps entropy constant
    u = generate_unitary(8)
    assert is_entropy_constant(u_p_u, p, u) == True
