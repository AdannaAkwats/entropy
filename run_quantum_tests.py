from entropy import *
from generate_random_quantum import *
from evolution import *
from non_shannon_quantum import *

p_t = generate(8)
p = generate_2(8)

######### INEQUALITIES
def test_random_density_matrix_is_density_matrix_1():
    """
    Returns true if density matrix fulfils:
    - trace = 1
    - positive semi definite
    - hermitian
    """
    assert np.isclose(np.trace(p_t).real, 1) == True
    assert is_close_to_zero(np.trace(p_t).imag) == True
    assert is_positive_semi_def(p_t) == True
    assert is_hermitian(p_t) == True

def test_random_density_matrix_is_density_matrix_2():
    """
    Returns true if density matrix fulfils:
    - trace = 1
    - positive semi definite
    - hermitian
    """
    assert np.isclose(np.trace(p).real, 1) == True
    assert is_close_to_zero(np.trace(p).imag) == True
    assert is_positive_semi_def(p) == True
    assert is_hermitian(p) == True

def test_random_unitary_matrix_is_unitary():
    """
    Returns true if random unitary matrix generated is indeed unitary
    """
    u = generate_unitary(4)
    assert is_unitary(u) == True

def test_random_hermitian_matrix_is_hermitian():
    """
    Returns true if random hermitian matrix generated is indeed hermitian
    """
    h = generate_hermitian(6)
    assert is_hermitian(h) == True

def test_vonneumann_non_negative():
    """
    Returns true if vonNeumann entropy >= 0
    """
    assert is_non_neg_VN(p) == True

def test_vonneumann_less_than_log():
    """
    Returns true if vonNeumann entropy <= logd for a d-dim hibert space i.e
    H(X) <= log|X|
    """
    assert is_vn_leq_log(p) == True

def test_pure_state_entropy_is_zero():
    """
    Returns true if vonNeumann entropy of pure state is zero
    """
    pure = generate_pure_state_2(4)
    vn = vonNeumann(pure)
    assert is_close_to_zero(vn) == True

def test_relative_entropy_non_negative():
    """
    Return true if relative entropy is non-negative, with equality
    if and only if p = r
    """
    r = generate(8)
    assert is_non_neg_relative_entropy(p,r) == True

def test_monotonicity_of_relative_entropy():
    """
    Returns true if relative entropy is monotonic i.e.
    H(pA || rA) <= H(pAB || rAB)
    """
    p2 = generate(4)
    r = generate(4)
    assert monotocity_relative_entropy(p2, r, 2) == True

def test_mutual_information_less_than_min():
    """
    Returns true if mutual information is within bound
    0 <= I(A:B) <= 2min(H(A),H(B))
    """
    p2 = generate(4) # qubit
    q3 = generate(9) # qutrit
    assert bound_mutual_information(p2, 2) == True
    assert bound_mutual_information(q3, 3) == True

def test_mutual_information_less_than_log():
    """
    Returns true if mutual information is within bound
    I(A:B) <= 2log|A| and 2log|B|
    """
    p2 = generate(4) # qubit
    q3 = generate(9) # qutrit
    assert bound_mutual_information_log(p2, 2) == True
    assert bound_mutual_information_log(q3, 3) == True

def test_weak_subadditivity():
    """
    Returns true if weak subadditivity holds: H(A,B) <= H(A) + H(B)
    """
    p2 = generate_2(4) # qubit
    q3 = generate_2(9) # qutrit
    assert weak_subadditivity(p2,2) == True
    assert weak_subadditivity(q3,3) == True

def test_strong_subadditivity():
    """
    Returns true if strong subadditivity holds: H(A,B,C) + H(B) <= H(A, B)
    """
    assert strong_subadditivity_q(p,2) == True
    q3 = generate_2(3**3) # qutrit
    assert strong_subadditivity_q(q3,3) == True


########### NON SHANNON INEQUALITIES
p4 = generate(16)
def test_non_shannon_1():
    """
    Returns true if 2I(C:D) <= I(A:B) + I(A:C,D) + 3I(C:D|A) + I(C:D|B)
    """
    assert non_shannon_1(p4, 2) == True

def test_non_shannon_2():
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 3I(A:C|B) + 3I(B:C|A) + 2I(A:D) +2I(B:C|D)
    """
    assert non_shannon_2(p4, 2) == True

def test_non_shannon_3():
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + I(A:C|B) + 2I(B:C|A) + 3I(A:B|D) + I(B:D|A) + 2I(C:D)
    """
    assert non_shannon_3(p4, 2) == True

def test_non_shannon_4():
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 4I(B:C|A) + 2I(A:C|D) + I(A:D|C) + ...
    2I(B:D) + I(C:D|A)
    """
    assert non_shannon_4(p4, 2) == True

def test_non_shannon_5():
    """
    Returns true if:
    2I(A:B) <= 5I(A:B|C) + 3I(A:C|B) + I(B:C|A) + 2I(A:D) + 2I(B:C|D)
    """
    assert non_shannon_5(p4, 2) == True

def test_non_shannon_6():
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + 4I(A:C|B) + I(B:C|A) + 2I(A:D) + 3I(B:C|D) + I(C:D|B)
    """
    assert non_shannon_6(p4, 2) == True

def test_non_shannon_7():
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 2I(B:C|A) + 2I(A:B|D) + I(A:D|B) + ...
    I(B:D|A) + 2I(C:D)
    """
    assert non_shannon_7(p4, 2) == True


p5 = generate(3**4) # qutrit
def test_non_shannon_1_q3():
    """
    Returns true if 2I(C:D) <= I(A:B) + I(A:C,D) + 3I(C:D|A) + I(C:D|B)
    """
    assert non_shannon_1(p5, 3) == True

def test_non_shannon_2_q3():
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 3I(A:C|B) + 3I(B:C|A) + 2I(A:D) +2I(B:C|D)
    """
    assert non_shannon_2(p5, 3) == True

def test_non_shannon_3_q3():
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + I(A:C|B) + 2I(B:C|A) + 3I(A:B|D) + I(B:D|A) + 2I(C:D)
    """
    assert non_shannon_3(p5, 3) == True

def test_non_shannon_4_q3():
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 4I(B:C|A) + 2I(A:C|D) + I(A:D|C) + ...
    2I(B:D) + I(C:D|A)
    """
    assert non_shannon_4(p5, 3) == True

def test_non_shannon_5_q3():
    """
    Returns true if:
    2I(A:B) <= 5I(A:B|C) + 3I(A:C|B) + I(B:C|A) + 2I(A:D) + 2I(B:C|D)
    """
    assert non_shannon_5(p5, 3) == True

def test_non_shannon_6_q3():
    """
    Returns true if:
    2I(A:B) <= 4I(A:B|C) + 4I(A:C|B) + I(B:C|A) + 2I(A:D) + 3I(B:C|D) + I(C:D|B)
    """
    assert non_shannon_6(p5, 3) == True

def test_non_shannon_7_q3():
    """
    Returns true if:
    2I(A:B) <= 3I(A:B|C) + 2I(A:C|B) + 2I(B:C|A) + 2I(A:B|D) + I(A:D|B) + ...
    I(B:D|A) + 2I(C:D)
    """
    assert non_shannon_7(p5, 3) == True



####### EVOLUTION
q = generate(2)

def test_bit_phase_flip_channel_is_PTP():
    """
    Returns true if bit_phase_flip_channel is positive and trace preserving
    w.r.t p
    """
    assert is_PTP(bit_phase_flip_channel, q) == True

def test_phase_flip_channel_is_PTP():
    """
    Returns true if phase_flip_channel is positive and trace preserving
    w.r.t p
    """
    assert is_PTP(phase_flip_channel, q) == True

def test_bit_flip_channel_is_PTP():
    """
    Returns true if bit flip channel is positive and trace preserving
    w.r.t p
    """
    assert is_PTP(bit_flip_channel, q) == True

def test_bit_flip_channel_is_unital():
    """
    Returns true if bit_flip_channel is unital i.e op(I) = I,
    I being the identity
    """
    assert is_unital(bit_flip_channel, 2) == True

def test_phase_flip_channel_is_unital():
    """
    Returns true if phase_flip_channel is unital i.e op(I) = I,
    I being the identity
    """
    assert is_unital(phase_flip_channel, 2) == True

def test_bit_phase_flip_channel_is_unital():
    """
    Returns true if bit_phase_flip_channel is unital i.e op(I) = I,
    I being the identity
    """
    assert is_unital(bit_phase_flip_channel, 2) == True

def test_depolarising_channel_not_unital():
    """
    Returns true if depolarising_channel is not unital
    """
    assert is_unital(depolarising_channel, 2) == False

def test_bit_flip_channel_entropy_more():
    """
    Returns true if quantum operator bit_flip_channel increases entropy
    """
    prob = 0
    for i in range(11):
        assert is_CPTP_entropy_more(bit_flip_channel, q, prob) == True
        prob = prob + 0.1

def test_phase_flip_channel_entropy_more():
    """
    Returns true if quantum operator phase_flip_channel increases entropy
    """
    prob = 0
    for i in range(11):
        assert is_CPTP_entropy_more(phase_flip_channel, q, prob) == True
        prob = prob + 0.1


def test_bit_phase_flip_channel_entropy_more():
    """
    Returns true if quantum operator bit_phase_flip_channel increases entropy
    """
    prob = 0
    for i in range(11):
        assert is_CPTP_entropy_more(bit_phase_flip_channel, q, prob) == True
        prob = prob + 0.1

def test_is_entropy_constant():
    """
    Returns true if unitary evolution keeps entropy constant
    """
    u = generate_unitary(8)
    assert is_entropy_constant(u_p_u, p, u) == True
