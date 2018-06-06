from entropy import *
from generate_random_quantum import *
import time

def test_unitary(u):
    """
    Returns true if u is unitary
    """
    assert is_unitary(u) == True

def test_hermitian(h):
    """
    Returns true if h is hermitian
    """
    assert is_hermitian(h) == True

def test_density_matrix(d):
    """
    Returns true if d is a density matrix
    """
    # Hermitian and positive semi definite
    assert is_hermitian(d) == True
    assert is_positive_semi_def(d) == True
    # Trace = 1
    assert np.isclose(np.trace(d).real, 1) == True
    assert is_close_to_zero(np.trace(d).imag) == True


# TEST
def test_defs(gen_func, test_func, lim):
    start_time = time.clock()
    for i in range(lim):
        p2 = gen_func(2**2) # 2-qubit state
        test_func(p2)
        p3 = gen_func(2**3) # 3-qubit state
        test_func(p3)
        p4 = gen_func(2**4) # 4-qubit state
        test_func(p4)
        p5 = gen_func(2**5) # 5-qubit state
        test_func(p5)

        q2 = gen_func(3**2) # 2-qutrit state
        test_func(q2)
        q3 = gen_func(3**3) # 3-qutrit state
        test_func(q3)
        q4 = gen_func(3**4) # 4-qutrit state
        test_func(q4)
        q5 = gen_func(3**5) # 5-qutrit state
        test_func(q5)

    print("%d of each of 2 - 5 qubit and qutrit states tested" % (lim))
    print("%s: --- PASSED in %s seconds ---" % (test_func, time.clock() - start_time))


# TEST
def test_defs_true(gen_func, test_func, lim):
    start_time = time.clock()
    for i in range(lim):
        p2 = gen_func(2**2) # 2-qubit state
        assert test_func(p2) == True
        p3 = gen_func(2**3) # 3-qubit state
        assert test_func(p3) == True
        p4 = gen_func(2**4) # 4-qubit state
        assert test_func(p4) == True
        p5 = gen_func(2**5) # 5-qubit state
        assert test_func(p5) == True

        q2 = gen_func(3**2) # 2-qutrit state
        assert test_func(q2) == True
        q3 = gen_func(3**3) # 3-qutrit state
        assert test_func(q3) == True
        q4 = gen_func(3**4) # 4-qutrit state
        assert test_func(q4) == True
        q5 = gen_func(3**5) # 5-qutrit state
        assert test_func(q5) == True


    print("%d of each of 2 - 5 qubit and qutrit states tested" % (lim))
    print("%s: --- PASSED in %s seconds ---" % (test_func, time.clock() - start_time))
