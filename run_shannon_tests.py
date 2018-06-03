import numpy as np
from shannon import *
from test_non_shannon import non_shannon_eqs


# Generate random Probability distribution to test with
pxy = randomProbabilityDist(4)

# def test_shannon_less_than_log():
#     # Returns true if H(X) <= log|X|
#     assert shannon_leq_log(pxy)

def test_shannon_subadditivity():
    # Returns true if H(X,Y) <= H(X) + H(Y)
    assert subadditivity(pxy) == True

def test_shannon_mutual_information_less_than_y():
    # Returns true if I(X:Y) <= H(Y)
    assert mutualInfo_leq_HY(pxy) == True

def test_shannon_mutual_information_less_than_min():
    # Returns true if I(X:Y) <= min(H(X),H(Y))
    assert mutualInfo_leqMin(pxy) == True

def test_shannon_mutual_information_less_than_min():
    # Returns true if I(X:Y) <= log|X| and log|Y|
    assert mutualInfo_leq_log(pxy) == True

def test_shannon_conditional_less_than_y():
    # Returns true if H(X|Y) <= H(X)
    assert cond_leq_HY(pxy) == True

def test_shannon_greater_than_max():
    # Returns true if H(X,Y) >= max[H(x), H(Y)]
    assert HXY_geq_max(pxy) == True

p3 = randomProbabilityDist(8)

def test_shannon_subadditivity():
    # Returns true if H(X,Y,Z) + H(Y) <= H(X,Y) + H(Y,Z)
    assert strongSubadditivity(p3) == True

p4 = randomProbabilityDist(16)

def test_non_shannon_eq_1():
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

    assert non_shannon_eqs(p4,0) == [True, True, True, True, True, True, True]
