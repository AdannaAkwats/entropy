import numpy as np
import math
import random
import sys

from entropy import vonNeumann
from partial_trace import separate
from generate_random_quantum import *


def bell_states(n):
    """
    Returns Bell's states
    n = 1: Return (|00> + |11>)/sqrt(2)
    n = 2: Return (|00> - |11>)/sqrt(2)
    n = 3: Return (|01> + |10>)/sqrt(2)
    n = 4: Return (|01> - |10>)/sqrt(2)
    """
    p = np.zeros((4,4))
    if(n == 1):
        p[0,0] = 0.5
        p[0,3] = 0.5
        p[3,0] = 0.5
        p[3,3] = 0.5
    elif(n == 2):
        p[0,0] = 0.5
        p[0,3] = -0.5
        p[3,0] = -0.5
        p[3,3] = 0.5
    elif(n == 3):
        p[1,1] = 0.5
        p[1,2] = 0.5
        p[2,1] = 0.5
        p[2,2] = 0.5
    elif(n == 4):
        p[1,1] = 0.5
        p[1,2] = -0.5
        p[2,1] = -0.5
        p[2,2] = 0.5
    else:
        print("Error in function 'bell_states' in entropy.py")
        print("bell state number given is not valid.")
        sys.exit()
    return p

def GHZ_states(M, dim):
    """
    Returns GHZ states |u> = 1/sqrt(2) (|0>^M + |1>^M), where M>= 3
    """
    if(M < 3):
        print("Error in function 'GHZ_states' in entropy.py")
        print("M should be more than or equal to 3")
        sys.exit()

    n = dim**M
    p = np.zeros((n,n))
    p[0,0] = 0.5
    p[0,n-1] = 0.5
    p[n-1,0] = 0.5
    p[n-1,n-1] = 0.5

    return p


def is_entangled(pAB, pB):
    """
    Returns true if p is entangled in A : B. If H(AB) - H(B) < 0
    """
    H_AB = vonNeumann(pAB)
    H_B = vonNeumann(pB)

    return (H_AB - H_B) < 0


def is_bell_state_max_entangled(n):
    """
    Returns true if bell states are maximally entangled
    0 < n <= 4 represents bell states; go to function bell_states in entropy.py
    """
    p = bell_states(n)
    V = vonNeumann(p)
    if(not np.isclose(V,0)): # Bell states are pure states
        return False

    s,_,_,_ = separate(p,2)

    # In bell state, separated pA = pB
    if(not is_entangled(p, s[0])):
        return False

    # Maximally entangled if H = log d
    H = vonNeumann(s[0])
    if(H == np.log2(s[0].shape[0])):
        return True
    return False


def mixed_entangled_bipartite(gen_func, lim, dim):
    """
    Return number of mixed states and number of mixed entangled states out of
    the n bipartite states generated
    """

    ent = 0
    for i in range(lim):
        p = gen_func(dim**2)
        s,_,_,_ = separate(p, dim)

        if(is_entangled(p, s[1])): #p and pB
            ent = ent + 1

    return lim, ent, lim-ent


def mixed_entangled_joint(gen_func, size, dim, cut, lim):
    """
    size: 3 (tri- partite), 4, 5 partite system
    Return number of mixed states and number of mixed entangled states out of
    the lim 3 4 or 5 partite states generated.
    cut = 1 -> entanglement between pAB|CDE
    cut = 2 -> entanglement between pABC|DE
    """
    ent = 0
    func = is_entangled_ABC
    if (size == 4):
        func = is_entangled_ABCD
    elif(size == 5):
        func = is_entangled_5
    elif(size != 3):
        print("Error in function 'mixed_entangled_joint' in entropy.py")
        print("size given is not valid.")
        sys.exit()

    for i in range(lim):
        p = gen_func(dim**size)
        if(func(p, dim, cut)):
            ent = ent + 1

    return lim, ent, lim-ent



def is_entangled_ABC(pABC, dim, cut):
    """
    Returns true if pABCD is entangled with cut s.t
    cut 1 - pA|BC : H(ABC) - H(A) < 0
    cut 2 - pAB|C : H(ABC) - H(AB) < 0
    """
    s, j, _,_ = separate(pABC, dim)
    if(cut == 1):
        pA = s[0]
        return is_entangled(pABC, pA)
    elif(cut == 2):
        pAB = j[0]
        return is_entangled(pABC, pAB)
    else:
        print("Error in function 'is_entangled_ABC'")
        print("Cut given is not valid.")
        sys.exit()


def is_entangled_ABCD(pABCD, dim, cut):
    """
    Returns true if pABCD is entangled with cut s.t
    cut 1 - pA|BCD : H(ABCD) - H(A) < 0
    cut 2 - pAB|CD : H(ABCD) - H(AB) < 0
    """
    s, j, j3,_ = separate(pABCD, dim)
    if(cut == 1):
        pA = s[0]
        return is_entangled(pABCD, pA)
    elif(cut == 2):
        pAB = j[0]
        return is_entangled(pABCD, pAB)
    else:
        print("Error in function 'is_entangled_ABC'")
        print("Cut given is not valid.")
        sys.exit()


def is_entangled_5(pABCDE, dim, cut):
    """
    Returns true if pABCD is entangled with cut s.t
    cut 1 - pAB|CDE : H(ABCDE) - H(AB) < 0
    cut 2 - pABC|DE : H(ABCDE) - H(ABC) < 0
    """
    s, j, j3,j4 = separate(pABCDE, dim)
    if(cut == 1):
        pA = j[0]
        return is_entangled(pABCDE, pA)
    elif(cut == 2):
        pAB = j3[0]
        return is_entangled(pABCDE, pAB)

    else:
        print("Error in function 'is_entangled_ABC' in entropy.py")
        print("Cut given is not valid.")
        sys.exit()
