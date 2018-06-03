import numpy as np
import math
import random

from numpy import linalg as LA
from shannon import randomProbabilityDist
from partial_trace import separate
from entropy import *
from utils import *


def new_eq1(pABC,dim):
    """
    Returns true if: I(A : C|B)  >= -log F(pABC, R_pBC,trC x I_A(pAB))
    where F is the Uhlmann's fidelity F(p,r) = ||sqrt(p) sqrt(r)||^2 _1
    from https://arxiv.org/pdf/1604.03023.pdf
    """

    # Ensure that system is a 3 qubit/qutrit quantum system
    check_n_q(pABC, dim, 3, "new_eq1 in entropy.py")

    # I(A:C|B) = H(A,B)-H(B)-H(A,B,C)-H(B,C)
    seps,joint,_ = separate(pABC,dim)
    pB = seps[1]
    pAB = joint[0]
    pBC = joint[1]
    H_ABC = vonNeumann(pABC)
    H_AB = vonNeumann(pAB)
    H_BC = vonNeumann(pBC)
    H_B = vonNeumann(pB)

    lhs = H_AB - H_B - H_ABC - H_BC

    # TODO: define R
    fidelity = fidelity(pABC, R)

    rhs = -np.log(fidelity)

    return lhs >= rhs

def fidelity(p,r):
    """
    Calculates fidelity between quantum states p and r
    F = ||sqrt(pr)||^2 _1
    """

    # Checks that p and r have same size and are both square
    check_same_size(p,r,"fidelity in entropy.py")

    sqrt_p = np.sqrt(p)
    p_r_p = sqrt_p*r*sqrt_p
    sqrt_prp = np.sqrt(p_r_p)

    values, _ = LA.eig(sqrt_prp)
    values = values.real

    f = np.sum(values)**2

    return f
