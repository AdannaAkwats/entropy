import numpy as np
from numpy import linalg as LA
import math
import random
from entropy import Unitary
from entropy import vonNeumann
from utils import isclose

# Unitary Evolution
# ------------------------------------------------------------------------
# Quantum operator e
# e = UpU*
# Function takes a unitary matrix U and a density matrix p and calculates
# UpU* where U* is the complex conjugate transpose
def u_p_u(U, p):
    U_mat = np.matrix(U)
    # Q conjugate
    U_conj = U_mat.getH()

    # Multiply matrices
    Up = np.matmul(U, p)
    UpU_conj = np.matmul(Up, U_conj)

    return UpU_conj

# Unitary dynamic evolution U = exp{iHt} where H = Hamitonian and t = time
# def dyn_evol_U(t):


# Takes a function that evolves p (op) and checks if the entropy stays constant
def isEntropyConstant(op, p, U):
    # Evolve p
    evolved_p = op(U, p)
    H_p = vonNeumann(p)
    H_ev = vonNeumann(evolved_p);

    print "Original density matrix entropy " + str(H_p)
    print "Unitary evolved entropy " + str(H_ev)

    return isclose(H_p, H_ev)
