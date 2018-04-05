import numpy as np
import numpy.matlib
from numpy import linalg as LA
import math
import random
from entropy import Unitary
from entropy import vonNeumann
from utils import isclose
from utils import isPowerof2
from utils import isPowerof3


# Unitary Evolution
# ------------------------------------------------------------------------

def u_p_u(U, p):
    """
    Quantum operator e
    e = UpU*
    Function takes a unitary matrix U and a density matrix p and calculates
    UpU* where U* is the complex conjugate transpose
    """

    U_mat = np.matrix(U)
    # Q conjugate
    U_conj = U_mat.getH()

    # Multiply matrices
    Up = np.matmul(U, p)
    UpU_conj = np.matmul(Up, U_conj)

    return UpU_conj


def dyn_evol_U(t, H):
    """
    Unitary dynamic evolution U = exp{iHt} where H = Hamitonian
    and t = time
    """

    return math.exp(1j*t*H)


def depolarising_channel(Q, prob):
    """
    Models state of quantum system after noise:
    E(p) = (prob * I)/d + (1-prob)*Q
    where Q is the quantum system and prob is the probability that
    the d-dimensional quantum system is depolarised
    """

    # We will focus on only qubit and qutrit
    dim = Q.shape[0]
    d = 2
    if(isPowerof3(dim)):
        d = 3
    elif(not isPowerof2(dim)):
        print "Error in Function 'depolarising_channel in evolution.py':"
        print "Density matrix given is not a qubit or qutrit system."
        sys.exit()

    I = np.matlib.identity(dim)
    E = (prob/d) * I

    return E + (1 - prob) * Q


def bit_flip_channel(Q, prob):
    """
    The bit flip channel flips the state of a qubit from 0 to 1 and vice versa
    with probability 1 - prob
    """

    I = np.matlib.identity(2)
    E_0 = math.sqrt(prob) * I
    II = np.zeros((2,2))
    II[1,0] = 1
    II[0,1] = 1
    E_1 = math.sqrt(1-prob) * II

    E_0_mat = np.matrix(E_0)
    E_1_mat = np.matrix(E_1)

    E = E_0.dot(Q).dot(E_0_mat.getH()) + E_1.dot(Q).dot(E_1_mat.getH())

    return E


def phase_flip_channel(Q, prob):
    """
    The phase flip channel changes the phase of the state of the qubit
    """

    I = np.matlib.identity(2)
    E_0 = math.sqrt(prob) * I
    II = np.zeros((2,2))
    II[0,0] = 1
    II[1,1] = -1
    E_1 = math.sqrt(1-prob) * II

    E_0_mat = np.matrix(E_0)
    E_1_mat = np.matrix(E_1)

    E = E_0.dot(Q).dot(E_0_mat.getH()) + E_1.dot(Q).dot(E_1_mat.getH())

    # Testing ... p = 1/2
    # P_0 = np.zeros((2,2))
    # P_0[0,0] = 1
    # P_1 = np.zeros((2,2))
    # P_1[1,1] = 1
    # E_pr = P_0.dot(Q).dot(P_0) + P_1.dot(Q).dot(P_1)

    return E


def bit_phase_flip_channel(Q, prob):
    """
    This is a combination of a phase flip and a bit flip
    """

    I = np.matlib.identity(2)
    E_0 = math.sqrt(prob) * I
    II = np.zeros((2,2))
    II[1,0] = -1j
    II[1,0] = 1j
    E_1 = math.sqrt(1-prob) * II

    E_0_mat = np.matrix(E_0)
    E_1_mat = np.matrix(E_1)

    E = E_0.dot(Q).dot(E_0_mat.getH()) + E_1.dot(Q).dot(E_1_mat.getH())

    return E


def isEntropyConstant(op, p, U):
    """
    Takes a function that evolves p (op) and checks if the entropy
    stays constant
    """

    # Evolve p
    evolved_p = op(U, p)
    H_p = vonNeumann(p)
    H_ev = vonNeumann(evolved_p);

    print "Original density matrix entropy " + str(H_p)
    print "Unitary evolved entropy " + str(H_ev)

    return isclose(H_p, H_ev)


def isCPTPEntropyMore(op, p, prob):
    """
    Check that CPTP (Completely positve trace preserving)
    quantum operator obeys inequatlity H(E(p)) >= H(p)
    """
    # Evolve p
    evolved_p = op(p, prob)
    H_e = vonNeumann(evolved_p)
    print "H_e: " + str(H_e)
    H_p = vonNeumann(p)
    print "H_p: " + str(H_p)

    return H_e >= H_p


# def is_CPTP(E):
#     """
#     Checks if operator E is completely positive and trace preserving
#     """
