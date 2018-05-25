import math
import numpy as np
from numpy import linalg as LA
import random
from utils import isclose

def shannon(probs):
    """
    Returns Shannon entropy H(x) = -sum(P(x)log(P(x)))
    """
    # Check that probabilities given add up to one and are > 0 and < 1
    checkSum = np.sum(probs)

    if(np.any(probs > 1) or np.any(probs < 0)) : # probabilities are between 0 and 1
        print "Error in Function 'shannon in shannon.py':"
        print("Error: Probabilities are not > 1 or < 0")
        sys.exit()

    if(not isclose(checkSum,1)) :
        print "Error in Function 'shannon in shannon.py':"
        print("Error: Probabilities do not add to one")
        sys.exit()

    # If probabilities are valid,
    # 2 represents bits
    v = probs*np.log2(probs)
    return -np.sum(v)


def randomProbabilityDist(n):
    """
    Generate a random probability distribution of n numbers
    """
    probs = np.random.random(n)
    probs /= probs.sum()
    return probs


def pxy(n):
    """
    Generate joint probability distribution Pxy matrix (n*n)
    """
    Pxy = np.zeros((n,n))
    probs = randomProbabilityDist(n*n)
    k = 0
    for i in range(n):
        for j in range(n):
            Pxy[i][j] = probs[k];
            k += 1
    return Pxy


def getPxPy(pxy):
    """
    Separate list pxy to Px and Py
    """
    # length of px, py square root of the length pxy
    n = len(pxy) ** (1. / 2)
    n = int(n)
    px = np.zeros(n)
    py = np.zeros(n)

    k = 0
    eachX = 0
    for i in range(len(pxy)):
        # Fill up py
        for j in range(n):
            if((i+1) % n == j):
                py[j] += pxy[i]

        # Fill up px
        eachX += pxy[i]
        if((i+1) % n == 0):
            px[k] = eachX
            eachX = 0
            k += 1

    return px, py


def getAllShannon(Pxy):
    """
    Returns H(XY), H(X) and H(Y)
    """
    px, py = getPxPy(Pxy)
    H_XY = shannon(Pxy)
    H_X = shannon(px)
    H_Y = shannon(py)
    return H_XY, H_X, H_Y


def subadditivity(Pxy):
    """
    Returns true if H(X,Y) <= H(X) + H(Y)
    """
    H_XY, H_X, H_Y = getAllShannon(Pxy)
    return H_XY <= H_X + H_Y


def conditionXY(Pxy):
    """
    Returns entropy of X conditional on knowing Y: H(X|Y)
    """

    H_XY, _, H_Y = getAllShannon(Pxy)
    cond_HXY = H_XY - H_Y
    return cond_HXY

def mutualInformation(Pxy):
    """
    Returns the mutual information content of X and Y: I(X:Y)
    """
    H_XY, H_X, H_Y = getAllShannon(Pxy)
    I_XY = H_X + H_Y - H_XY
    return I_XY, I_XY >= 0


def mutualInfo_leq_HY(Pxy):
    """
    Returns true if I(X:Y) <= H(Y)
    """
    I_XY = mutualInformation(Pxy)
    _, _, H_Y = getAllShannon(Pxy)
    return I_XY <= H_Y

def mutualInfo_leqMin(Pxy):
    """
    Returns true if I(X:Y) <= min(H(X),H(Y))
    """
    I_XY = mutualInformation(Pxy)
    _, H_X, H_Y = getAllShannon(Pxy)
    return I_XY <= np.minimum(H_X,H_Y)

def cond_leq_HY(Pxy):
    """
    Returns true if H(X|Y) <= H(X)
    """

    cond_XY = conditionXY(Pxy)
    _, H_X, _ = getAllShannon(Pxy)
    return cond_XY <= H_X


def HXY_geq_max(Pxy) :
    """
    Returns true if H(X,Y) >= max[H(x), H(Y)]
    """
    H_XY, H_X, H_Y = getAllShannon(Pxy)
    return H_XY >= max(H_X, H_Y)


def getPxPyPz(pxyz):
    """
    Separate list pxyz to get pxy, pyz and pxz
    """

    # length of pxy is a cube
    # length of pxy, pyz, pxz is the square of the cube root of pxyz
    n = len(pxyz) ** (1. / 3)
    lp = int(n ** 2)
    pxy = np.zeros(lp)
    pyz = np.zeros(lp)
    pxz = np.zeros(lp)

    n = int(n)

    k = 0
    eachXY = 0
    l = 0
    for i in range(len(pxyz)):
        # Fill up pxz
        for j in range(lp):
            if((i+1) % lp == j):
                pxz[j] += pxyz[i]

        if((i+1) % lp == 0):
            l += 1

        # Fill up pxy
        eachXY += pxyz[i]
        if((i+1) % n == 0):
            pxy[k] = eachXY
            eachXY = 0
            k += 1

        # Fill up pyz
        for r in range(n):
            if((i % n) == r):
               pyz[i % lp] += pxyz[i]

    return pxy, pyz, pxz


def strongSubadditivity(pxyz):
    """
    Returns true if H(X,Y,Z) + H(Y) <= H(X,Y) + H(Y,Z)
    """

    pxy, pyz, _ = getPxPyPz(pxyz)
    _, py = getPxPy(pxy)
    H_XYZ = shannon(pxyz)
    H_XY = shannon(pxy)
    H_YZ = shannon(pyz)
    H_Y = shannon(py)
    return H_XYZ + H_Y <= H_XY + H_YZ


# TODO
def getPxyzw(pxyzw):
    """
    Separate list pxyzw to get pxyz, pyzw, pxzw and pxyw
    """

    # length of pxyz is a to the power of 4
    # length ofpxyz, pyzw, pxzw and pxyw is the cube of the 4th  root of pxyzw
    n = len(pxyz) ** (1. / 4)
    lp = int(n ** 3)
    pxyz = np.zeros(lp)
    pxyw = np.zeros(lp)
    pyzw = np.zeros(lp)
    pxzw = np.zeros(lp)

    n = int(n)

    k = 0
    eachXYZ = 0
    l = 0
    for i in range(len(pxyzw)):
        # Fill up pxz
        for j in range(lp):
            if((i+1) % lp == j):
                pxz[j] += pxyz[i]

        if((i+1) % lp == 0):
            l += 1

        # Fill up pxyz
        eachXYZ += pxyzw[i]
        if((i+1) % n == 0):
            pxyz[k] = eachXYZ
            eachXYZ = 0
            k += 1

        # Fill up pyz
        for r in range(n):
            if((i % n) == r):
               pyz[i % lp] += pxyz[i]

    return pxyz, pxyw, pyzw, pxzw
