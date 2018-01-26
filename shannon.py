import math
import numpy as np
from numpy import linalg as LA
import random

# Shannon entropy H(x) = -sum(P(x)log(P(x)))
def shannon(probs):
    # Check that probabilities given add up to one and are > 0 and < 1
    checkSum = 0
    for p in probs :
        if(p > 1 or p < 0) : # probabilities are between 0 and 1
            print("Error: Probabilities are not > 1 or < 0")
            return
        checkSum = checkSum + p

    if(checkSum != 1) :
        if(not abs(checkSum - 1.0) <= 1e-5):
            print("Error: Probabilities do not add to one")
            return

    # If probabilities are valid
    sumProbs = 0
    for p in list(probs) :
        sumProbs = sumProbs + p*math.log(p,2) # 2 represents bits
    return -sumProbs


# Generate a random probability distribution of n numbers
def randomProbabilityDist(n):
    probs = np.random.random(n)
    probs /= probs.sum()
    return probs


# Generate joint probability distribution Pxy matrix (n*n)
def pxy(n):
    Pxy = np.zeros((n,n))
    probs = randomProbabilityDist(n*n)
    k = 0
    for i in range(n):
        for j in range(n):
            Pxy[i][j] = probs[k];
            k += 1
    return Pxy


# Separate matrix Pxy to Px and Py
#def getPxAndPy(Pxy):
    # Px: sum each row ; Py: sum each column
    # ie for 2x2 matrix
    # Px = {P11 + P12, P21 + P22}
    # Py = {P11 + P21, P21 + P12}
#    Px = np.sum(Pxy, axis=1)
#    Py = np.sum(Pxy, axis=0)
#    return Px, Py


# Separate list pxy to Px and Py
def getPxPy(pxy):
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


# Returns H(XY), H(X) and H(Y)
def getAllShannon(Pxy):
    px, py = getPxPy(Pxy)
    H_XY = shannon(Pxy)
    H_X = shannon(px)
    H_Y = shannon(py)
    return H_XY, H_X, H_Y

# Returns true if H(X,Y) <= H(X) + H(Y)
def subadditivity(Pxy):
    H_XY, H_X, H_Y = getAllShannon(Pxy)
    return H_XY <= H_X + H_Y

# Returns entropy of X conditional on knowing Y: H(X|Y)
def conditionXY(Pxy):
    H_XY, _, H_Y = getAllShannon(Pxy)
    cond_HXY = H_XY - H_Y
    return cond_HXY

# Returns the mutual information content of X and Y: I(X:Y)
def mutualInformation(Pxy):
    H_XY, H_X, H_Y = getAllShannon(Pxy)
    I_XY = H_X + H_Y - H_XY
    return I_XY, I_XY >= 0


# Returns true if I(X:Y) <= H(Y)
def mutualInfo_leq_HY(Pxy):
    I_XY = mutualInformation(Pxy)
    _, _, H_Y = getAllShannon(Pxy)
    return I_XY <= H_Y

# Returns true if H(X|Y) <= H(X)
def cond_leq_HY(Pxy):
    cond_XY = conditionXY(Pxy)
    _, H_X, _ = getAllShannon(Pxy)
    return cond_XY <= H_X

# Returns true if H(X,Y) >= max[H(x), H(Y)]
def HXY_geq_max(Pxy) :
    H_XY, H_X, H_Y = getAllShannon(Pxy)
    return H_XY >= max(H_X, H_Y)


# Separate list pxyz to get pxy, pyz and pxz
def getPxPyPz(pxyz):
    # length of pxy is a cube
    # length of px, py, pz is the square of the cube root of pxy
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


# Returns true if H(X,Y,Z) + H(Y) <= H(X,Y) + H(Y,Z)
def strongSubadditivity(pxyz):
    pxy, pyz, _ = getPxPyPz(pxyz)
    _, py = getPxPy(pxy)
    H_XYZ = shannon(pxyz)
    H_XY = shannon(pxy)
    H_YZ = shannon(pyz)
    H_Y = shannon(py)
    return H_XYZ + H_Y <= H_XY + H_YZ
