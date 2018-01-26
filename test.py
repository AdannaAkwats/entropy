import numpy as np
from numpy import linalg as LA
import math
import random

from shannon import randomProbabilityDist
from shannon import pxy
from shannon import shannon
from shannon import subadditivity
from shannon import getPxPy
from shannon import getPxPyPz
from shannon import strongSubadditivity


# Testing output...
p = randomProbabilityDist(3**2)
print(p)
px, py = getPxPy(p)
print(px)
print(py)
