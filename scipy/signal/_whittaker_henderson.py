import numpy as np
from scipy.sparse.linalg import splu
from scipy.sparse import identity
import scipy.sparse as sparse
from scipy.sparse import spdiags
import matplotlib.pyplot as plt

from scipy.sparse import random
from scipy import stats
from numpy.random import default_rng
import pandas as pd
from numpy import nan

import timeit
start = timeit.default_timer()

# y = 10*(np.random.rand(100))

x = np.array([65,  72,  76,  79,  95, 115, 119, 124, 128, 134, 138, 139, 139,
              139, 140, 141, 138, 135, 125, 118, 122, 114,  89,  52,  45,  40,
              40,  45,  48,  47,  46,  48,  49,  50,  52,  58,  62,  62,  61,
              67,  78,  87,  87,  87,  93, 102, 108, 116, 101, 102, 113, 111,])

y1 = np.array([143, 136, 132, 130, 130, 130, 129, 129, 130, 132, 126, 119, 122,
        112,  87,  43,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,
          45,  46,  50,  52,  53,  59,  74,  89,  93,  91,  90,  90,  89,
          91,  88, 100, 115, 123, 131, 130, 118, 106, 100,  99,  90,  77])

y2 = np.array([143, 136, 132, 130, 130, 130, 129, 129, 130, 132, 126, 119, 122,
        112,  87,  43,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
          45,  46,  50,  52,  53,  59,  74,  89,  93,  91,  90,  90,  89,
          91,  88, 100, 115, 123, 131, 130, 118, 106, 100,  99,  90,  77])

y=y1

d=3
lmbd = 100
def whihen(y, d, lmbd):
    """
    Parameters
    ----------
    y : array_like
        The data to be smoothed
    d : int
        Order of the differences, 
    lmbd : float
        Lambda, the larger lambda is, the smoother the z will be.

    Returns
    -------
    z : array_like
        Smoothed series
        
    References
    ----------
    Eilers, Paul H. C. “A Perfect Smoother.” Analytical Chemistry, vol. 75, 
    no. 14, 30 May 2003, pp. 3631–3636, 10.1021/ac034173t. 
    Accessed 30 May 2022.

    """
    
    # E = identity(m)
    
    """
    Calculate the d-th discrete difference in each row of the matrix.
    This part of the function is the same as numpy.diff() 
    However, numpy.diff() does not accept csc format, and resulting in slower
    execution.
    """
    m = len(y)
    
    shape = (m-d, m)
    diagonals = np.zeros(2*d + 1)
    diagonals[d] = 1.
    for i in range(d):
        diff = diagonals[:-1] - diagonals[1:]
        diagonals = diff
    offsets = np.arange(d+1)    
    D = sparse.diags(diagonals, offsets, shape, format="csc")

    """
    Handle missing data points.
    Generate weight array from input data, element without value is convert to 
    0 and with value is 1.
    """
    S = np.where(np.isnan(y), y, 1)
    S = np.where(~np.isnan(S), S, 0)
    print(S)
    W = spdiags(S, 0, m, m)

    C = W + lmbd * D.T @ D
    
    """
    scipy.splu does not able to handle NaN, so convert to 0.
    """
    y = np.where(~np.isnan(y), y, 0)
    z = splu(C).solve(y)

    return z

result = whihen(y, d, lmbd)

plt.plot(y, "*")
plt.plot(result)
plt.ylim(0)
plt.xlim(0)
plt.show()

plt.scatter(x, y)
plt.ylim(0)
plt.xlim(0)
plt.show()

stop = timeit.default_timer()
print('Time: ', stop - start)  