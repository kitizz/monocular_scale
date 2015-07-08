import numpy as np

from scipy import ndimage


def whereFirst(x):
    '''
    Find the first occurence of a non-zero value in x.
    :param x: 1D numpy array
    :return: Index of first non-zero element. -1 if there are none.
    '''
    for i in xrange(len(x)):
        if x[i] != 0:
            return i

    return -1


def goldenSectionSearch(f, a, b, tol=1e-3):
    '''
    Do a golden section search over function, f(x). The function should return
    2 values: a residual and a result.
    residual, result = f(x)

    Credit: From Wikipedia, http://en.wikipedia.org/wiki/Golden_section_search
    '''
    gr = 0.618  # Golden ratio
    x = b - gr*(b - a)
    y = a + gr*(b - a)
    while abs(x - y) > tol:
        fx, rx = f(x)
        fy, ry = f(y)
        if fx < fy:
            b = y
            y = x  # fy = fx; fx = f(x)
            x = b - gr*(b - a)
        else:
            a = x
            x = y  # fx = fy; fy = f(y)
            y = a + gr*(b - a)
    return (b + a)/2.0


def indexDilation(inds, dilateSize, maxLength=-1):
    if len(inds) == 0:
        return []

    if maxLength <= 0:
        maxLength = inds.max() + dilateSize

    logicInd = np.zeros(maxLength, dtype=bool)
    dilateStruc = np.ones(dilateSize, dtype=bool)
    logicInd[inds] = True

    logicInd = ndimage.morphology.binary_dilation(logicInd, dilateStruc)
    return np.where(logicInd)[0]


def boundValues(source, target, result='subarray'):
    '''
    Given 2 sorted arrays, return a subarray of source whose values
    are bounded by the min and max values of the target.

    :param result: Select what kind of result to receive. 'subarray' will return
        a subarray of source. 'bool' will return a boolean selection array for
        source.
    '''
    minV, maxV = target[0], target[-1]
    s, e = 0, source.shape[0] - 1

    while s <= e:
        if source[s] >= minV:
            break
        s += 1

    while e >= s:
        if source[e] <= maxV:
            break
        e -= 1

    if result == 'bool':
        result = np.zeros(source.shape, bool)
        result[s:e] = True
        return result
    else:
        return source[s:e]


def normalize(v):
    return v/np.linalg.norm(v)


def clamp(val, minv, maxv):
    ''' Clamp val to be between minv and maxv, inclusive. '''
    return max(minv, min(maxv, val))


def generalizedESD(x, maxOLs, alpha=0.05, fullOutput=False):
    """
    Carry out a Generalized ESD Test for Outliers.

    This code is straight from PyAstronomy

    The Generalized Extreme Studentized Deviate (ESD) test for
    outliers can be used to search for outliers in a univariate
    data set, which approximately follows a normal distribution.
    A description of the algorithm is, e.g., given at
    `Nist <http://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm>`_
    or [Rosner1983]_.

    Parameters
    ----------
    maxOLs : int
        Maximum number of outliers in the data set.
    alpha : float, optional
        Significance (default is 0.05).
    fullOutput : boolean, optional
        Determines whether additional return values
        are provided. Default is False.

    Returns
    -------
    Number of outliers : int
        The number of data points characterized as
        outliers by the test.
    Indices : list of ints
        The indices of the data points found to
        be outliers.
    R : list of floats, optional
        The values of the "R statistics". Only provided
        if `fullOutput` is set to True.
    L : list of floats, optional
        The lambda values needed to test whether a point
        should be regarded an outlier. Only provided
        if `fullOutput` is set to True.

    """
# The MIT License (MIT)
# Copyright (c) 2014 PyAstronomy
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
    from scipy.stats import t
    if maxOLs < 1:
        raise(Exception("Maximum number of outliers, `maxOLs`, must be > 1."))
    import numpy.ma as ma
    xm = ma.array(x)
    n = len(xm)
    # Compute R-values
    R = []
    L = []
    minds = []
    for i in xrange(maxOLs + 1):
        # Compute mean and std of x
        xmean = xm.mean()
        xstd = xm.std()
        # Find maximum deviation
        rr = np.abs((xm - xmean)/xstd)
        minds.append(np.argmax(rr))
        R.append(rr[minds[-1]])
        if i >= 1:
            p = 1.0 - alpha/(2.0*(n - i + 1))
            perPoint = t.ppf(p, n-i-1)
            L.append((n-i)*perPoint / np.sqrt((n-i-1+perPoint**2) * (n-i+1)))
        # Mask that value and proceed
        xm[minds[-1]] = ma.masked
    # Remove the first entry from R, which is of no meaning for the test
    R.pop(-1)
    # Find the number of outliers
    ofound = False
    for i in xrange(maxOLs-1, -1, -1):
        if R[i] > L[i]:
            ofound = True
            break
    # Prepare return value
    if ofound:
        if not fullOutput:
            # There are outliers
            return i+1, minds[0:i+1]
        else:
            return i+1, minds[0:i+1], R, L, minds
    else:
        # No outliers could be detected
        if not fullOutput:
            # There are outliers
            return 0, []
        else:
            return 0, [], R, L, minds


def findOutliers(data, alpha=0.05, k=-1, axis=-1):
    ''' Find outliers in the provided data.

    '''
    # Originally written by Bob Newell, February 1996
    # Modified by Jaco de Groot, May 2006
    # Bob Newell used a fixed value for lambda. This script calculates the
    # critical values for lambda based on the equations in
    # "Quality control of semi-continuous mobility size-fractionated
    #    particle number concentration data",
    # Atmospheric Environment 38 (2004) 33413348, Rong Chun Yu,*,
    #     Hee Wen Teh, Peter A. Jaques, Constantinos Sioutas, John R. Froines)
    # -----------------------------------------------------
    # Modified by Paul J. Ganssle, June 2012
    # Ported from Matlab code to Python by Christopher Ham

    if axis > 1:
        raise Exception('findOutliers only handles 1D or 2D arrays.')

    data = data.astype(float)

    if len(data.shape) == 1:
        N = len(data)
        M = 1
        data = np.array([data]).astype(float)
        axis = 1

    elif axis < 0:
        axis = 1
        N = data.shape[1]
        M = data.shape[0]

    else:
        N = int(data.shape[axis])
        M = int(data.shape[(axis + 1) % 2])

    if k < 0: k = N-2

    outliers = []

    for i in range(M):
        y = data[i]
        outliers.append(np.array(generalizedESD(y, k, alpha=alpha)[1]))

    if M == 1:  # No need to stick it in a list...
        return outliers[0]
    else:
        return np.array(outliers, int)
