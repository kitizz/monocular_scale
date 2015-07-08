from __future__ import division, print_function

import numpy as np
from numpy import linalg, dot, zeros, ones
from numpy.linalg import norm
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy import signal

import UtilScale as Util
import IO
import Plotting

import sys


class SolverException(BaseException):
    pass


defaultConfig = {
    # Cutoff is the one and only real magic number...
    'cutoff': 1.5,  # Hz
    'blurType': 'LowPass',

    'initialDelay': 0.,
    'minDelay': 0,
    'maxDelay': 0.5,
    'doDelaySearch': True,
    'delaySearchSize': 7,

    'solveLinearBias': True,
    # If constrain gravity is <= 0, its magnitude will be unconstrained
    'constrainGravity': 0,
    'objective': 'LeastSquares',
    # If constrain scale is a list/tuple of length, 2. It will use the 1st and
    # 2nd elements as min and max constraints, respectively.
    'constrainScale': [],

    'useL2L1': False,
    'useAdmm': False,
    'plotAdmm': False,

    'trimSides': 2.,  # seconds

    'firstSecondsOfData': np.inf,
    'alignWithFirstSeconds': np.inf,
    'sampleEachNthFrame': 1,

    'skipped': np.empty(0, int),
    'outliers': np.empty(0, int),

    'removeSkipped': True,
    'removeOutliers': False,
    'outlierThreshold': 0.05,

    'verbose': False
}


def solveConstrained(A, b, g, stopFilter,
                     x=None,
                     rho=1e-4, mu=1.01, maxRho=10.0,
                     eps=10e-6, minIter=10, maxIter=500,
                     verbose=False):
    '''
    Use an ADMM to solve:
    argmin_x  || Ax - b ||_2^2
    s.t. x in G
    g(x) is a function that performs a Euclidean projection of x onto set G.
    stopFilter(x, last_x) returns a delta. When this value between
    iterations is < eps, the algorithm stops.
    '''
    N, M = A.shape

    # Precompute to save some time.
    AA = np.dot(A.T, A)
    Ab = np.dot(A.T, b)
    I = np.eye(M)

    def update1(z, u):
        return np.linalg.lstsq(AA + rho*I, Ab + rho*(z - u))[0]

    def update2(x, u):
        return g(x + u)

    def update3(x, z, u):
        return u + x - z

    if x is not None:
        u = np.zeros(M)
        z = x
    else:
        u = np.random.standard_normal(M)
        z = np.random.standard_normal(M)

    i = 0
    debounce = 4
    db = debounce
    # stopVar = stopFilter(u)
    last_x = u

    while i < maxIter:
        x = update1(z, u)
        z = update2(x, u)
        u = update3(x, z, u)

        # Stopping criteria
        deltaVar = stopFilter(x, last_x)
        last_x = x
        if abs(deltaVar) < eps:
            db -= 1
        else:
            db = debounce

        # Update rho
        rho = min(mu*rho, maxRho)

        i += 1
        if verbose and (i - 1) % 10 == 0:
            print("%d: \trho: %.5g, \tscale: %g, delta: %.5g, " % (i, rho, x[-1], deltaVar))

        if db <= 0 and i >= minIter and g(x)[-1] == x[-1]:
            break

    return x


def solveL21Constrained(A, B, g, stopFilter,
                        x=None,
                        rho=1e-4, mu=1.01, maxRho=10.0,
                        eps=10e-6, minIter=10, maxIter=500,
                        verbose=False):
    '''
    Use an ADMM to solve:
    argmin_x  sum_{f=1}^F || A_f x - b_f ||_21
    s.t. x in G
    g(x) is a function that performs a Euclidean projection of x onto set G.
    stopFilter(x, last_x) returns a delta. When this value between
    iterations is < eps, the algorithm stops.

    :param A: FxNxM matrix where F is the number of sums/subproblems, N is the
        amount of data in each subproblem, and M is the number of variables to solve.
    :param B: FxN matrix (b_f is the fth row of B).
    '''
    F, N, M = A.shape
    A_flat = A.reshape(F*N, M)
    b = B.reshape(-1)  # Stack the rows into a vector

    # Precompute to save some time.
    AA_I = np.dot(A_flat.T, A_flat) + np.eye(M)
    # Ab = np.dot(A_flat.T, b)
    # I = np.eye(M)

    def update_x(u, v, y, z):
        # return np.linalg.lstsq(AA_I, A_flat.T.dot(b + v + u/rho) + z - y/rho)[0]
        return np.linalg.lstsq(AA_I, A_flat.T.dot(b + v + u) + z - y)[0]

    def update_z(x, y):
        # return g(x + y/rho)
        return g(x + y)

    def update_V(x, U):
        # M = A.dot(x) - B - U/rho
        M = A.dot(x) - B - U
        M_norm = linalg.norm(M, axis=1).reshape(-1, 1)
        return np.maximum(1 - 1./(rho*M_norm), 0)*M

    def update_y(x, y, z):
        # return rho*(z - x)
        return y + x - z

    def update_U(V, x, B):
        # return rho*(A.dot(x) - B - V)
        return U + V - A.dot(x) + B

    if x is not None:
        y = zeros(M)
        z = x
        U = zeros((F, N))
        V = A.dot(x) - B

    else:
        y = np.random.standard_normal(M)
        z = np.random.standard_normal(M)
        U = np.random.standard_normal((F, N))
        V = np.random.standard_normal((F, N))

    i = 0
    debounce = 4
    db = debounce
    # stopVar = stopFilter(u)
    last_x = z

    while i < maxIter:
        x = update_x(U.reshape(-1), V.reshape(-1), y, z)
        z = update_z(x, y)
        V = update_V(x, U)
        y = update_y(x, y, z)
        U = update_U(V, x, B)

        # Stopping criteria
        deltaVar = stopFilter(x, last_x)
        last_x = x
        if abs(deltaVar) < eps:
            db -= 1
        else:
            db = debounce

        # Update rho
        rho = min(mu*rho, maxRho)

        i += 1
        if verbose and (i - 1) % 10 == 0:
            print("%d: \trho: %.5g, \tscale: %g, delta: %.5g, " % (i, rho, x[-1], deltaVar))

        if db <= 0 and i >= minIter and g(x)[-1] == x[-1]:
            if verbose:
                print("%d (break): \trho: %.5g, \tscale: %g, delta: %.5g, " %
                      (i, rho, x[-1], deltaVar))
            break

    return x


def SolveLS(camData, imuData, chirpData=None, config={}, lastInternals=None):
    ''' Solve for the scale, bias, and gravity given the camData and imuData.

    This method makes a gaussian noise assumption on the input data. It also
    also assumes that the accelerometer bias is varying linearly with time.

    Assumes that the timestamps of the camData and imuData have a common origin, and that the accelerations are in the same coordinate frame.

    :param camData: A `CameraData` object containing the transforms and time stamps of the camera. The `accel` field should also be set.
    :param imuData: An `IMUData` object containing all the sensor data and timestamps of the IMU.
    :param delay: The number samples by which the IMU data should be delayed.
    '''
    config = _defaultConfig(config)

    imuAccel = imuData.accel.data
    imuTimes = imuData.accel.times

    camAccel = camData.accel  # Mx3 array
    camTimes = camData.times

    # Get the sampling periods
    Pimu = np.mean( np.diff(imuTimes) )
    Pcam = np.mean( np.diff(camTimes) )

    P = max(Pcam, Pimu)

    # Blur + Downsample the IMU acceleration to be the same length as camAccel
    if Pcam > Pimu:  # IMU sampling rate is higher
        sel = Util.boundValues(camTimes, imuTimes, result='bool')
        times = camTimes[sel]

        skipped = config['skipped'] - Util.whereFirst(sel)

        camAccel = camData.accel[sel]
        gravComp = camData.R[sel]

        camPos = camData.transforms[sel,:3,3]

        imuAccel = downsample(times, imuTimes, imuData.accel.data, axis=0)

    else:  # Cam sampling rate is higher
        sel = Util.boundValues(imuTimes, camTimes, result='bool')
        times = imuTimes[sel]

        skip_bool = zeros(len(camTimes))
        skip_bool[config['skipped']] = True
        skipped = np.where( downsample(times, camTimes, skip_bool) > 0 )[0]

        camAccel = downsample(times, camTimes, camData.accel, axis=0)
        gravComp = zeros((len(times), 3, 3))
        for i in range(3):
            for j in range(3):
                gravComp[:, i, j] = downsample(times, camTimes, camData.R[:, i, j])

        camPos = downsample(times, camTimes, camData.transforms[:,:3,3], axis=0)

        imuAccel = imuData.accel.data[sel]

    # M is the number of samples we are now working with
    M = len(times)

    if lastInternals is None:
        outliersInt = np.empty(0, int)
    else:
        outliersInt = lastInternals['outliers']

    # Need to offset 'skipped' to reflect new data
    removeIndices = np.r_[skipped, outliersInt]
    removeIndices = np.unique(removeIndices)

    # Construct the 'A' matrix
    # Full set of columns correspond to: [ bx by bz tbx tby tbz gx gy gz s ]

    # First construct the gravity rotation matrix.
    # The acceleration values are organised like so: [x1 x2 .. xn y1 y2 .. zn]
    # The gravity rotation matrix should transform the gravity in the vision
    # coord frame to the phone body frame: [gx1 gx2 ... gzn] = G * [gx gy gz]
    # The following transpose operation gets the inverse rotation of the camera
    # in vision space, and also groups the axis outputs together so that, for
    # example, the X outputs take up the first third if the matrix (see paper).

    # Contruct the weighting matrix
    keep = ones(M, bool)
    # keep = set( np.arange(len(b_imu)) )
    trim = int(config['trimSides']/P)
    firstN = min(config['firstSecondsOfData']/P, M)
    keep[:trim] = False
    keep[-trim-1:] = False
    keep[firstN:] = False
    keep[removeIndices] = False

    nKeep = keep.sum()

    len_u = 10  # At least [ bias, linear bias, grav, scale ]

    # Voila the memory
    A_full = zeros((M, 3, len_u))  # Cube!

    for i in range(3):
        # The constant bias components are simply ones, blocked diagonalized
        A_full[:, i, i] = 1
        # The linear bias components are monotonically increasing
        A_full[:, i, 3+i] = times

    # The scale component is for the vectorized camera acceleration, blurred.
    A_full[:, :, -1] = camAccel

    # Let the rotation at each from be: R_f = [rx_f, ry_f, rz_f]
    # The following reshaping gives:
    # R_lowpass = [ rx_1, ..., rx_F, ry_1, ..., ry_F, ..., rz_F ]^T
    A_full[:, :, -4:-1] = gravComp.transpose(0,2,1)

    # Construct b vector
    B = imuAccel

    # Weight the matrices (by removing unwanted entries)
    A_keep, B_keep = A_full[keep], B[keep]
    # The order='F' thing means the matrices are reshaped column-by-column
    # This would be the default behaviour for Armadillo...
    A = A_keep.reshape(-1, len_u, order='F')
    b = B_keep.reshape(-1, order='F')

    if nKeep < len_u:
        print("Warning (SolveLS): No useful data points left.")
        u = np.zeros(len_u, float)
        u[-1] = 0.
        res = np.inf

    else:
        u = np.linalg.lstsq(A, b)[0]  # Solve

        gMag = config['constrainGravity']
        scaleRange = config['constrainScale']

        def constrainU(u):
            u = u.copy()

            # Constrain magnitude of gravity
            if gMag > 0:
                u[-4:-1] = gMag*Util.normalize(u[-4:-1])

            # Constrain the bias of each IMU axis
            for i in range(3):
                u[i] = np.clip(u[i], -3.0, 3.0)

            # Constrain scale
            if len(scaleRange) == 2:
                u[-1] = np.clip(u[-1], scaleRange[0], scaleRange[1])
                # print "Clamp scale: %f -> %f" % (s, u[-1])

            return u

        def stopFilter(u, last_u):
            return (u[-1] - last_u[-1])/u[-1]

        if gMag > 0 or len(scaleRange) == 2:
            u = solveL21Constrained(A_keep, B_keep, constrainU, stopFilter,
                x=constrainU(u),
                rho=1e-3, mu=1.03, eps=1e-9, maxIter=2e3, maxRho=1e8)

    res = np.mean( np.abs(A.dot(u) - b) )

    bias = u[0:3]
    scale = u[-1]
    biasLinear = u[3:6]
    uPrint = {'scale':scale, 'bias':bias, 'biasLinear': biasLinear}

    grav = u[-4:-1]
    uPrint['grav'] = grav

    internals = {
        'A': A_full.reshape(-1, len_u, order='F'),
        'camOut': A_full.dot(u).reshape(-1,3).T,
        'imuOut': imuAccel.T,
        'u': u,
        'b': B.reshape(-1, order='F'),
        'useful': np.tile(keep, (3,1)),
        'selCam': sel,
        'res': res,
        'uPrint': uPrint,
        'biasLinear': biasLinear,
        'bias': bias,
        'camPos': camPos,
        'grav': grav,
        'scale': scale,
        'delaySeconds': imuData.totalOffset,
        'times': times
    }
    internals['err'] = internals['camOut'] - internals['imuOut']

    return bias, biasLinear, grav, scale, internals


def FindScale(camData, imuData, chirpData=None, config={}, lastInternals=None, showPlot=False):
    ''' Given camData and imuData, find the metric scale of the camera frame.

    :param camData: CameraData object.
    :param imuData: IMUData object.
    :param Rci: The rotation matrix of the IMU's alignment with respect to
        the camera body frame.
    :param sigma: The level of blurring to apply to the data. The one and only
        magic number
    :param showPlot: If true, a plot of the results will be shown.
    :return: (scale, internals) The best matching scale, and a dictionary of
        internally calculated values then may be useful in development.
    '''
    config = _defaultConfig(config)

    camData = camData.copy()
    imuData = imuData.copy()

    verbose = config['verbose']

    Pimu = np.mean( np.diff(imuData.accel.times) )
    Pcam = np.mean( np.diff(camData.times) )

    # Work out if the camera needs resampling
    sampling = config['sampleEachNthFrame']
    camData.transforms = camData.transforms[::sampling]
    camData.times = camData.times[::sampling]
    Pcam *= sampling

    M = len(camData.times)

    # Apply low pass to camera rotation transform. Note that this makes them
    # invalid SO3 transforms. But keeps the objective mathematically consistent
    # with low passing, and downsampling of Cam/IMU
    camR = camData.transforms[:, 0:3, 0:3]
    camR = blur(camR, camData.times, axis=0,
                method=config['blurType'],
                cutoff=config['cutoff'])
    camData.R = camR

    imuData.accel.data = blur(imuData.accel.data, imuData.accel.times, axis=0,
                              method=config['blurType'],
                              cutoff=config['cutoff'])

    # Get the camera acceleration
    camPos = camData.transforms[:, 0:3, 3]  # Becomes Mx3 position array
    diff2 = np.r_[1., -2., 1.]/(Pcam**2)  # 1st-order 2nd derivative
    # diff2 = np.r_[-1., 16, -30, 16, -1]/(12*Pcam**2)  # 2nd order 2nd derivative
    camData.accel = ndimage.convolve1d(camPos, diff2, axis=0)

    for i in range(M):
        R = camData.transforms[i, 0:3, 0:3]
        camData.accel[i] = np.dot(R.T, camData.accel[i])

    camData.accel = blur(camData.accel, camData.times, axis=0,
                         method=config['blurType'],
                         cutoff=config['cutoff'])

    camSkipped = np.array(Util.indexDilation(camData.skipped, 220))

    if config['removeSkipped']:
        config['skipped'] = camSkipped

    def assessAtDelay(offset):
        lastInternals = None
        imuData.setOffset(-offset)

        for j in range(2):
            bias, biasLinear, grav, scale, internals = \
                SolveLS(camData, imuData, chirpData, config=config, lastInternals=lastInternals)

            internals['delay'] = imuData.totalOffset
            showResultsNutmeg(internals)

            res = internals['res']

            if j == 1 or res == np.inf or not config['removeOutliers']:
                return res, internals

            mask = internals['useful'].any(axis=0)
            err = norm(internals['err'], axis=0)
            origInds = np.where(mask)[0]
            err = err[origInds]  # Ignore existing outliers

            k = int(min(100, 0.1*len(err)))
            alpha = config['outlierThreshold']
            inds = Util.findOutliers(err, alpha=alpha, k=k)

            if len(inds) > 0:
                inds = Util.indexDilation(origInds[inds], 31,
                                          internals['err'].shape[1])

            internals['outliers'] = inds.astype(int)
            lastInternals = internals

        return res, internals

    # Let's aim for sub frame accuracy
    if config['doDelaySearch']:
        minOffset = config['minDelay']
        maxOffset = config['maxDelay']
        threshold = 0.25*min(Pcam, Pimu)
        offset = Util.goldenSectionSearch(assessAtDelay, minOffset, maxOffset, tol=threshold)

    else:
        offset = config['initialDelay']

    # Do a final execution to extract the internals
    res, internals = assessAtDelay(offset)
    internals['delaySeconds'] = -offset

    if verbose:
        print('Best Res, Offset: %f, %f' %
              (res, offset))
        printResult(internals['uPrint'])

    if showPlot:
        showResults(internals)

    return internals['scale'], internals


def downsample(t, tp, yp, axis=-1):
    ''' Blur and downsample the signal, yp, by interpolation.

    :param t: 1D array-like object. The times of the desired output signal.
    :param tp: 1D array-like object. The times of the input signal.
    :param yp: 1D or 2D array-like object. If it is 2D, you must provide an axis along which to do the downsample.
    :param axis: The axis along which the downsample is executed.
    :return: A 1D or 2D array with the downsampled and blurred signal.
    '''
    if len(yp.shape) > 1 and (axis < 0 or axis > 1):
        raise Exception('Axis must be 0 or 1 for 2D input')

    if axis == -1:
        return np.interp(t, tp, yp)

    N = len(t)  # Length of output signal
    M = yp.shape[(axis+1) % 2]  # Number of signals to process

    out = np.zeros((M, N))

    # Axis play
    if axis == 0:
        yp = yp.T

    for ax in range(M):
        out[ax] = np.interp(t, tp, yp[ax])

    # Undo axis transpose
    if axis == 0:
        out = out.T

    return out


def printResult(uPrint):
    if len(uPrint.keys()) == 0:
        return

    prec = np.get_printoptions()['precision']

    print('Scale:\n\t', uPrint['scale'])
    b = tuple(uPrint['bias'])
    print('Bias:\n\t%.3f, %.3f, %.3f' % b)
    if 'biasLinear' in uPrint:
        # lb = tuple(uPrint['biasLinear'])
        np.set_printoptions(precision=3)
        print('Linear Bias:\n\t', uPrint['biasLinear'])
        np.set_printoptions(precision=prec)
    if 'grav' in uPrint:
        g = uPrint['grav']
        g_p = (np.linalg.norm(g),g[0],g[1],g[2])
        print('Gravity:\n\t(%.3f) %.3f, %.3f, %.3f' % g_p)

    if 'echoPlane' in uPrint:
        n = uPrint['echoPlane']
        n_p = (np.linalg.norm(n),n[0], n[1], n[2])
        print('Echo Plane(/s):\n\t(%.3g) %.3g, %.3g, %.3g' % n_p)
    if 'echoPlaneEst' in uPrint:
        print('\tFrom: %.3g, %.3g, %.3g' % tuple(uPrint['echoPlaneEst']))


def showResults(internals, params={}, ylimUseful=True, show=True):
    useful = internals['useful']
    A,u = internals['A'], internals['u']
    times = internals['times']

    Ax = np.reshape( np.dot(A,u), (3,-1) )
    b = np.reshape( internals['b'], (3,-1) )

    params['color'] = "blue"
    Plotting.plot3Axes(times, Ax, params=params, show=False)
    params['color'] = "red"
    axes = Plotting.plot3Axes(times, b, linspec='--r', params=params, show=False)

    if ylimUseful:
        for i, ax in enumerate(axes):
            A = Ax[i][useful[i]]
            ax.set_ylim([A.min()-2, A.max()+2])

    Plotting.highlight3Axes(times, useful, alpha=0.3, show=show)

    return axes


def lowPass(x, T, cutoff, axis=-1):
    '''
    :param x: The signal to be filtered (1D or 2D)
    :param T: Sampling rate of the signal, x
    :param cutoff: Cutoff frequency in Hertz
    '''
    # Construct the filter
    Fs = 1./T
    ntaps = np.ceil(Fs/cutoff)
    # Make sure its odd
    ntaps += 1 - (ntaps % 2)
    # window = ('kaiser', 1)
    # window = ('chebwin', 120)
    # window = ('gaussian', ntaps//3)
    # window = 'nuttall'
    # window = 'hamming'
    window = 'boxcar'
    b = signal.firwin(ntaps, cutoff, window=window, nyq=0.5*Fs)
    return ndimage.convolve1d(x, b, axis=axis)


def blur(x, t, axis=0, **kwargs):
    '''
    Remove high frequencies from a signal.
    :param x: The signal to be filters. ndarray
    :param t: The times of each sample, x
    :param axis: If x has more than 1 dimensions, select the axis along which to filter. Default = 0.
    :param cutoff: For LowPass method. Set the cutoff frequency of the low pass filter.
    '''
    T = np.mean(np.diff(t))
    cutoff = kwargs['cutoff']
    return lowPass(x, T, cutoff, axis=axis)


accFig = None
posFig = None
def showResultsNutmeg(internals, newChirp=None):
    try:
        import Nutmeg
    except ImportError:
        return

    if not Nutmeg.initialized():
        Nutmeg.init(timeout=10000)

    global accFig, posFig
    if accFig is None:
        accFig = Nutmeg.figure('cam vs imu', 'figs/tripleFig.qml')

    A,u = internals['A'], internals['u']
    times = internals['times']

    Ax = np.dot(A,u).reshape(3,-1)
    b = internals['b'].reshape(3,-1)

    accFig.set('ax[:].blue', x=times, y=Ax)
    accFig.set('ax[:].red', x=times, y=b)

    # Useful
    i = 0
    for (i, useful) in enumerate(internals['useful']):
        maxY = max(Ax[i].max(), b[i].max())
        minY = min(Ax[i].min(), b[i].min())
        accFig.set('ax[%d].green' % i, x=times, y=useful*(maxY - minY) + minY)
        i += 1


def _defaultConfig(userConfig={}):
    '''
    Get the default config settings used in parts of the scale estimation.
    Apply the customised user config on top of the defaults.
    '''
    # Throw in the user's settings over the top
    config = {}
    for key in defaultConfig:
        if key in userConfig:
            config[key] = userConfig[key]
        else:
            config[key] = defaultConfig[key]

    return config


def _quickTest(vid):
    Rci = np.array([[-1,0,0],[0,-1,0],[0,0,1]],float)  # iPhone Back Portrait

    imuData = IO.getImuDataFor(vid, Rci, forceReload=True)[1]
    camData = IO.getTrackingDataFor(vid, forceReload=True)[1]

    config = {
        'cutoff': 1.2,  # Hz
        'blurType': 'LowPass',
        'trimSides': 0.7,
        'minDelay': -0.1,
        'maxDelay': 0.1,

        'constrainGravity': 9.81,
        'constrainScale': [0, np.inf],

        'verbose': True
    }

    scale, internal = FindScale(camData, imuData, showPlot=True, config=config)


if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage: python ScaleFit <videoFile>")
    _quickTest(sys.argv[1])
