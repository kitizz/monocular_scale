from numpy import array, empty, dot, eye, sin, cos, arccos, sqrt
import numpy as np

from scipy import ndimage, signal

from numba import jit

import os
import json
from subprocess import call


# -------------------------
#       IO Methods
# -------------------------
def read_external_poses(path):
    '''
    Read 6DoF poses created by external code in ASCII, CSV format.
    Poses should be packed in as (rx,ry,rz,tx,ty,tz)

    First line is expected to be the header. (rx, ry, rz) is the axis-angle
    representation of the orientation. The poses are in the local reference
    frame such that a global point is projected into each camera's reference
    frame by:

    p_local = R . p_global + t

    Where R is created from (rx,ry,rz) by the method `w_to_R`

    return: Rs, ts
    '''
    R_list = []
    w_list = []
    t_list = []
    with open(path) as file:
        # Read out header
        file.readline()
        while True:
            text_values = file.readline().strip().split(',')
            if len(text_values) != 6:
                break
            values = [ float(v) for v in text_values ]
            w = array(values[:3])
            R_list.append(w_to_R(w))
            t_list.append(array(values[3:]))
            w_list.append(w)

    Rs = array(R_list)
    ws = array(w_list)
    ts = array(t_list)

    return Rs, ws, ts


def read_imu(path):
    '''
    Extract IMU data from file recorded with data collection app.
    Assumes common timestamps for different readings...
    True for iPhone, but may differ on other platforms.

    Example of expected format:
        Accelerometer,x,y,z,timestamp
        Gyroscope,x,y,z,timestamp

    Where timestamp is a long int in nanoseconds.

    return: times, acceleration, angular velocity
    '''
    labels = ['Accelerometer', 'Gyroscope']
    outlists = [ [] for name in labels ]
    timelists = [ [] for name in labels ]
    with open(path) as file:
        while True:
            text_values = file.readline().strip().split(',')
            if len(text_values) < 4:
                break

            label = text_values[0]
            try:
                index = labels.index(label)
            except ValueError:
                continue

            arr = array([ float(v) for v in text_values[1:-1] ])
            timelists[index].append(int(text_values[-1]))
            outlists[index].append(arr)

    # Compare timestamps across readings, make sure it's all dandy
    timestamps = array(timelists[0])
    for i in range(1, len(timelists)):
        if (timestamps != array(timelists[i])).any():
            raise ValueError("Timestamps not aligned. Need to rewrite code.")

    output = [(timestamps - timestamps[0])*1e-9]  # Nanoseconds -> seconds
    measurements = [ array(lst) for lst in outlists ]
    output.extend(measurements)

    return output


def get_video_timestamps(vidpath):
    ''' Get the video's timestamps for each frame.
    Uses ffprobe, ensure that ffmpeg is installed and in the environment path.

    return: 1D `numpy.ndarray` of timestamps.
    '''
    print("Getting frame times for:", vidpath)
    vidpath = os.path.abspath(vidpath)
    info_path, ext = os.path.splitext(vidpath)
    info_path += '_frames.txt'

    def get_info():
        try:
            with open(info_path, 'w') as F:
                args = ['ffprobe', '-i', vidpath, '-show_frames', '-print_format', 'json']
                call(args, stdout=F, stderr=FNULL)
        except:
            os.remove(info_path)
            raise

    # Call ffprobe to extract the timeframe info
    FNULL = open(os.devnull, 'w')
    if not os.path.exists(info_path):
        get_info()

    timestamps = []
    with open(info_path, 'r') as F:
        J = json.load(F)
        if 'frames' not in J:
            raise IOError('Unable to find video: %s' % vidpath)
        frames = J['frames']
        timestamps = [ float(frame['pkt_pts_time']) for frame in frames
                       if frame['media_type'] == 'video']

    return array(timestamps)


# -------------------------
#       Algorithms
# -------------------------
def golden_section_search(f, a, b, tol=1e-3):
    '''
    Do a golden section search over function, f(x). The function should return
    2 values: a residual and a result.
    residual, result = f(x)

    Credit: From Wikipedia, http://en.wikipedia.org/wiki/Golden_section_search
    '''
    gr = 0.618  # Golden ratio
    # x = a
    # y = b
    x = b - gr*(b - a)
    y = a + gr*(b - a)
    fx, rx = f(x)
    fy, ry = f(y)
    vs = [x, y]
    fs = [fx, fy]

    while abs(x - y) > tol:
        fx, rx = f(x)
        fy, ry = f(y)
        vs.append(x)
        fs.append(fx)
        vs.append(y)
        fs.append(fy)

        if fx < fy:
            b = y
            y = x  # fy = fx; fx = f(x)
            x = b - gr*(b - a)

        else:
            a = x
            x = y  # fx = fy; fy = f(y)
            y = a + gr*(b - a)

    return (x + y)/2.0, array(vs), array(fs)


@jit(nopython=True, nogil=True)
def interp(t, t0, P0):
    '''
    Multichannel interpolation of array along axis 0
    '''
    N = len(t)
    M = len(t0)

    cols = P0.shape[1]

    P = empty((N, cols))
    # val = empty(N, nb.bool_)

    ind = 0
    max_ind = M - 2
    for i in range(N):
        time = t[i]
        while ind < max_ind and t0[ind + 1] < time:
            ind += 1

        # if not valid[ind] or not valid[ind + 1]:
        # val[i] = valid[ind] and valid[ind + 1]

        a, b = t0[ind], t0[ind + 1]
        frac = (time - a) / (b - a)

        for c in range(cols):
            P[i, c] = (1 - frac)*P0[ind, c] + (frac)*P0[ind + 1, c]

    return P


def low_pass(x, T, cutoff, axis=-1):
    '''
    :param x: The signal to be filtered (1D or 2D)
    :param T: Sampling rate of the signal, x
    :param cutoff: Cutoff frequency in Hertz
    '''
    # Construct the filter
    Fs = 1./T
    ntaps = np.ceil(Fs/cutoff) + 10
    # Make sure its odd
    ntaps += 1 - (ntaps % 2)
    # window = ('kaiser', 1)
    # window = ('chebwin', 120)
    # window = ('gaussian', ntaps//3)
    # window = 'nuttall'
    window = 'hamming'
    # window = 'boxcar'
    b = signal.firwin(ntaps, cutoff, window=window, nyq=0.5*Fs)
    return ndimage.convolve1d(x, b, axis=axis)


# -------------------------
#       Geometry
# -------------------------
@jit(nopython=True, nogil=True)
def global_to_local(Rs, ts):
    '''
    Note: can also be used as local_to_global
    '''
    lRs = np.empty_like(Rs)
    lts = np.empty_like(ts)

    for i in range(ts.shape[0]):
        Rw = Rs[i]
        lRs[i] = Rw.T
        lts[i] = -dot(Rw.T, ts[i])

    return lRs, lts


@jit(nopython=True, nogil=True)
def norm1d(a):
    N = a.shape[0]
    mag = 0
    for i in range(N):
        mag += a[i]**2

    return sqrt(mag)


@jit(nopython=True, nogil=True)
def norm3(a):
    x, y, z = a
    return sqrt(x*x + y*y + z*z)


@jit(nopython=True, nogil=True)
def make_skew(w):
    K = empty((3,3), w.dtype)
    K[0,0], K[0,1], K[0,2] = 0, -w[2], w[1]
    K[1,0], K[1,1], K[1,2] = w[2], 0, -w[0]
    K[2,0], K[2,1], K[2,2] = -w[1], w[0], 0
    return K


@jit(nopython=True, nogil=True)
def w_to_R(w):
    # AKA: Rodrigues
    # Ref: https://en.wikipedia.org/wiki/Axis-angle_representation
    theta = norm3(w)
    if theta == 0:
        return eye(3)

    ax = w/theta
    K = make_skew(ax)  # Cross product matrix
    R = eye(3) + sin(theta)*K + (1 - cos(theta))*dot(K,K)
    return R


@jit(nopython=True, nogil=True)
def R_to_w(R):
    trace = 0.0
    for i in range(3):
        trace += R[i,i]

    theta = arccos( (trace - 1)*0.5 )
    if theta == 0:
        return np.zeros(3)

    w = empty(3)
    w[0] = R[2,1] - R[1,2]
    w[1] = R[0,2] - R[2,0]
    w[2] = R[1,0] - R[0,1]
    w *= theta/(2*sin(theta))

    return w
