from numpy import empty, zeros, full, dot, sqrt, sin, arccos
import numpy as np

from numba import jit

from Util import norm1d

_sgns = full((4,3), -1.)
_sgns[0] = 1
for i in range(3):
    _sgns[i+1,i] = 1


@jit(nopython=True, nogil=True)
def R2quats(Rs, eps=1e-14):
    N = Rs.shape[0]
    Q = empty((N,4))
    tr = empty(3)

    for i in range(N):
        q = Q[i]
        R = Rs[i]

        for j in range(3):
            tr[j] = R[j,j]

        den = 1 + dot(_sgns, tr)
        u = den.argmax()
        q[u] = 0.5 * sqrt(den[u])
        denom = 1 / (4.0 * q[u])

        if u == 0:
            q[1] = (R[2,1] - R[1,2]) * denom
            q[2] = (R[0,2] - R[2,0]) * denom
            q[3] = (R[1,0] - R[0,1]) * denom

        elif u == 1:
            q[0] = (R[2,1] - R[1,2]) * denom
            q[2] = (R[1,0] + R[0,1]) * denom
            q[3] = (R[2,0] + R[0,2]) * denom

        elif u == 2:
            q[0] = (R[0,2] - R[2,0]) * denom
            q[1] = (R[1,0] + R[0,1]) * denom
            q[3] = (R[2,1] + R[1,2]) * denom

        elif u == 3:
            q[0] = (R[1,0] - R[0,1]) * denom
            q[1] = (R[2,0] + R[0,2]) * denom
            q[2] = (R[2,1] + R[1,2]) * denom

        q /= norm1d(q)

    return Q


@jit(nopython=True, nogil=True)
def quat2Rs(Q):
    N = Q.shape[0]
    Rs = empty((N,3,3))
    for i in range(N):
        q = Q[i]
        R = Rs[i]

        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        xx2 = 2 * x * x
        yy2 = 2 * y * y
        zz2 = 2 * z * z
        xy2 = 2 * x * y
        wz2 = 2 * w * z
        zx2 = 2 * z * x
        wy2 = 2 * w * y
        yz2 = 2 * y * z
        wx2 = 2 * w * x

        R[0,0] = 1. - yy2 - zz2
        R[0,1] = xy2 - wz2
        R[0,2] = zx2 + wy2
        R[1,0] = xy2 + wz2
        R[1,1] = 1. - xx2 - zz2
        R[1,2] = yz2 - wx2
        R[2,0] = zx2 - wy2
        R[2,1] = yz2 + wx2
        R[2,2] = 1. - xx2 - yy2

    return Rs


@jit(nopython=True, nogil=True)
def quatinv(q):
    p = empty(4)
    p[0] = q[0]
    p[1:] = -q[1:]
    return p


@jit(nopython=True, nogil=True)
def slerp(p0, p1, t):
    ''' Spherical linear interpolation for quaternions.
    :param p0: The first quaternion rotation.
    :param p1: The second quaternion rotation.
    :param t: The interpolation factor (0.0 <= t <= 1.0).
    '''
    # print p0, p1
    dp = dot(p0/norm1d(p0), p1/norm1d(p1))
    omega = arccos(min(dp,1.0))
    if omega == 0:
        return p0
    so = sin(omega)
    return sin((1.0-t)*omega) / so * p0 + sin(t*omega)/so * p1


@jit(nopython=True, nogil=True)
def slerp1d(t, tp, R):
    ''' Spherically interpolate the rotation matrices in R from tp -> t.
    :param t: 1d array desired sampled times
    :param tp: 1d array, size N, of sampled times of rotation measurements
    :param R: Nx3x3 array of rotation matrices.
    '''
    # if t[0] < tp[0] or t[-1] > tp[-1]:
        # raise ValueError("'t' must be bound by 'tp'. Cannot perform extrapolation.")

    Q = R2quats(R)

    newQ = zeros((t.size, 4))
    # Slow for the moment
    i = 1
    for j, T in enumerate(t):
        while tp[i] < T:
            i += 1
        frac = (T - tp[i-1]) / (tp[i] - tp[i-1])

        newQ[j] = slerp(Q[i-1], Q[i], frac)

    return quat2Rs(newQ)


def fill_gaps_Rt(R, t, valid):
    # QUickly look for blips in the rotations
    Rdiff = np.diff(R[:,0], axis=0)
    valid = valid[:-1] & (np.abs(Rdiff) < 0.2).all(axis=1)

    diff = np.diff(valid.astype(np.int8))
    starts = np.where(diff < 0)[0]
    ends = np.where(diff > 0)[0]

    N = min(len(starts), len(ends))

    for i in range(N):
        s, e = starts[i], ends[i]
        ta, tb = t[s], t[e+1]
        qa = R2quats(R[s:s+1])[0]
        qb = R2quats(R[e+1:e+2])[0]
        sz = e + 1 - s
        for j in range(1, sz):
            w = j/sz
            t[s+j] = (1-w)*ta + w*tb
            q = slerp(qa, qb, w)
            R[s+j] = quat2Rs(q.reshape(1,4))
