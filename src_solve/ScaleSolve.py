import numpy as np
from numpy import dot, empty, empty_like, zeros, eye, r_
from numpy.linalg import norm
from scipy.ndimage import filters

import os

import Util
import Quat

# import pynutmeg


def init_jacobian(
        grav, bias, scale, Rs, aw_cams, a_imus,
        solve_bias=False,
        w_grav=1e6, w_scale=1e6, w_bias=1,
        edge=10):
    '''
    Initialize the Jacobian, residual and weight arrays.
    '''
    F = len(Rs)
    E = 3*F + 2 + 3*solve_bias
    U = 4 + 3*solve_bias

    J = empty((E, U))
    r = empty(E)
    w = empty(E)
    J[:] = np.nan
    r[:] = np.nan
    w[:] = np.nan

    for f in range(F):
        R = Rs[f]
        a_cam = dot(R, aw_cams[f])

        s = 3*f
        e = s + 3

        # Jacobian derivatives
        J[s:e, 0:3] = R  # wrt g
        if solve_bias:
            J[s:e, 3:6] = eye(3)
        J[s:e, -1] = a_cam
        w[s] = 1
        w[s + 1] = 1
        w[s + 2] = 0

    rowG = 3*F
    rowS = 3*F + 1

    edge_sz = 3*edge
    w[:edge_sz] = 0
    w[3*F - edge_sz:3*F] = 0
    w[rowG] = w_grav
    w[rowS] = 0

    J[rowG] = 0
    J[rowS] = 0
    J[rowS, -1] = 1

    if solve_bias:
        rowB = 3*F + 2
        w[rowB: rowB + 3] = w_bias
        J[rowB: rowB + 3] = 0
        J[rowB: rowB + 3, 3:6] = eye(3)

    return J, r, w


def jacobian(
        J, r, w,
        grav, bias, scale,
        Rs, aw_cams, a_imus,
        solve_bias=False,
        w_grav=1e6, w_scale=1e6,
        mag_grav=9.81):
    '''
    Load in updated residuals and derivatives of the function given the current estimate.
    '''
    F = len(Rs)

    for f in range(F):
        s = 3*f
        e = s + 3

        R = Rs[f]
        a_cam = dot(R, aw_cams[f])
        a_imu = a_imus[f]

        # Acceleration residual
        r[s:e] = dot(R, grav) + bias + scale*a_cam - a_imu

    # Keep |grav| == mag_grav
    rowG = 3*F
    gx, gy, gz = grav
    r[rowG] = gx*gx + gy*gy + gz*gz - mag_grav*mag_grav
    # Derive
    J[rowG, 0] = 2*gx
    J[rowG, 1] = 2*gy
    J[rowG, 2] = 2*gz

    # Keep scale positive
    rowS = 3*F + 1
    r[rowS] = scale
    w[rowS] = 0  #w_scale * (scale < 0)

    if solve_bias:
        # Regularise bias slightly
        rowB = 3*F + 2
        r[rowB: rowB + 3] = bias

    Jw = J.T * w
    H = dot(Jw, J)
    x = dot(Jw, r)

    return H, x


def solve_at_delay(
        delay, Rws, tws, times_cam, a_imu, times_imu,
        state,
        cutoff=50, solve_bias=False,
        its=50, eps=1e-6, plot=False):
    '''
    Delay camera times by delay amount and solve.
    Modifies the entries of the state argument.

    state: [g, b, s]
    cutoff: Cuttoff frequency in Hz
    its: Maximum iterations
    eps: Minimum delta for convergence
    '''
    grav, bias, scale = state
    # print("Delay:", delay)
    # print("g, b, s:", grav, bias, scale)
    # print("|g| =", norm(grav))

    times_delay = times_cam - delay
    s, e = 0, len(times_imu) - 1
    while times_imu[s] < times_delay[0]:
        s += 1
    while times_imu[e] > times_delay[-1]:
        e -= 1

    times = times_imu[s:e]
    a_imu = a_imu[s:e]

    Rws_int = Quat.slerp1d(times, times_delay, Rws)
    Rs_int = Rws_int.transpose(0,2,1)

    N = len(times_delay)
    P_cam = (times_delay[-1] - times_delay[0]) / (N - 1)
    P_imu = (times[-1] - times[0]) / (e - s - 1)

    # tws = Util.low_pass(tws, P_cam, cutoff, axis=0)
    a_imu = Util.low_pass(a_imu, P_imu, cutoff, axis=0)

    diff2 = 1/P_cam**2 * r_[1., -2., 1.]

    aws = filters.convolve1d(tws, diff2, axis=0)
    aws_int = Util.interp(times, times_delay, aws)
    aw_cams = Util.low_pass(aws_int, P_imu, cutoff, axis=0)

    J, r, w = init_jacobian(grav, bias, scale, Rs_int, aw_cams, a_imu, solve_bias=solve_bias, w_grav=1)

    for it in range(its):
        H, x = jacobian(J, r, w, grav, bias, scale, Rs_int, aw_cams, a_imu, solve_bias=solve_bias, w_grav=1)

        u = -np.linalg.solve(H, x)
        if np.isnan(u).any():
            raise ValueError("NaNs in solution!")

        grav += u[0:3]
        scale += u[-1]
        if solve_bias:
            bias += u[3:6]

        normu = norm(u)
        if normu < eps:
            break

    # Visualize in nutmeg
    F = len(aw_cams)
    if plot:
        fig = pynutmeg.figure('cam vs imu', 'figs/fig_triple.qml')
        a_cams = empty_like(aw_cams)
        for f in range(F):
            R = Rs_int[f]
            a_cams[f] = dot(R, grav) + bias + scale*dot(R, aw_cams[f])

        titles = ['X', 'Y', 'Z']
        for i in range(3):
            ax = 'ax{}'.format(i)
            camhandle = ax + '.P1'
            imuhandle = ax + '.P2'
            fig.set(ax + '.yAxis', label=titles[i] + ' Accel (m/s^2)')
            fig.set(camhandle, x=times_cam, y=a_cams[:,i])
            fig.set(imuhandle, x=times_cam, y=a_imu[:,i])
        fig.set('ax0', minX=0, maxX=times_cam[-1])
        fig.set('ax2.xAxis', label='Time (s)')

    state[0] = grav
    state[1] = bias
    state[2] = scale

    res = norm((r*w)[:3*F])
    # print("Res:", res)

    return res, state


def solve(Rs, ts, times_cam, a_imu, times_imu, cutoff=50, eps_time=1e-4):
    '''
    cutoff: Smoothing applied to visual accelerations and IMU data before optimization
    '''

    grav = r_[0, 9.81, 0]
    bias = zeros(3)
    scale = 1
    state = [grav, bias, scale]

    Rws, tws = Util.global_to_local(Rs, ts)

    def assess_at_delay(delay):
        return solve_at_delay(delay, Rws, tws, times_cam, a_imu, times_imu, state, plot=False)

    delay = Util.golden_section_search(assess_at_delay, -0.1, 0.1)[0]

    print("Solved")
    print("Gravity: {}, Bias: {}, Scale: {}".format(*state))
    print("Delay: {}".format(delay))
    # state = solve_at_delay(delay, Rs, ts, times_cam, a_imu, state)[0]

    return state


def process_sequence(path):
    vidpath = os.path.join(path, 'imu.mp4')
    trackpath = os.path.join(path, 'tracking.csv')
    imupath = os.path.join(path, 'imu.txt')

    times_cam = Util.get_video_timestamps(vidpath)
    Rs, ws, ts = Util.read_external_poses(trackpath)
    times_imu, a_imu, w_imu = Util.read_imu(imupath)
    F = len(Rs)

    dR = np.array([
        [0, -1, 0],
        [-1, 0, 0],
        [0, 0, 1]
    ])
    a_imu = dot(a_imu, dR.T)

    return solve(Rs, ts, times_cam[:F], a_imu, times_imu)


if __name__ == '__main__':
    process_sequence('../data/robot_arm')
