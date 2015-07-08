import numpy as np
import os
import csv
import json
from subprocess import call

from AutoAssign import autoassign


class BadArgument(BaseException):
    pass


class BadDataException(BaseException):
    pass


class LogData:
    @autoassign
    def __init__(self, data, times):
        pass

    def copy(self):
        lg = LogData(self.data.copy(), self.times.copy())
        return lg


class IMUData:
    ''' Holds data from an IMU Log.

    Has the following LogData objects (each has a `data` and `times` property):
    gyro, grav, accel, linearAccel, rotation
    '''
    def __init__(self, measurements=None, times=None):
        # Keep a track of how much the times have been offset
        self.totalOffset = 0

        # Init with empty
        self.accel = LogData( np.array([]), np.array([]) )

        if measurements and times: # Load in the values from dictionaries
            if 'Accelerometer' in measurements:
                self.accel.data = measurements['Accelerometer']
                self.accel.times = times['Accelerometer']

    def reset(self):
        ''' Reset the time offsets.

        If the programmer knows what they're doing, this is a faster way of
        allowing reuse of the IMU data.'''
        self.offsetTimestamps(-self.totalOffset)
        self.totalOffset = 0.

    def copy(self):
        data = IMUData()
        data.accel = self.accel.copy()

        data.offsetTimestamps(-self.totalOffset)
        data.totalOffset = 0.

        return data

    def setOffset(self, offset):
        delta = offset - self.totalOffset
        self.offsetTimestamps(delta)

    def offsetTimestamps(self, delta):
        ''' Offset the timestamps of all the IMU data by `delta`.
        :param delta: Timestamp offset in the units of the timestamps.
        '''
        self.totalOffset += delta
        self.accel.times += delta


class CameraData:
    ''' Holds the extrinsics from tracked camera data.

    CameraData has 2 properties.
    transforms: A list of 4x4 numpy.ndarrays which represents the camera's 6DoF pose at each frame.
    times: 1D numpy.ndarray with a timestamp for each transform.object
    '''
    @autoassign
    def __init__(self, transforms, times, skipped=np.array([])):
        self.accel = None

    def copy(self):
        cd = CameraData(self.transforms.copy(),
                        self.times.copy(),
                        self.skipped.copy())
        # cd.accel = self.accel.copy()
        return cd


def getImuDataFor(videoFile, Rci=None, forceReload=False):
    ''' Collect the data from the IMU log corresponding with the video.

    Assumes that the log is a '.txt' file with the same name as the video file.
    After the first read, the log data is kept in a more efficient format for
    fast read-in speeds.

    :param forceReload: If true, the data will be reloaded from the original log
        file.
    '''
    success = False
    measurements = {}
    times = {}

    basePath, ext = os.path.splitext(videoFile)
    # videoDir = os.path.split(basePath)[0]

    # npSavePath = os.path.join(videoDir, 'imu.npz')
    npSavePath = basePath + '_imu.npz'
    logPath = basePath + '.txt'

    if os.path.exists(npSavePath) and not forceReload:
        # print "Loading IMU Log Data from", npSavePath
        loaded = np.load(npSavePath)
        measurements = loaded['measurements'].item()
        times = loaded['times'].item()
        success = True

    elif os.path.exists(logPath):
        # print "Loading IMU Log Data from", logPath
        # sources = {'Accelerometer':0, 'Gyroscope':1, 'LinearAcceleration':2,
        # 'Gravity':3, 'RotationVector':4}
        print("Attempting to read %s" % logPath)
        with open(logPath, 'rb') as logFile:
            success = True
            reader = csv.reader(logFile)
            firstRun = True

            for row in reader:
                source = row[0]
                meas = [float(v) for v in row[1:-1]]
                T = float(row[-1])
                # Offset the times so the first is zero
                if firstRun:
                    offsetT = T
                    firstRun = False
                # To append or to init
                if source in measurements:
                    measurements[source].append(meas)
                    times[source].append(T)
                else:
                    measurements[source] = [meas]
                    times[source] = [T]

            for source in measurements:
                measurements[source] = np.array(measurements[source])
                times[source] = (np.array(times[source]) - offsetT)/(10.**9)

            np.savez(npSavePath, measurements=measurements, times=times)

    if success:
        imuData = IMUData(measurements, times)
        if Rci is not None:
            imuData.accel.data = np.dot(Rci, imuData.accel.data.T).T
    else:
        raise BadDataException("Unable to read IMU measurements from %s" % logPath)

    return success, imuData


def getTrackingDataFor(vid, forceReload=False):
    success = False
    transforms = []

    # Locate the directory which should contain the tracked data.
    # Should be a subfolder of the same name as the video file.
    # videoDir, f = os.path.split( vid )
    videoDir = os.path.splitext(vid)[0] + '_output'
    # videoName, ext = os.path.splitext(f)
    # dataDir = getDataDir(videoFile)

    transformsFile = os.path.join(videoDir, 'cameraTransforms.npz')
    jsonFile = os.path.join(videoDir, 'cameraTransforms.json')

    if os.path.exists(transformsFile) and not forceReload:
        print("Loading Camera Transforms from %s" % transformsFile)
        compressed = np.load(transformsFile)
        transforms = compressed['transforms']
        times = compressed['times']
        if 'skipped' in compressed:
            skipped = compressed['skipped']
            skipped.sort()
        else:
            skipped = np.array([])
            transformsFile = os.path.join(videoDir, '/cameraTransforms.npz')

        success = True

    elif os.path.exists(jsonFile):
        with open(jsonFile) as file:
            J = json.load(file)
            transforms = np.array(J['transforms'])
            times = getFrameTimesFor(vid)
            times = times[-len(transforms):]
            times -= times[0]
            skipped = np.array(J['skipped'])
            np.savez(transformsFile, transforms=transforms, times=times, skipped=skipped)
            success = True

    else:
        raise(Exception("File, %s, does not exist." % transformsFile))

    camData = CameraData(transforms, times, skipped)

    return success, camData


def getFrameTimesFor(videoFile, probe_exe='avprobe'):
    ''' Get the video's timestamps for each frame.

    Uses ffprobe, ensure that ffmpeg is installed and in the environment path.

    :param probe_exe: Depending on what you have on your system, you should use 'ffprobe' or 'avprobe'. Default='avprobe'

    :return: 1D `numpy.ndarray` of timestamps.
    '''
    infoPath, ext = os.path.splitext(videoFile)
    infoPath += '_frames.txt'

    # Call ffprobe to extract the timeframe info
    # FNULL = open(os.devnull, 'w')
    with open(infoPath, 'w') as F:
        args = [probe_exe, '-i', videoFile, '-show_frames', '-print_format', 'json']
        call(args, stdout=F)

    timestamps = []
    with open(infoPath, 'r') as F:
        J = json.load(F)
        if 'frames' not in J:
            raise IOError('Unable to find video: %s' % videoFile)
        frames = J['frames']
        timestamps = [ float(frame['pkt_pts_time']) for frame in frames
                       if frame['media_type'] == 'video']

    return np.array(timestamps)
