# Monocular Scale
Source code for estimating scale of tracking data collected on smart phones and other smart devices.


## Usage
`src/ScaleFit.py` is where it's all at.

The data for a sequence should all be in one directory. It should contain
 - The video file: `videoName.[mp4,mov,etc]`
 - Camera tracking data: `cameraTransforms.json`
 - IMU log file: `videoName.txt`

Then run `python src/ScaleFit.py path/to/videoName.ext`


### Tracking data format
Tracking data is expected to be in JSON format with 2 values
```json
{
    "transforms": [ [[...],[...],[...]], ...],
    "skipped": [...]
}
```
Where
 - transforms: An Fx3x4 array. F is the number of frames. Each frame describes the rotation and translation of the camera in world coordindates [R | t].
 - skipped: Indices for any frames whose transforms should be disregarded in the scale estimation.


### IMU log format
A very simple format with each line being:
`<Sensor Type> X Y Z timestamp`

The only `<Sensor Type>` this project pays attention to is `Accelerometer`. `timestamp` is in nanoseconds.

### Fiddle Factor
You may need to fiddle with the `Rci` parameter in ScaleFit for your particular setup. `Rci` is used to align the local IMU measurements with the camera's frame of reference. By default it is set for IMU data collected on iPhone 6 and video recorded in portrait mode on the rear-facing camera.
