References
----------
 - *Hand Waving Away Scale*, C. Ham, S. Lucey, S. Singh, In European Conference on Computer Vision 2014
 - *Absolute Scale Estimation of 3D Monocular Vision on Smart Devices*, C. Ham, S. Lucey, S. Singh, In Mobile Cloud Visual Media Computing 2015

Requirements
------------
 - Python 3.5
 - Numpy

Optional
--------
 - Numba (Python JIT compiler) (numba.pydata.org/)
 - Nutmeg (Fast visualization) (github.com/kitizz/nutmeg/)

Notes
-----
This method works best with large, in and out motions. May need to find appropriate cutoff frequency based on the expected motions. The lower you can push it (given the motions) the better.

Numba will accelerate certain aspects of this code. If it's not wanted, simply remove import and the @jit decorators in Util.py

With the iPhone plugged in, open it in iTunes. Go to apps, 'Camera IMU', selected folders of desired sequences, and 'Save To' destination of choice.

The tracking method is expected to output a CSV file as described in the `read_external_poses` method in `Util.py`.

`ScaleSolve.py` has method `process_sequence(path)` that accepts a path to any of these data directories with an additional `tracking.csv` file.

License
-------
GNU GENERAL PUBLIC LICENSE Version 3 (GPLv3)
