# particle-burning-times

repository for caluclating particle burn times from stack of tiffs from a video file.

image_calibration.py subtracts the background given an origin folder, calibration frame and destination folder

particle_tracking.py tracks the particles in the calibrated frames, outputting a nested dictionary containing particle location and brightness for the frames that it exists

these initial burn times are skewed due to particle_tracking detecting erroneous particles. Further processing can be done with particle_driver.ipynb to find accurate burn times.
