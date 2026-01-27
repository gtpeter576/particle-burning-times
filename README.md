# particle-burning-times

repository for caluclating particle burn times from stack of tiffs from a video file.

image_calibration.py subtracts the background given an origin folder, calibration frame and destination folder

particle_tracking.py tracks the particles in the calibrated frames, outputting a nested dictionary containing particle location and brightness for the frames that it exists

these initial particle tracks include erroneous tracks. Further processing is done with tracked_particle_analyzer.py to remove erroneous tracks, and to classify particles into ignition modes to suit the experimental setup.
