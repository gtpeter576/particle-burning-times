import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def image_calibration(origin_folder_path, calibration_path, write_folder, ymin=0, ymax=799, xmin=0, xmax=1279):

    frame_paths = sorted(glob.glob(os.path.join(origin_folder_path, "*.tif")))
    calibration = cv.imread(calibration_path, cv.IMREAD_UNCHANGED).astype(np.float32)

    # Crop the calibration frame to the region of interest

    for i in range(len(frame_paths)):
        if frame_paths[i] is None:
            print(f"image {i} not found")

    cal_frame = calibration[ymin:ymax+1, xmin:xmax+1]

    for i, path in enumerate(frame_paths):
        # if i>50:
        #     break
        frame = cv.imread(path, cv.IMREAD_UNCHANGED).astype(np.float32)
        #cv sees a 12-bit image and scales it to 16-bit when we read it, giving us values > 4096
        #the calibration frame gets saved as a 16-bit image, so when cv opens it it does not scale it. Therefore we need
            #to scale it by 16
        calibrated_frame = np.clip(frame[ymin:ymax+1, xmin:xmax+1] - 16*cal_frame, 0, 65535).astype(np.uint16)
        base_name = path.split("\\")[-1].split("_")[0]
        if len(str(i)) == 1:
            write_path = write_folder + "\\" + base_name + "_calibrated_0000" + str(i) + ".tif"
        elif len(str(i)) == 2:
            write_path = write_folder + "\\" + base_name + "_calibrated_000" + str(i) + ".tif"
        elif len(str(i)) == 3:
            write_path = write_folder + "\\" + base_name + "_calibrated_00" + str(i) + ".tif"
        elif len(str(i)) == 4:
            write_path = write_folder + "\\" + base_name + "_calibrated_0" + str(i) + ".tif"
        else:
            write_path = write_folder + "\\" + base_name + "_calibrated_" + str(i) + ".tif"
        cv.imwrite(write_path, calibrated_frame)
        
        if i%20 == 0:
            print(f"Processed frame {i+1}/{len(frame_paths)}: {write_path}")