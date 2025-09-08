'''import os
import glob
import cv2 as cv
import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory

# Global reference to calibration data in shared memory
_shared_calibration = None
cal_shape = None


def init_worker(shared_name, shape, dtype):
    """
    Each worker process attaches to the shared calibration frame.
    """
    global _shared_calibration, cal_shape
    shm = shared_memory.SharedMemory(name=shared_name)
    _shared_calibration = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    cal_shape = shape


def process_frame(args):
    """
    Processes a single image frame by calibrating and saving it.
    """
    path, write_folder, ymin, xmin = args
    try:
        frame = cv.imread(path, cv.IMREAD_UNCHANGED).astype(np.float32)

        # Crop frame and calibrate using shared calibration
        calibrated_frame = np.clip(
            frame[ymin:ymin + cal_shape[0], xmin:xmin + cal_shape[1]] - 16 * _shared_calibration,
            0, 65535
        ).astype(np.uint16)

        # Construct output filename
        base_name = os.path.basename(path).split("_")[0]
        i = int(os.path.basename(path).split("_")[-1].split(".")[0])
        write_path = os.path.join(write_folder, f"{base_name}_calibrated_{i:05d}.tif")

        cv.imwrite(write_path, calibrated_frame)
        return f"Processed: {write_path}"

    except Exception as e:
        return f"Error processing {path}: {e}"


def image_calibration_multiprocessed(origin_folder_path, calibration_path, write_folder,
                                     ymin=0, ymax=799, xmin=0, xmax=1279):
    """
    Calibrates a series of images in parallel using multiprocessing.
    """
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    frame_paths = sorted(glob.glob(os.path.join(origin_folder_path, "*.tif")))

    # Load calibration once in main process
    calibration = cv.imread(calibration_path, cv.IMREAD_UNCHANGED).astype(np.float32)
    cal_frame = calibration[ymin:ymax + 1, xmin:xmax + 1]

    # Put calibration frame into shared memory
    shm = shared_memory.SharedMemory(create=True, size=cal_frame.nbytes)
    shm_arr = np.ndarray(cal_frame.shape, dtype=cal_frame.dtype, buffer=shm.buf)
    np.copyto(shm_arr, cal_frame)

    try:
        # Prepare args (without calibration frame)
        args = [(path, write_folder, ymin, xmin) for path in frame_paths]

        # Use pool with limited workers
        with Pool(processes=cpu_count(),
                  initializer=init_worker,
                  initargs=(shm.name, cal_frame.shape, cal_frame.dtype)) as pool:
            
            for result in pool.imap(process_frame, args, chunksize=10):
                print(result)

    finally:
        # Cleanup shared memory
        shm.close()
        shm.unlink()'''
        
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from multiprocessing import Pool, cpu_count
            
# A helper function to process a single image frame
def process_frame(args):
    """
    Processes a single image frame by calibrating and saving it.

    Args:
        args (tuple): A tuple containing the path to the frame, the cropped
                      calibration frame, the write folder, ymin, and xmin.
    """
    path, cal_frame, write_folder, ymin, xmin = args
    
    # Read the image frame
    frame = cv.imread(path, cv.IMREAD_UNCHANGED).astype(np.float32)
    
    # Crop the frame and perform calibration
    calibrated_frame = np.clip(frame[ymin:ymin+cal_frame.shape[0], xmin:xmin+cal_frame.shape[1]] - 16 * cal_frame, 0, 65535).astype(np.uint16)
    
    # Construct the output filename
    base_name = os.path.basename(path).split("_")[0]
    i = int(os.path.basename(path).split("_")[-1].split(".")[0])
    write_path = os.path.join(write_folder, f"{base_name}_calibrated_{i:05d}.tif")
    
    # Save the calibrated image
    cv.imwrite(write_path, calibrated_frame)

    return(f"Processed frame: {write_path}")

def image_calibration_multiprocessed(origin_folder_path, calibration_path, write_folder, ymin=0, ymax=799, xmin=0, xmax=1279):
    """
    Calibrates a series of images in parallel using multiprocessing.

    Args:
        origin_folder_path (str): Path to the folder containing the raw TIFF images.
        calibration_path (str): Path to the calibration (dark) frame TIFF file.
        write_folder (str): Path to the folder where the calibrated images will be saved.
        ymin, ymax, xmin, xmax (int): Coordinates defining the region of interest (ROI)
                                      for cropping the images.
    """
    # Create the write folder if it doesn't exist
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # Get a sorted list of all TIFF files
    frame_paths = sorted(glob.glob(os.path.join(origin_folder_path, "*.tif")))

    # Read and crop the calibration frame once
    calibration = cv.imread(calibration_path, cv.IMREAD_UNCHANGED).astype(np.float32)
    cal_frame = calibration[ymin:ymax+1, xmin:xmax+1]

    # Prepare the arguments for each process
    args = [(path, cal_frame, write_folder, ymin, xmin) for path in frame_paths]
    print("ATG")

    # Determine number of processes to use (leave one CPU free)
    num_processes = max(1, cpu_count() - 1)
    # Use a multiprocessing Pool to distribute the workload
    with Pool(processes=num_processes) as pool:
        i = 0
        for result in pool.imap_unordered(process_frame, args):
            # Print a progress update as each process completes
            if i % 20 == 0:
                print(f"Processed frame {i+1}/{len(args)}: {result}")
            i += 1



if __name__ == '__main__':
    origin_folder_paths = [r"D:\02092025\Test5"]
    calibration_paths = [r"D:\02092025\Test5_Calibrated\AVG_Test5_Cal_Samples.tif"]
    write_folders = [r"D:\02092025\Test5_Calibrated"]

    for i in range(1):
        image_calibration_multiprocessed(origin_folder_paths[i], calibration_paths[i], write_folders[i])
