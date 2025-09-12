import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import csv

def track_particle_burns(tiff_folder_path, threshold, frame_rate, max_distance=35, search_previous_frames=1, stop=False, stop_frame=200):
    # Get a sorted list of all TIFF files in the folder
    tiff_files = sorted(glob.glob(os.path.join(tiff_folder_path, "*.tif")))
    if not tiff_files:
        print("Error: No TIFF files found in the specified folder.")
        return

    # Dictionary to store particle burn durations, start/end coordinates, start/end frames, brightnesses and sizes
    particle_lifetimes = {}
    particle_start_frames = {}
    particle_end_frames = {}
    particle_coordinates = {}
    particle_brightness = {}
    next_particle_id = 0
    frame_counter = 0


    for tiff_file in tiff_files:

        if stop and frame_counter > stop_frame:
            break
        print(f"Processing frame {frame_counter}/{len(tiff_files)}: {tiff_file}")

        # Read the TIFF image as grayscale
        frame = cv2.imread(tiff_file, cv2.IMREAD_ANYDEPTH)
        

        # # Calibration, if done in this code
        # new_frame = np.zeros((calibration.shape[0], calibration.shape[1]), dtype=np.uint8)
        # for i in range(frame.shape[0]):
        #     for j in range(frame.shape[1]):
        #     # Only take blue component when subtracting calibration image
        #         if frame[i, j, 0] < calibration[i, j, 0]:
        #             new_frame[i, j] = 0
        #         else:
        #             new_frame[i, j] = frame[i, j, 0] - calibration[i, j, 0]
        # frame = new_frame

        if frame is None:
            print(f"Error: Could not read file {tiff_file}")
            continue
        
        # Threshold to isolate white dots
        _, thresh = cv2.threshold(frame, threshold, 65535, cv2.THRESH_BINARY)
        thresh8 = thresh.astype(np.uint8)

        # Find contours of the white dots
        contours, _ = cv2.findContours(thresh8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get centroids of detected particles
        current_centroids = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # if w > 1 or h > 1:  # Filter out small contours
            cx, cy = x + w // 2, y + h // 2
            current_centroids.append((cx, cy, w, h))

        # Match current centroids to previous particles
        unmatched_centroids = set(range(len(current_centroids))) #make list of unmatched centroids

        #loop through previous particles to find matches with current centroids
        for particle_id in range(next_particle_id):
            (px, py) = particle_coordinates[particle_id][particle_end_frames[particle_id]]
            #exit this loop iteration if the particle is too old
            if frame_counter - particle_end_frames[particle_id] > search_previous_frames:
                continue
            
            best_match = None
            best_distance = max_distance*(frame_counter - particle_end_frames[particle_id])  # Initialize with a large distance based on the frame difference

            for i, (cx, cy, w, h) in enumerate(current_centroids):
                distance = np.sqrt((cx - px) ** 2 + (cy - py) ** 2) #calculate distance
                #Find closest particle to previous particle
                if distance < best_distance and i in unmatched_centroids:
                    best_match = i
                    best_distance = distance

            # If a match is found, update the particle position, end coordinates, end frame and lifetime
            if best_match is not None:
                print("new centroid matched to a particle", frame_counter - particle_end_frames[particle_id], "frames prior to current frame")
                (cx, cy, w, h) = current_centroids[best_match]
                coords = (cx, cy)
                particle_coordinates[particle_id][frame_counter] = coords
                particle_brightness[particle_id][frame_counter] = np.array(frame[coords[1]-2:coords[1]+2, coords[0]-2:coords[0]+2]).mean()

                particle_lifetimes[particle_id] += (frame_counter - particle_end_frames[particle_id])/frame_rate*1000  # Update lifetime by appropriate frame difference, convert to ms
                particle_end_frames[particle_id] = frame_counter
                unmatched_centroids.discard(best_match)

        # Assign new IDs to unmatched centroids
        # Must have at least one pixel within 3 pixels of the center greater than the threshold for the new centroid to count as a new particle
        for i in unmatched_centroids:
            (px, py, w, h) = current_centroids[i]
            coords = (px, py)
            print(f"New particle detected at frame {frame_counter}: {coords}; id: {next_particle_id}")
            particle_lifetimes[next_particle_id] = 0

            particle_start_frames[next_particle_id] = frame_counter

            particle_end_frames[next_particle_id] = frame_counter
            particle_coordinates[next_particle_id] = {}
            particle_coordinates[next_particle_id][frame_counter] = coords

            particle_brightness[next_particle_id] = {}
            particle_brightness[next_particle_id][frame_counter] = np.array(frame[py-2:py+2, px-2:px+2]).mean()

            next_particle_id += 1
        frame_counter += 1

    cv2.destroyAllWindows()

    # Print particle lifetimes
    # for particle_id, lifetime in particle_lifetimes.items():
    #     try:
    #         print(f"Particle {particle_id}: start frame: {particle_start_frames[particle_id]}; start coordinate: {particle_start_coordinates[particle_id]}; burn time: {round(lifetime, 1)}; end frame: {particle_end_frames[particle_id]}; end coordinate: {particle_end_coordinates[particle_id]}")
    #     except Exception as e:
    #         print(f"Error processing particle {particle_id}: {e}")
    
    # Check if output file exists, and add a number in parentheses if it does
    test_num = tiff_folder_path.split(os.sep)[-1]
    base_csv = os.path.join(tiff_folder_path, "..", test_num + "_particle_tracking_results.csv")
    output_csv = base_csv
    count = 1
    while os.path.exists(output_csv):
        output_csv = os.path.splitext(base_csv)[0] + f"({count})" + ".csv"
        count += 1

    # Write results to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'threshold={threshold}, frame_rate={frame_rate}, max_distance={max_distance}, search_previous_frames={search_previous_frames}'])
        for particle_id in particle_lifetimes.keys():

            frame_range = np.arange(particle_start_frames[particle_id], particle_end_frames[particle_id]+1)
            writer.writerow([f'Particle {particle_id}'])
            frame_write = list(['frame'])
            for i in range(len(frame_range)):
                frame_write.append(frame_range[i])
            writer.writerow(frame_write)

            xcoord_write = list(['particle x coordinate'])
            ycoord_write = list(['particle y coordinate'])
            for f in frame_range:
                try:
                    xcoord_write.append(particle_coordinates[particle_id][f][0])
                    ycoord_write.append(particle_coordinates[particle_id][f][1])
                except KeyError:
                    xcoord_write.append('')
                    ycoord_write.append('')
            writer.writerow(xcoord_write)
            writer.writerow(ycoord_write)

            brightness_write = list(['particle brightness'])
            for f in frame_range:
                try:
                    brightness_write.append(particle_brightness[particle_id][f])
                except KeyError:
                    brightness_write.append('')
            writer.writerow(brightness_write)

            # lifetime = particle_lifetimes[particle_id]

            # start_frame = particle_start_frames[particle_id]
            # start_x, start_y = particle_start_coordinates[particle_id]

            # end_frame = particle_end_frames[particle_id]
            # end_x, end_y = particle_end_coordinates[particle_id]
            # brightnesses = particle_brightness[particle_id]

            # total_data = np.array([particle_id, lifetime, start_frame, start_x, start_y, end_frame, end_x, end_y])
            # for b in brightnesses:
            #     total_data = np.append(total_data, b)
            # print(total_data)
            # writer.writerow(total_data)
    print(f"Results written to {output_csv}")
    return

