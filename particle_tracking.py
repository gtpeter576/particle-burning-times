import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import csv

def track_particle_burns(tiff_folder_path, threshold, frame_rate, max_distance=35, search_previous_frames=1, stop=False, stop_frame=200, verbose=True, brightness_mode='mean'):
    # Get a sorted list of all TIFF files in the folder
    tiff_files = sorted(glob.glob(os.path.join(tiff_folder_path, "*.tif")))
    if not tiff_files:
        print("Error: No TIFF files found in the specified folder.")
        return

    # Dictionary to store particle burn durations, start/end coordinates, start/end frames, brightnesses and sizes
    particle_coordinates = {}
    particle_brightness = {}
    next_particle_id = 0
    frame_counter = 0


    for tiff_file in tiff_files:

        if stop and frame_counter > stop_frame:
            break
        if verbose and frame_counter % 1000 == 0:
            print(f"Processing frame {frame_counter}/{len(tiff_files)}: {tiff_file}")

        # Read the TIFF image as grayscale
        frame = cv2.imread(tiff_file, cv2.IMREAD_ANYDEPTH)
        
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
            #exit this loop iteration if the particle is too old
            if frame_counter - max(particle_coordinates[particle_id].keys()) > search_previous_frames:
                continue
            (px, py) = particle_coordinates[particle_id][max(particle_coordinates[particle_id].keys())]
            predicted_x, predicted_y = (px, py)
            if len(list(particle_coordinates[particle_id].keys())) >= 2:
                (ppx, ppy) = particle_coordinates[particle_id][sorted(particle_coordinates[particle_id].keys())[-2]]
                #simple linear motion prediction
                predicted_x, predicted_y = (px + (px - ppx), py + (py - ppy))
            
            best_match = None
            best_distance = max_distance*(frame_counter - max(particle_coordinates[particle_id].keys()))  # Initialize with a large distance based on the frame difference
            
            for i, (cx, cy, w, h) in enumerate(current_centroids):
                distance = np.sqrt((cx - predicted_x) ** 2 + (cy - predicted_y) ** 2) #calculate distance
                #Find closest particle to previous particle
                if distance < best_distance and i in unmatched_centroids:
                    best_match = i
                    best_distance = distance

            # If a match is found, update the particle position, end coordinates, end frame and lifetime
            if best_match is not None:
                #uncomment for more verbose output
                #print("new centroid matched to a particle", frame_counter - particle_end_frames[particle_id], "frames prior to current frame")
                (cx, cy, w, h) = current_centroids[best_match]
                coords = (cx, cy)
                particle_coordinates[particle_id][frame_counter] = coords
                particle_brightness[particle_id][frame_counter] = np.array(frame[cy-2:cy+2, cx-2:cx+2]).mean()
                unmatched_centroids.discard(best_match)

        # Assign new IDs to unmatched centroids
        # Must have at least one pixel within 3 pixels of the center greater than the threshold for the new centroid to count as a new particle
        for i in unmatched_centroids:
            (cx, cy, w, h) = current_centroids[i]
            coords = (cx, cy)
            #uncomment for more verbose output
            #print(f"New particle detected at frame {frame_counter}: {coords}; id: {next_particle_id}")
            particle_coordinates[next_particle_id] = {}
            particle_coordinates[next_particle_id][frame_counter] = coords

            particle_brightness[next_particle_id] = {}
            if brightness_mode == 'max':
                particle_brightness[next_particle_id][frame_counter] = np.array(frame[cy-2:cy+2, cx-2:cx+2]).max()
            else:
                particle_brightness[next_particle_id][frame_counter] = np.array(frame[cy-2:cy+2, cx-2:cx+2]).mean()

            next_particle_id += 1
        frame_counter += 1

    cv2.destroyAllWindows()
    
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
        for particle_id in particle_coordinates.keys():

            frame_range = np.array(list(particle_coordinates[particle_id].keys()))
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
    print(f"Results written to {output_csv}")
    return
