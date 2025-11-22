import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import csv
from scipy.optimize import linear_sum_assignment
import time
import image_calibration as ic

class Particle:
    def __init__(self, frame, pos, brightness, size):
        self.history = {}
        self.history[frame] = {}
        self.current_frame = frame
        self.history[frame]['pos'] = pos
        self.history[frame]['brightness'] = brightness
        self.history[frame]['size'] = size
        self.missed_frames = 0

    def update(self, frame, pos, brightness, size):
        self.current_frame = frame
        self.history[frame] = {}
        self.history[frame]['pos'] = pos
        self.history[frame]['brightness'] = brightness
        self.history[frame]['size'] = size
        self.missed_frames = 0
    
    def predict(self, predict_frame, fps):
        
        frame_list = sorted(list(self.history.keys()))

        #predict the position at (input) frame based on current velocity
        current_pos = self.history[self.current_frame]['pos']
        if len(frame_list) < 2:
            x_pos, y_pos = current_pos
            predicted_pos = (x_pos, y_pos-3) #predict initial speed is 1m/s, or about 3 pix per frame
        else:
            prev_x, prev_y = self.history[frame_list[-2]]['pos']
            frame_diff_forward = predict_frame - self.current_frame
            frame_diff_back = self.current_frame - frame_list[-2]
            vel_x = (current_pos[0] - prev_x)/frame_diff_back
            vel_y = (current_pos[1] - prev_y)/frame_diff_back
            predicted_x = current_pos[0] + vel_x*frame_diff_forward
            predicted_y = current_pos[1] + vel_y*frame_diff_forward
            predicted_pos = (predicted_x, predicted_y)
        
        #predict the brightness at (input) frame based on the three previous brightnesses
        brightness_rates_25k = [20000, 10000, 2000, -2000, -1000, -1000] #pixel counts per frame at 25000 fps, MUST CONVERT TO PROPER FPS
        brightness_rates = []
        for i in range(frame_list[-1] - frame_list[0] + 1):
            frame_index_25k = i*25000/fps
            if frame_index_25k <= 4:
                interpolated_rate_25k = brightness_rates_25k[int(frame_index_25k)] + (frame_index_25k - int(frame_index_25k))*(brightness_rates_25k[int(frame_index_25k)+1] - brightness_rates_25k[int(frame_index_25k)])
                brightness_rates.append(interpolated_rate_25k*25000/fps)
            else:
                brightness_rates.append(-1000*25000/fps)
        current_brightness = self.history[self.current_frame]['brightness']
        avg_prev_brightness = current_brightness
        avg_frame_diff = 1
        avg_brightness_rate = 0
        if len(frame_list) < 5:
            avg_prev_brightness = current_brightness
            avg_frame_diff = predict_frame - self.current_frame
            initial_frame = frame_list[0]
            frame_index = self.current_frame - initial_frame
            avg_brightness_rate = brightness_rates[frame_index]
        else:
            avg_prev_brightness = (self.history[frame_list[-1]]['brightness'] + self.history[frame_list[-2]]['brightness'] + self.history[frame_list[-3]]['brightness'])/3
            avg_frame_diff = predict_frame - sum(frame_list[-3:])/3
            initial_frame = frame_list[0]
            frame_indexes = (np.array(frame_list) - initial_frame*np.ones(len(frame_list))).astype(int)
            # print(len(brightness_rates))
            # print(frame_indexes[-1], '\n')
            avg_brightness_rate = (brightness_rates[frame_indexes[-3]] + brightness_rates[frame_indexes[-2]] + brightness_rates[frame_indexes[-1]])/3
        predicted_brightness = avg_prev_brightness + avg_brightness_rate*avg_frame_diff

        predicted_size = self.history[self.current_frame]['size']

        return predicted_pos, predicted_brightness, predicted_size
    

def track_particles(tiff_folder_path, threshold, max_distance, max_missed, fps, verbose=True, stop=False, stop_frame=2000, brightness_coeff=0.001, length_coeff=2):

    tiff_files = sorted(glob.glob(os.path.join(tiff_folder_path, "*.tif")))
    if not tiff_files:
        print("Error: No TIFF files found")
        return

    current_particles = []
    old_particles = []
    frame_counter = 0
    start_time = time.time()

    for path in tiff_files:
        if verbose and frame_counter % 1000 == 0:
            elapsed_time = time.time() - start_time
            print(f"Tracking particles in frame {frame_counter}/{len(tiff_files)}: {path}\nElapsed time: {int((elapsed_time - elapsed_time%60)/60)}mins {elapsed_time%60:.1f}s\n")
        if stop and frame_counter > stop_frame:
            break
        # read 16-bit original
        frame = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if frame is None:
            frame_counter += 1
            continue
        # ensure single-channel
        if frame.ndim == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame.copy()

        # threshold (threshold provided in same units as input)
        _, thresh = cv2.threshold(frame_gray, threshold, 65535, cv2.THRESH_BINARY)
        # convert to uint8 binary image for contours
        thresh8 = (thresh > 0).astype(np.uint8) * 255

        contours, _ = cv2.findContours(thresh8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        predictions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                area = 0.5*cv2.arcLength(cnt, True)
                if area == 0:
                    area = 1
                x, y, w, h = cv2.boundingRect(cnt)
                cx = x + w/2
                cy = y + h/2
            else:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            # brightness measured from original 16-bit frame, within bounding box mask
            x, y, w, h = cv2.boundingRect(cnt)
            x0, x1 = max(0, x), min(frame_gray.shape[1], x + w)
            y0, y1 = max(0, y), min(frame_gray.shape[0], y + h)
            mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
            cnt_shift = cnt.copy()
            cnt_shift[:, 0, 0] -= x0
            cnt_shift[:, 0, 1] -= y0
            cv2.drawContours(mask, [cnt_shift], -1, 1, -1)
            roi = frame_gray[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            mean_brightness = float(np.sum(roi * mask) / (mask.sum() + 1e-9))
            detections.append({'pos': (cx, cy), 'brightness': mean_brightness, 'size': area})
        
        for particle in current_particles:
            predicted_pos, predicted_brightness, predicted_size = particle.predict(frame_counter, fps)
            predictions.append({'pos': predicted_pos, 'brightness': predicted_brightness, 'size': predicted_size})
        
        # create cost matrix
        cost_matrix = np.zeros((len(predictions), len(detections)), dtype=np.float32)
        for i, pred in enumerate(predictions):
            for j, det in enumerate(detections):
                dist = np.linalg.norm(np.array(pred['pos']) - np.array(det['pos']))
                brightness_diff = abs(pred['brightness'] - det['brightness'])
                length_diff = abs(np.sqrt(pred['size']) - np.sqrt(det['size']))
                cost_matrix[i, j] = dist + brightness_coeff*brightness_diff + length_coeff*length_diff

        # solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] <= max_distance:
                current_particle = current_particles[r]
                det = detections[c]
                current_particle.update(frame_counter, det['pos'], det['brightness'], det['size'])
            else:
                current_particles[r].missed_frames += 1
                new_det = detections[c]
                new_particle = Particle(frame_counter, new_det['pos'], new_det['brightness'], new_det['size'])
                current_particles.append(new_particle)
        unmatched_predictions = set(range(len(predictions))) - set(row_ind)
        unmatched_detections = set(range(len(detections))) - set(col_ind)

        # handle unmatched predictions and detections
        for r in unmatched_predictions:
            current_particles[r].missed_frames += 1
        for c in unmatched_detections:
            new_det = detections[c]
            new_particle = Particle(frame_counter, new_det['pos'], new_det['brightness'], new_det['size'])
            current_particles.append(new_particle)
        
        # Remove particles that have missed too many frames
        for particle in current_particles:
            if particle.missed_frames >= max_missed:
                old_particles.append(particle)
        current_particles = [p for p in current_particles if p.missed_frames < max_missed]
        frame_counter += 1

    # Prepare CSV output path
    test_num = os.path.basename(os.path.normpath(tiff_folder_path))
    base_csv = os.path.join(tiff_folder_path, "..", test_num + "_particle_tracking_results.csv")
    output_csv = base_csv
    count = 1
    while os.path.exists(output_csv):
        output_csv = os.path.splitext(base_csv)[0] + f"({count})" + ".csv"
        count += 1

    # Write CSV in same structure as original script
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f'threshold={threshold}, max_distance={max_distance}, max_missed={max_missed}, brightness_coeff={brightness_coeff}, length_coeff={length_coeff}'])
        for id, particle in enumerate(old_particles):
            frames = sorted(particle.history.keys())
            writer.writerow([f'Particle {id}'])
            row = ['frame'] + frames
            writer.writerow(row)
            xrow = ['particle x coordinate'] + [particle.history[f]['pos'][0] for f in frames]
            yrow = ['particle y coordinate'] + [particle.history[f]['pos'][1] for f in frames]
            brow = ['particle brightness'] + [particle.history[f]['brightness'] for f in frames]
            srow = ['particle size'] + [particle.history[f]['size'] for f in frames]
            writer.writerow(xrow)
            writer.writerow(yrow)
            writer.writerow(brow)
            writer.writerow(srow)

    print(f"Results written to {output_csv}")
    return

if __name__ == '__main__':

    #ic.image_calibration(r"D:\20251110\test18\test18", r"D:\20251110\test18\AVG_bg.tif", r"D:\20251110\test18\test18_calibrated")

    tiff_folder_path = r"D:\20251110\test18\CODE_TESTING_CALIBRATED"
    thresh = 1000
    max_dist = 50
    max_missed = 2
    fps = 25000
    start_time = time.time()
    track_particles(tiff_folder_path, thresh, max_dist, max_missed, fps, stop=False, stop_frame=2000, brightness_coeff=0.0006, length_coeff=1.5)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Total elapsed time: {int((elapsed_time - elapsed_time%60)/60)}mins {elapsed_time%60:.1f}s')
