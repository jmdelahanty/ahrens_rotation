# Fish Position Tracking
# Script by Dhruv Zocchi PhD, Janelia Research Campus
# Updated by Jeremy Delahanty, Hopkins Neuroscience Program Rotation Student Ahrens Lab

# cv2 is the backbone of this processing pipeline
# In short, there's a workflow that performs background subtraction,
# thresholding, blob detection, and distance calculation all through
# opencv and numpy. Numpy is the numerical computation/optimized python distro
import cv2
import numpy as np

from exceptions import VideoReadError

from threading import Thread
from queue import Queue

import multiprocessing
import warnings

# Tkinter is a simple GUI library
from tkinter import Tk
from tkinter.filedialog import askdirectory

# Pathlib makes path manipulation a little nicer
from pathlib import Path

# tqdm gives you pretty progress bars
from tqdm.auto import tqdm

# We use time for pausing thread execution, logging, and progress bars
import time

# This workflow relies heavily upon multithreading so work for processing
# and writing data. It doesn't have to, but it was fun to learn about
# Queues are how threads can "put" data in a spot that's ready to be
# taken up by another thread!
import threading
import queue
from queue import Empty as QueueEmpty

# Multiprocessing allows you to have multiple, easily accessed, independent
# non-blocking python processes to enable parallelism
from joblib import Parallel, delayed

# h5py allows you to write to HDF5 files which are nice to use for organising
# multidimensional datasets with built in metadata/attributes
import h5py

# Currently, ROIs for a given set of experiments are stored as a config file
# stored as json
import json

# These are for debugging weird thread issues, not really in use now...
import traceback

def visualize_frame(original_frame, processed_frame, cX, cY, points):
    # Create a copy of the original frame for drawing
    vis_frame = original_frame.copy()
    
    # Draw dots for selected points
    for point_name, (x, y) in points.items():
        if x is not None and y is not None:
            cv2.circle(vis_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(vis_frame, point_name, (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw the detected fish position
    if not np.isnan(cX) and not np.isnan(cY):
        cv2.circle(vis_frame, (int(cX), int(cY)), 5, (0, 0, 255), -1)
    
    # Resize frames for display
    height, width = original_frame.shape[:2]
    new_height = 480
    new_width = int(width * (new_height / height))
    vis_frame_resized = cv2.resize(vis_frame, (new_width, new_height))
    processed_frame_resized = cv2.resize(processed_frame, (new_width, new_height))
    
    # Concatenate frames horizontally
    combined_frame = np.hstack((vis_frame_resized, processed_frame_resized))
    
    # Display the combined frame
    cv2.imshow('Fish Tracking Visualization', combined_frame)
    cv2.waitKey(1)  # Display frame for 1ms

def preprocess_video(video, syringe_contents):
    # pass video object
    # Set the video position to the first frame
    # We shouldn't enter this function unless the video can be opened.
    # Get the first image as the ROI selection after you're sure you've
    # gotten the frame successfully read
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # cap.read() tells you if the frame retrieval was successful (boolean)
    # through ret, frame is the numpy array array that opencv returns
    ret, frame = video.read()

    # If the frame is returned (if ret is true)
    if ret:
        # Get the image and convert it to a numpy array shaped as an image in grayscale
        first_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # select multiple points for each port in the chamber
    points = get_points(first_image, syringe_contents)
    
    # Now we want some other frame as background, you can technically pick any random
    # frame but it'd be cool in the future to have a static background image gathered for each
    # recording
    # Get random video frame, hoping that the fish has moved by now to a new position
    video.set(cv2.CAP_PROP_POS_FRAMES, 200)

    ret, frame = video.read()

    if ret:
        # Get the image and convert it to a numpy array shaped as an image in grayscale
        background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return points, background

def get_points(frame, syringe_contents):
    points = []
    copy = frame.copy()
    
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(copy, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(copy, f"Point {len(points)}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(window_name, copy)
            if len(points) >= 4:
                cv2.setMouseCallback(window_name, lambda *args : None)

    # Create a descriptive window name
    window_name = f"Port Selection - Left: {syringe_contents['left']}, Right: {syringe_contents['right']}"
    
    # Setting up the callback
    cv2.imshow(window_name, copy)
    cv2.setMouseCallback(window_name, click_event)

    # Provide instructions to the user
    print("Please select points in the following order:")
    print(f"1. Left ({syringe_contents['left']}) 2. Right ({syringe_contents['right']}) 3. Inlet 4. Outlet")

    while len(points) < 4:
        cv2.waitKey(1)

    cv2.destroyAllWindows()

    # Label the points based on syringe contents
    labeled_points = {
        'Left': points[0],
        'Right': points[1],
        'Inlet': points[2],
        'Outlet': points[3],
        'H2O': points[0] if syringe_contents['left'] == 'H2O' else points[1] if syringe_contents['right'] == 'H2O' else None,
        'EtOH': points[0] if syringe_contents['left'] == 'EtOH' else points[1] if syringe_contents['right'] == 'EtOH' else None,
        'empty': points[0] if syringe_contents['left'] == 'empty' else points[1] if syringe_contents['right'] == 'empty' else None
    }

    return labeled_points
    

class VideoConfig:
    def __init__(self, video_path, config_path):
        self.video_path = video_path
        self.config_path = config_path
        self.config_attrs = self._load_config_attrs()
        self.syringe_contents = self._determine_syringe_contents()

    def _load_config_attrs(self):
        try:
            with h5py.File(self.config_path, 'r') as h5f:
                return dict(h5f.attrs)
        except Exception as e:
            warnings.warn(f"Failed to load attributes from {self.config_path}: {str(e)}")
            return {}

    def _determine_syringe_contents(self):
        right_syringe = self.config_attrs.get('right_syringe')
        left_syringe = self.config_attrs.get('left_syringe')

        contents = {'left': 'unknown', 'right': 'unknown'}

        if right_syringe in ['EtOH', 'H2O', 'water']:
            contents['right'] = right_syringe
        if left_syringe in ['EtOH', 'H2O', 'water']:
            contents['left'] = left_syringe

        # Normalize 'water' to 'H2O'
        contents = {k: 'H2O' if v == 'water' else v for k, v in contents.items()}

        # Change 'unknown' to 'empty'
        contents = {k: 'empty' if v == 'unknown' else v for k, v in contents.items()}

        return contents

    def __repr__(self):
        return f"VideoConfig(video_path={self.video_path}, config_path={self.config_path}, syringe_contents={self.syringe_contents})"

def get_videos_and_configs(data_directory):
    # data_directory must be a pathlib Path object
    video_paths = sorted([path for path in data_directory.glob("lgfl/rawdata/*/*OnePort*/*.mp4")])
    config_paths = sorted([path for path in data_directory.glob("lgfl/rawdata/*/*OnePort*/*.h5")])
    
    video_configs = []
    for video_path in video_paths:
        matching_config = next((cfg for cfg in config_paths if cfg.stem == video_path.stem), None)
        
        if matching_config is None:
            warnings.warn(f"No matching config file found for video: {video_path}")
            continue
        
        video_config = VideoConfig(video_path, matching_config)
        video_configs.append(video_config)
    
    # Check for config files without matching videos
    for config_path in config_paths:
        if not any(vc.config_path == config_path for vc in video_configs):
            warnings.warn(f"Config file without matching video: {config_path}")
    
    return video_configs

def check_h5(video_path):
    # check if the path has an h5 file already. If it does we probably don't mean
    # to reanalyze it (unless we do at some point?)
    files = [file for file in video_path.parent.glob("*output.h5")]

def monitor_queues(data_queue, progress_queue, stop_event, interval=60.0):
    while not stop_event.is_set():
        data_queue_size = data_queue.qsize()
        progress_queue_size = progress_queue.qsize()
        print(f"[Queue Monitor] Data Queue Size: {data_queue_size}, Progress Queue Size: {progress_queue_size}")
        print(f"Data Queue Full: {data_queue.full()}, Progress Queue Full: {progress_queue.full()}")
        time.sleep(interval)

def get_total_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

def data_producer(video, data_queue, stop_event, initial_background, points, previous_location, total_frames, progress_queue):
    frame_count = 0
    max_retries = 10
    retry_count = 0
    background = initial_background
    background_update_interval = 250
    max_allowed_distance = 50  # Maximum allowed distance between consecutive points (in pixels)

    try:
        while not stop_event.is_set() and video.isOpened() and frame_count < total_frames:
            ret, frame = video.read()
            if not ret:
                while retry_count < max_retries:
                    print(f"Retrying frame {frame_count}, attempt {retry_count + 1}/{max_retries}")
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = video.read()
                    retry_count += 1
                    if ret:
                        break

            frame_count += 1   
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2GRAY)

            # Update background every 250 frames
            if frame_count % background_update_interval == 0:
                background = frame

            if not isinstance(background, np.ndarray):
                raise ValueError("Background frame (background) is not a numpy array.")
            if frame.shape != background.shape:
                raise ValueError(f"Video frame shape {frame.shape} does not match background frame shape {background.shape}.")
    
            subtracted = cv2.subtract(background, frame)
            blurred = cv2.GaussianBlur(subtracted, (3, 3), 0)
            _, thresholded = cv2.threshold(blurred, 20, 255, 0)
            inverted = cv2.bitwise_not(thresholded)
            median = cv2.medianBlur(thresholded, 7)
            contours, _ = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours_and_areas = [(contour, cv2.contourArea(contour)) for contour in contours if cv2.contourArea(contour) > 20]
            
            if filtered_contours_and_areas:
                filtered_contours = [contour for contour, area in [max(filtered_contours_and_areas, key=lambda x: x[1])]]
                contour = filtered_contours[0]
                moments = cv2.moments(contour)
                cX = round(int(moments["m10"] / moments["m00"]), 2)
                cY = round(int(moments["m01"] / moments["m00"]), 2)
                
                # Check distance from previous point
                if previous_location[0] is not None and previous_location[1] is not None:
                    distance = np.linalg.norm(np.array([cX, cY]) - np.array(previous_location))
                    if distance > max_allowed_distance:
                        print(f"Warning: Large jump detected at frame {frame_count}. Using previous location.")
                        cX, cY = previous_location
                
                previous_location[0], previous_location[1] = cX, cY
            else:
                if previous_location[0] is not None and previous_location[1] is not None:
                    cX, cY = previous_location
                else:
                    cX = np.nan
                    cY = np.nan
    
            dist_left = np.linalg.norm(np.array(points['Left']) - np.array([cX, cY]))
            dist_right = np.linalg.norm(np.array(points['Right']) - np.array([cX, cY]))

            data_queue.put((cX, cY, dist_left, dist_right))
            progress_queue.put(1)
    
    except VideoReadError as e:
        print(f"Video read error: {str(e)}")
        print(f"Last successfully read frame: {frame_count}")
        print(f"Video info: total_frames={total_frames}, isOpened={video.isOpened()}")
    except Exception as e:
        print(f"Unexpected error in producer: {str(e)}")
        print(traceback.format_exc())
    finally:
        stop_event.set()

def data_consumer(data_queue, stop_event, file_path, points, syringe_contents):
    with h5py.File(file_path, 'w') as h5f:
        dset_center = h5f.create_dataset('center', (0, 2), maxshape=(None, 2), dtype='f')
        
        # Create datasets based on syringe contents
        left_content = syringe_contents['left']
        right_content = syringe_contents['right']
        
        dset_dist_left = h5f.create_dataset(f'dist_{left_content}', (0,), maxshape=(None,), dtype='f')
        dset_dist_right = h5f.create_dataset(f'dist_{right_content}', (0,), maxshape=(None,), dtype='f')

        # Store syringe contents as attributes
        h5f.attrs['left_syringe'] = left_content
        h5f.attrs['right_syringe'] = right_content
        # Convert points dictionary to JSON string
        points_json = json.dumps({k: v if v is not None else 'None' for k, v in points.items()})
        h5f.attrs['points'] = points_json

        while not stop_event.is_set() or not data_queue.empty():
            try:
                cX, cY, dist_left, dist_right = data_queue.get(timeout=1)

                dset_center.resize(dset_center.shape[0] + 1, axis=0)
                dset_dist_left.resize(dset_dist_left.shape[0] + 1, axis=0)
                dset_dist_right.resize(dset_dist_right.shape[0] + 1, axis=0)

                dset_center[-1] = [cX, cY]
                dset_dist_left[-1] = dist_left
                dset_dist_right[-1] = dist_right

                data_queue.task_done()
            except queue.Empty:
                continue

def process_video(video_config, progress_queue, position):
    path = video_config.video_path
    syringe_contents = video_config.syringe_contents
    print(f"Processing: {path.stem}")

    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise ValueError("Failed to open the video file.")
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_name = Path(path).stem
    
    points, background = preprocess_video(video, syringe_contents)

    h5_filepath = path.parent / f"{vid_name}_output.h5"
    data_queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    previous_location = [None, None]
    
    producer_thread = threading.Thread(target=data_producer, args=(video, data_queue, stop_event, background, points, previous_location, total_frames, progress_queue))
    consumer_thread = threading.Thread(target=data_consumer, args=(data_queue, stop_event, h5_filepath, points, syringe_contents))

    progress_bar = tqdm(total=total_frames, desc=f'Process {path.stem}', position=position, leave=True, ncols=100)

    def update_progress():
        last_update = 0
        while not stop_event.is_set():
            try:
                progress = progress_queue.get(timeout=0.1)
                last_update += progress
                if last_update >= 10 or progress_queue.empty():  # Update every 10 frames or when queue is empty
                    progress_bar.update(last_update)
                    last_update = 0
            except queue.Empty:
                continue

    progress_thread = threading.Thread(target=update_progress)

    producer_thread.start()
    consumer_thread.start()
    progress_thread.start()

    try:
        producer_thread.join()
        consumer_thread.join()
        stop_event.set()  # Ensure the progress thread stops
        progress_thread.join()
    except KeyboardInterrupt:
        print("Stopping...")
        stop_event.set()
    finally:
        video.release()
        cv2.destroyAllWindows()
        progress_bar.close()
        print(f"Data processing completed for {path.stem}.")

def process_single_video(video_config, data_directory, position):
    progress_queue = multiprocessing.Queue()
    process_video(video_config, progress_queue, position)


# def main():
#     Tk().withdraw()
#     data_directory = Path("~/OneDrive").expanduser().resolve()
#     video_configs = get_videos_and_configs(data_directory)

#     num_cores = multiprocessing.cpu_count() - 1
#     num_jobs = min(8, num_cores)

#     # Create a manager for progress bars
#     manager = multiprocessing.Manager()
#     progress_bars = manager.list([None] * len(video_configs))

#     def init_tqdm():
#         tqdm.set_lock(multiprocessing.RLock())
        
#     Parallel(n_jobs=num_jobs, verbose=0)(
#         delayed(process_single_video)(video_config, data_directory, i) 
#         for i, video_config in enumerate(video_configs)
#     )

def main():
    Tk().withdraw()
    data_directory = Path("~/OneDrive").expanduser().resolve()
    video_configs = get_videos_and_configs(data_directory)

    # Select a specific video config
    # For example, let's select the first video config in the list
    selected_video_config = video_configs[-2]

    # Process the selected video
    process_single_video(selected_video_config, data_directory, 0)

if __name__ == "__main__":
    main()