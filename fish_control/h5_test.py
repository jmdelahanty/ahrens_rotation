import h5py
import numpy as np
import cv2
import os

def read_valve_timestamps(h5_filename='valve_timestamps.h5', video_filename='experiment.avi'):
    if not os.path.exists(h5_filename):
        print(f"Error: H5 file '{h5_filename}' not found.")
        return

    if not os.path.exists(video_filename):
        print(f"Error: Video file '{video_filename}' not found.")
        return

    # Read frame rate from video file
    video = cv2.VideoCapture(video_filename)
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    video.release()

    print(f"Video frame rate: {frame_rate} fps")

    with h5py.File(h5_filename, 'r') as f:
        if 'valve_events' not in f:
            print("Error: 'valve_events' dataset not found in the H5 file.")
            return

        dataset = f['valve_events']
        intended_fps = f.attrs.get('intended_fps')
        if intended_fps and intended_fps != frame_rate:
            print(f"Warning: Intended frame rate ({intended_fps} fps) differs from video frame rate ({frame_rate} fps)")
        
        if len(dataset) == 0:
            print("No valve events recorded.")
            return

        # Get the first timestamp to use as a reference
        first_timestamp = dataset[0]['timestamp']
        
        print("Valve Events:")
        print("-------------")
        print("Event | Time (s) | Relative Time | Est. Frame")
        print("------|----------|---------------|------------")
        
        for event in dataset:
            event_type = event['event'].decode('utf-8')
            timestamp = event['timestamp']
            relative_time = timestamp - first_timestamp
            estimated_frame = int(relative_time * frame_rate)
            
            print(f"{event_type:<5} | {timestamp:.6f} | +{relative_time:.6f}s | {estimated_frame}")

        print("\nSummary:")
        print(f"Total events: {len(dataset)}")
        print(f"Duration: {dataset[-1]['timestamp'] - first_timestamp:.2f} seconds")
        print(f"Estimated total frames: {int((dataset[-1]['timestamp'] - first_timestamp) * frame_rate)}")

if __name__ == "__main__":
    read_valve_timestamps('valve_timestamps.h5', '20240703_experiment.avi')