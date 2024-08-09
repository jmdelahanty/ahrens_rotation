import h5py
import cv2
import os
from tabulate import tabulate
from pathlib import Path

def read_valve_timestamps(h5_filename=None, video_filename=None):
    if not os.path.exists(str(h5_filename)):
        print(f"Error: H5 file '{h5_filename}' not found.")
        return

    # if not os.path.exists(video_filename):
    #     print(f"Error: Video file '{video_filename}' not found.")
    #     return

    # # Read frame rate from video file
    # video = cv2.VideoCapture(str(video_filename))
    # frame_rate = video.get(cv2.CAP_PROP_FPS)
    # video.release()

    # print(f"Video frame rate: {frame_rate:.2f} fps")

    with h5py.File(h5_filename, 'r') as f:
        print(f"H5 file opened: {h5_filename}")

        # Print all attributes of the file
        print("\nFile Attributes:")
        for key, value in f.attrs.items():
            print(f"{key}: {value}")
        
        if 'valve_events' not in f:
            print("Error: 'valve_events' dataset not found in the H5 file.")
            return

        dataset = f['valve_events']
        
        intended_fps = f.attrs.get('intended_fps')
        # if intended_fps:
        #     print(f"Intended FPS: {intended_fps}")
        # if intended_fps and intended_fps != frame_rate:
        #     print(f"Warning: Intended frame rate ({intended_fps} fps) differs from video frame rate ({frame_rate:.2f} fps)")
        
        experiment_start_time = f.attrs.get('experiment_start_time')
        if experiment_start_time:
            print(f"Experiment start time: {experiment_start_time}")
        
        if len(dataset) == 0:
            print("No valve events recorded.")
            return

        # Get the first timestamp to use as a reference
        first_timestamp = dataset[0]['timestamp']
        
        print("\nValve Events:")
        
        table_data = []
        for event in dataset:
            event_type = event['event'].decode('utf-8')
            stim_number = event['stim_number']
            pulse_number = event['pulse_number']
            timestamp = event['timestamp']
            relative_time = timestamp - first_timestamp
            # estimated_frame = int(relative_time * frame_rate)
            
            stim_str = f"{stim_number}" if stim_number != -1 else "N/A"
            pulse_str = f"{pulse_number}" if pulse_number != -1 else "N/A"
            
            table_data.append([
                event_type,
                stim_str,
                pulse_str,
                f"{timestamp:.6f}",
                f"+{relative_time:.6f}s"
            ])
            #     estimated_frame
            # ])

        headers = ["Event", "Stim", "Pulse", "Time (s)", "Relative Time", "Est. Frame"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

        print("\nSummary:")
        print(f"Total events: {len(dataset)}")
        print(f"Duration: {dataset[-1]['timestamp'] - first_timestamp:.2f} seconds")
        # print(f"Estimated total frames: {int((dataset[-1]['timestamp'] - first_timestamp) * frame_rate)}")

if __name__ == "__main__":
    read_valve_timestamps(
        h5_filename = Path("/home/jmdelahanty/OneDrive/lgfl/rawdata/sub-0001/20240718_sub-0001_OnePortEtohExperiment/20240718_sub-0001_OnePortEtohExperiment.h5"),
        video_filename = Path("/home/jmdelahanty/OneDrive/lgfl/rawdata/sub-0001/20240718_sub-0001_OnePortEtohExperiment/20240718_sub-0001_OnePortEtohExperiment.mp4")
    )