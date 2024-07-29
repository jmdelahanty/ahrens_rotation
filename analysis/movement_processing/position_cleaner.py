import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy import interpolate
import cv2
import pathlib
from pathlib import Path
from typing import List
from tqdm import tqdm

def correct_fish_position(h5_file_path: Path, x_threshold: float = None, y_threshold: float = None, method: str = 'linear'):
    """
    Load fish coordinates from an h5 file, identify frames where x and y is below a threshold,
    allow manual correction of these frames, and save the corrected data.

    Args:
    h5_file_path (Path): Path to the h5 file containing fish tracking data.
    y_threshold (float): Y-axis threshold for identifying problematic frames.
    method (str): Interpolation method ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
    """
    # Load data
    with h5py.File(h5_file_path, 'r') as f:
        center = f['center'][:]
    
    df = pd.DataFrame(center, columns=['x', 'y'])
    frames = np.arange(len(df))
    
    # Interpolate NaN values
    mask_x, mask_y = ~np.isnan(df['x']), ~np.isnan(df['y'])
    f_x = interpolate.interp1d(frames[mask_x], df['x'][mask_x], kind=method, fill_value="extrapolate")
    f_y = interpolate.interp1d(frames[mask_y], df['y'][mask_y], kind=method, fill_value="extrapolate")
    df_interpolated = pd.DataFrame({'x': f_x(frames), 'y': f_y(frames)})
    
    # Identify frames below threshold
    below_threshold_y = df_interpolated['y'] < y_threshold
    below_threshold_x = df_interpolated['x'] < x_threshold if x_threshold else np.zeros_like(below_threshold_y)
    frames_below_threshold = frames[below_threshold_y & below_threshold_x]
    
    print(f"Number of frames where y < {y_threshold}: {len(frames_below_threshold)}")
    
    # Find the corresponding video file
    video_file = list(h5_file_path.parent.glob("*.mp4"))
    if not video_file:
        print("No corresponding video file found.")
        return
    
    video_file = video_file[0]
    cap = cv2.VideoCapture(str(video_file))
    
    # Function to handle mouse clicks
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print(f'x = {ix}, y = {iy}')
        
        # Update the dataframe with the new coordinates
        df_interpolated.loc[frame_number, 'x'] = ix
        df_interpolated.loc[frame_number, 'y'] = iy
        
        # Close the figure to move to the next frame
        plt.close()
    
    # Correct frames
    for frame_number in frames_below_threshold:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Frame {frame_number} - Click to correct position")
            
            # Plot current position
            ax.plot(df_interpolated.loc[frame_number, 'x'], df_interpolated.loc[frame_number, 'y'], 'ro', markersize=10)
            
            # Connect the onclick function
            fig.canvas.mpl_connect('button_press_event', onclick)
            
            plt.show()
    
    cap.release()
    
    # Save corrected data
    corrected_file_path = h5_file_path.with_name(f"{h5_file_path.stem}_corrected.h5")
    with h5py.File(corrected_file_path, 'w') as f:
        f.create_dataset('center', data=df_interpolated.values)
    
    print(f"Corrected data saved to {corrected_file_path}")
    
    return df_interpolated

def get_output_h5_files(data_directory: pathlib.Path) -> List[pathlib.Path]:
    """
    Find all output h5 files in the given data directory.

    Args:
    data_directory (pathlib.Path): The root directory to search for output files.

    Returns:
    List[pathlib.Path]: A list of paths to the output h5 files.
    """
    # Pattern to match: any file ending with '_output.h5'
    pattern = '*_output.h5'
    
    # Use glob to find all matching files recursively
    output_files = list(data_directory.rglob(pattern))
    
    # Sort the files to ensure consistent ordering
    output_files.sort()
    
    print(f"Found {len(output_files)} output h5 files.")
    
    return output_files

# def correct_fish_position_with_roi(h5_file_path: Path, method: str = 'linear', consecutive_threshold: int = 5, display_interval: int = 5):
#     """
#     Load fish coordinates from an h5 file, apply automated contour detection,
#     allow user to select a region of interest (ROI) for manual correction,
#     and save the corrected data.

#     Args:
#     h5_file_path (Path): Path to the h5 file containing fish tracking data.
#     method (str): Interpolation method ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
#     consecutive_threshold (int): Number of consecutive identical positions to trigger automated correction.
#     display_interval (int): Number of frames to process before updating the display in automated correction.

#     Returns:
#     pd.DataFrame: Dataframe with corrected fish positions.
#     """
#     # Load data
#     with h5py.File(h5_file_path, 'r') as f:
#         center = f['center'][:]
    
#     df = pd.DataFrame(center, columns=['x', 'y'])
#     frames = np.arange(len(df))
    
#     # Find the corresponding video file
#     video_file = list(h5_file_path.parent.glob("*.mp4"))
#     if not video_file:
#         print("No corresponding video file found.")
#         return df
#     video_file = video_file[0]

#     # Apply automated contour detection
#     print("Applying automated contour detection...")
#     df_auto_corrected = automate_contour_detection(h5_file_path, video_file, consecutive_threshold, display_interval)

#     # Interpolate NaN values in the auto-corrected data
#     mask_x, mask_y = ~np.isnan(df_auto_corrected['x']), ~np.isnan(df_auto_corrected['y'])
#     f_x = interpolate.interp1d(frames[mask_x], df_auto_corrected['x'][mask_x], kind=method, fill_value="extrapolate")
#     f_y = interpolate.interp1d(frames[mask_y], df_auto_corrected['y'][mask_y], kind=method, fill_value="extrapolate")
#     df_interpolated = pd.DataFrame({'x': f_x(frames), 'y': f_y(frames)})

#     # Plot for ROI selection
#     fig, ax = plt.subplots(figsize=(10, 8))
#     scatter = ax.scatter(df_interpolated['x'], df_interpolated['y'], alpha=0.5)
#     ax.set_title('Select Region of Interest for Manual Correction')
#     ax.set_xlabel('X coordinate')
#     ax.set_ylabel('Y coordinate')
#     ax.invert_yaxis()
    
#     roi_points = []
    
#     def line_select_callback(eclick, erelease):
#         x1, y1 = eclick.xdata, eclick.ydata
#         x2, y2 = erelease.xdata, erelease.ydata
#         nonlocal roi_points
#         roi_points = [min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)]
#         print(f"ROI selected: x1={roi_points[0]:.2f}, x2={roi_points[1]:.2f}, y1={roi_points[2]:.2f}, y2={roi_points[3]:.2f}")
    
#     rs = RectangleSelector(ax, line_select_callback, useblit=True,
#                            button=[1], minspanx=5, minspany=5,
#                            spancoords='pixels', interactive=True)
    
#     plt.show()
    
#     if not roi_points:
#         print("No ROI selected. Skipping manual correction.")
#         return df_interpolated

#     # Identify frames within ROI
#     in_roi = ((df_interpolated['x'] >= roi_points[0]) & (df_interpolated['x'] <= roi_points[1]) &
#               (df_interpolated['y'] >= roi_points[2]) & (df_interpolated['y'] <= roi_points[3]))
#     frames_to_correct = frames[in_roi]
    
#     print(f"Number of frames to manually correct: {len(frames_to_correct)}")
    
#     cap = cv2.VideoCapture(str(video_file))
    
#     # Function to handle mouse clicks for manual correction
#     def onclick(event):
#         nonlocal ix, iy
#         ix, iy = event.xdata, event.ydata
#         print(f'Selected coordinates: x = {ix:.2f}, y = {iy:.2f}')
#         plt.close()
    
#     # Manual correction for frames within ROI
#     for frame_number in frames_to_correct:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
#         ret, frame = cap.read()
#         if ret:
#             fig, ax = plt.subplots(figsize=(10, 8))
#             ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#             ax.set_title(f"Frame {frame_number} - Click to correct position")
            
#             # Plot current position
#             ax.plot(df_interpolated.loc[frame_number, 'x'], df_interpolated.loc[frame_number, 'y'], 'ro', markersize=10)
            
#             # Initialize ix and iy
#             ix, iy = df_interpolated.loc[frame_number, 'x'], df_interpolated.loc[frame_number, 'y']
            
#             # Connect the onclick function
#             fig.canvas.mpl_connect('button_press_event', onclick)
            
#             plt.show()
            
#             # Update coordinates
#             df_interpolated.loc[frame_number, 'x'] = ix
#             df_interpolated.loc[frame_number, 'y'] = iy
#             print(f"Updated coordinates: x = {ix:.2f}, y = {iy:.2f}")
    
#     cap.release()
    
#     # Save corrected data
#     corrected_file_path = h5_file_path.with_name(f"{h5_file_path.stem}_corrected.h5")
#     with h5py.File(corrected_file_path, 'w') as f:
#         f.create_dataset('center', data=df_interpolated.values)
    
#     print(f"Corrected data saved to {corrected_file_path}")
    
#     # Plot the original and corrected data
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
#     ax1.scatter(center[:, 0], center[:, 1], alpha=0.5)
#     ax1.set_title('Original Data')
#     ax1.set_xlabel('X coordinate')
#     ax1.set_ylabel('Y coordinate')
#     ax1.invert_yaxis()
    
#     ax2.scatter(df_interpolated['x'], df_interpolated['y'], alpha=0.5)
#     ax2.set_title('Corrected Data')
#     ax2.set_xlabel('X coordinate')
#     ax2.set_ylabel('Y coordinate')
#     ax2.invert_yaxis()
    
#     # Plot ROI on both axes
#     for ax in [ax1, ax2]:
#         ax.add_patch(plt.Rectangle((roi_points[0], roi_points[2]), 
#                                    roi_points[1] - roi_points[0], 
#                                    roi_points[3] - roi_points[2], 
#                                    fill=False, edgecolor='r'))
    
#     plt.tight_layout()
#     plt.show()
    
#     return df_interpolated

def correct_fish_position_with_roi(h5_file_path: Path, method: str = 'linear', visualize: bool = True):
    # Load data
    with h5py.File(h5_file_path, 'r') as f:
        center = f['center'][:]
    
    df = pd.DataFrame(center, columns=['x', 'y'])
    frames = np.arange(len(df))
    
    # Interpolate NaN values
    mask_x, mask_y = ~np.isnan(df['x']), ~np.isnan(df['y'])
    f_x = interpolate.interp1d(frames[mask_x], df['x'][mask_x], kind=method, fill_value="extrapolate")
    f_y = interpolate.interp1d(frames[mask_y], df['y'][mask_y], kind=method, fill_value="extrapolate")
    df_interpolated = pd.DataFrame({'x': f_x(frames), 'y': f_y(frames)})

    # Plot for ROI selection
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(df_interpolated['x'], df_interpolated['y'], alpha=0.5)
    ax.set_title('Select Region of Interest')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.invert_yaxis()
    
    roi_points = []
    
    def line_select_callback(eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        nonlocal roi_points
        roi_points = [min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)]
        print(f"ROI selected: x1={roi_points[0]:.2f}, x2={roi_points[1]:.2f}, y1={roi_points[2]:.2f}, y2={roi_points[3]:.2f}")
    
    rs = RectangleSelector(ax, line_select_callback, useblit=True,
                           button=[1], minspanx=5, minspany=5,
                           spancoords='pixels', interactive=True)
    
    plt.show()
    
    if not roi_points:
        print("No ROI selected. Exiting.")
        return df_interpolated

    # Identify frames within ROI
    in_roi = ((df_interpolated['x'] >= roi_points[0]) & (df_interpolated['x'] <= roi_points[1]) &
              (df_interpolated['y'] >= roi_points[2]) & (df_interpolated['y'] <= roi_points[3]))
    frames_to_correct = frames[in_roi]
    
    print(f"Number of frames to correct: {len(frames_to_correct)}")
    
    # Find the corresponding video file
    video_file = list(h5_file_path.parent.glob("*.mp4"))
    if not video_file:
        print("No corresponding video file found.")
        return df_interpolated
    
    video_file = video_file[0]
    cap = cv2.VideoCapture(str(video_file))
    
    # Get background frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)  # Assuming frame 200 is a good background
    ret, background = cap.read()
    if not ret:
        print("Could not read background frame.")
        return df_interpolated
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    def process_frame(frame, background, roi):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subtracted = cv2.subtract(background, frame_gray)
        blurred = cv2.GaussianBlur(subtracted, (3, 3), 0)
        _, thresholded = cv2.threshold(blurred, 20, 255, 0)
        inverted = cv2.bitwise_not(thresholded)
        median = cv2.medianBlur(thresholded, 7)
        
        # Create a mask to exclude the ROI
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        cv2.rectangle(mask, (int(roi[0]), int(roi[2])), (int(roi[1]), int(roi[3])), 0, -1)
        
        # Apply the mask to the median image
        masked_median = cv2.bitwise_and(median, median, mask=mask)
        
        contours, _ = cv2.findContours(masked_median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours_and_areas = [(contour, cv2.contourArea(contour)) for contour in contours if cv2.contourArea(contour) > 20]
        
        cX, cY = None, None
        largest_contour = None
        if filtered_contours_and_areas:
            largest_contour = max(filtered_contours_and_areas, key=lambda x: x[1])[0]
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cX = round(int(moments["m10"] / moments["m00"]), 2)
                cY = round(int(moments["m01"] / moments["m00"]), 2)
        
        processed_frame_color = None
        if visualize:
            # Create a color version of the processed frame for contour visualization
            processed_frame_color = cv2.cvtColor(masked_median, cv2.COLOR_GRAY2BGR)
            if largest_contour is not None:
                cv2.drawContours(processed_frame_color, [largest_contour], -1, (0, 255, 0), 2)
            
            # Draw the ROI on the processed frame
            cv2.rectangle(processed_frame_color, (int(roi[0]), int(roi[2])), (int(roi[1]), int(roi[3])), (255, 0, 0), 2)
        
        return cX, cY, processed_frame_color

    if visualize:
        # Set up the display
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        ax1.set_title("Original Frame")
        ax2.set_title("Processed Frame with Contours")
        im1 = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        im2 = ax2.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        orig_point, = ax1.plot([], [], 'bo', markersize=10)
        new_point, = ax1.plot([], [], 'ro', markersize=10)
        proc_orig_point, = ax2.plot([], [], 'bo', markersize=10)
        proc_new_point, = ax2.plot([], [], 'ro', markersize=10)
        plt.tight_layout()
    
    # Correct frames
    for frame_number in tqdm(frames_to_correct, desc="Processing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            cX, cY, processed_frame = process_frame(frame, background, roi_points)
            
            if cX is not None and cY is not None:
                df_interpolated.loc[frame_number, 'x'] = cX
                df_interpolated.loc[frame_number, 'y'] = cY
            
            if visualize:
                # Update the display
                im1.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                im2.set_data(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                orig_point.set_data(df_interpolated.loc[frame_number, 'x'], df_interpolated.loc[frame_number, 'y'])
                proc_orig_point.set_data(df_interpolated.loc[frame_number, 'x'], df_interpolated.loc[frame_number, 'y'])
                
                if cX is not None and cY is not None:
                    new_point.set_data(cX, cY)
                    proc_new_point.set_data(cX, cY)
                else:
                    new_point.set_data([], [])
                    proc_new_point.set_data([], [])
                
                plt.draw()
                plt.pause(0.001)  # Add a small pause to allow the plot to update
        else:
            print(f"Could not read frame {frame_number}")
    
    if visualize:
        plt.ioff()  # Turn off interactive mode
        plt.close()
    
    cap.release()
    
    # Save corrected data
    corrected_file_path = h5_file_path.with_name(f"{h5_file_path.stem}_corrected.h5")
    with h5py.File(corrected_file_path, 'w') as f:
        f.create_dataset('center', data=df_interpolated.values)
    
    print(f"Corrected data saved to {corrected_file_path}")
    
    # Plot the original and corrected data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.scatter(center[:, 0], center[:, 1], alpha=0.5)
    ax1.set_title('Original Data')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.invert_yaxis()
    
    ax2.scatter(df_interpolated['x'], df_interpolated['y'], alpha=0.5)
    ax2.set_title('Corrected Data')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.invert_yaxis()
    
    # Plot ROI on both axes
    for ax in [ax1, ax2]:
        ax.add_patch(plt.Rectangle((roi_points[0], roi_points[2]), 
                                   roi_points[1] - roi_points[0], 
                                   roi_points[3] - roi_points[2], 
                                   fill=False, edgecolor='r'))
    
    plt.tight_layout()
    plt.show()
    
    return df_interpolated

def manual_correction_with_roi(h5_file_path: Path):
    """
    Manually correct fish positions for frames within a user-selected ROI and update the H5 file.

    Args:
    h5_file_path (Path): Path to the corrected H5 file.

    Returns:
    None
    """
    # Load the corrected data
    with h5py.File(h5_file_path, 'r+') as f:
        center = f['center'][:]
    
    df = pd.DataFrame(center, columns=['x', 'y'])

    # Find the corresponding video file
    video_file = list(h5_file_path.parent.glob("*.mp4"))
    if not video_file:
        print("No corresponding video file found.")
        return
    
    video_file = video_file[0]
    cap = cv2.VideoCapture(str(video_file))

    # Function to select ROI
    def select_roi():
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(df['x'], df['y'], alpha=0.5)
        ax.set_title('Select Region of Interest for Manual Correction')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.invert_yaxis()

        roi_coords = {}

        def line_select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            roi_coords['x1'], roi_coords['x2'] = sorted([x1, x2])
            roi_coords['y1'], roi_coords['y2'] = sorted([y1, y2])
            print(f"ROI selected: x1={roi_coords['x1']:.2f}, x2={roi_coords['x2']:.2f}, y1={roi_coords['y1']:.2f}, y2={roi_coords['y2']:.2f}")

        rs = RectangleSelector(ax, line_select_callback, useblit=True,
                               button=[1], minspanx=5, minspany=5,
                               spancoords='pixels', interactive=True)
        
        plt.show()
        return roi_coords

    roi = select_roi()

    # Identify frames within ROI
    frames_to_correct = df[(df['x'] >= roi['x1']) & (df['x'] <= roi['x2']) &
                           (df['y'] >= roi['y1']) & (df['y'] <= roi['y2'])].index.tolist()

    print(f"Number of frames to correct: {len(frames_to_correct)}")

    def onclick(event):
        nonlocal ix, iy
        ix, iy = event.xdata, event.ydata
        plt.close()

    for frame_number in frames_to_correct:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Frame {frame_number} - Click to correct position")
            
            # Plot current position
            ax.plot(df.loc[frame_number, 'x'], df.loc[frame_number, 'y'], 'ro', markersize=10)
            
            # Initialize ix and iy
            ix, iy = df.loc[frame_number, 'x'], df.loc[frame_number, 'y']
            
            # Connect the onclick function
            fig.canvas.mpl_connect('button_press_event', onclick)
            
            plt.show()
            
            # Update the dataframe with the new coordinates
            df.loc[frame_number, 'x'] = ix
            df.loc[frame_number, 'y'] = iy
            print(f"Updated coordinates for frame {frame_number}: x = {ix:.2f}, y = {iy:.2f}")
        else:
            print(f"Could not read frame {frame_number}")

    cap.release()

    # Save the updated data back to the H5 file
    with h5py.File(h5_file_path, 'r+') as f:
        del f['center']  # Delete the existing dataset
        f.create_dataset('center', data=df.values)  # Create a new dataset with the updated values

    print(f"Manual corrections saved to {h5_file_path}")

    # Plot the corrected data
    plt.figure(figsize=(12, 8))
    plt.scatter(df['x'], df['y'], alpha=0.5)
    plt.title('Corrected Data with Manual Adjustments')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.gca().invert_yaxis()
    
    # Draw the ROI
    plt.gca().add_patch(plt.Rectangle((roi['x1'], roi['y1']), 
                                      roi['x2'] - roi['x1'], 
                                      roi['y2'] - roi['y1'], 
                                      fill=False, edgecolor='r'))
    
    plt.tight_layout()
    plt.show()

def correct_fish_position_with_two_rois(h5_file_path: Path, method: str = 'linear', visualize: bool = True):
    # Load data
    with h5py.File(h5_file_path, 'r') as f:
        center = f['center'][:]
    
    df = pd.DataFrame(center, columns=['x', 'y'])
    frames = np.arange(len(df))
    
    # Interpolate NaN values
    mask_x, mask_y = ~np.isnan(df['x']), ~np.isnan(df['y'])
    f_x = interpolate.interp1d(frames[mask_x], df['x'][mask_x], kind=method, fill_value="extrapolate")
    f_y = interpolate.interp1d(frames[mask_y], df['y'][mask_y], kind=method, fill_value="extrapolate")
    df_interpolated = pd.DataFrame({'x': f_x(frames), 'y': f_y(frames)})

    # Find the corresponding video file
    video_file = list(h5_file_path.parent.glob("*.mp4"))
    if not video_file:
        print("No corresponding video file found.")
        return df_interpolated
    
    video_file = video_file[0]
    cap = cv2.VideoCapture(str(video_file))

    # Get the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Could not read the first frame of the video.")
        return df_interpolated

    # Function to select ROI
    def select_roi(title, frame):
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        scatter = ax.scatter(df_interpolated['x'], df_interpolated['y'], alpha=0.5, c='red', s=10)
        ax.set_title(title)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.invert_yaxis()
        
        roi_points = []
        
        def line_select_callback(eclick, erelease):
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            roi_points.append([min(x1, x2), max(x1, x2), min(y1, y2), max(y1, y2)])
            print(f"ROI selected: x1={roi_points[-1][0]:.2f}, x2={roi_points[-1][1]:.2f}, y1={roi_points[-1][2]:.2f}, y2={roi_points[-1][3]:.2f}")
        
        rs = RectangleSelector(ax, line_select_callback, useblit=True,
                               button=[1], minspanx=5, minspany=5,
                               spancoords='pixels', interactive=True)
        
        plt.show()
        return roi_points[0] if roi_points else None

    # Select two ROIs
    roi1 = select_roi('Select First Region of Interest', first_frame)
    roi2 = select_roi('Select Second Region of Interest', first_frame)

    if not roi1 and not roi2:
        print("No ROIs selected. Exiting.")
        return df_interpolated

    # Identify frames within ROIs
    in_roi1 = ((df_interpolated['x'] >= roi1[0]) & (df_interpolated['x'] <= roi1[1]) &
               (df_interpolated['y'] >= roi1[2]) & (df_interpolated['y'] <= roi1[3])) if roi1 else pd.Series(False, index=df_interpolated.index)
    in_roi2 = ((df_interpolated['x'] >= roi2[0]) & (df_interpolated['x'] <= roi2[1]) &
               (df_interpolated['y'] >= roi2[2]) & (df_interpolated['y'] <= roi2[3])) if roi2 else pd.Series(False, index=df_interpolated.index)
    
    frames_to_correct = frames[in_roi1 | in_roi2]
    print(f"Number of frames to correct: {len(frames_to_correct)}")

    # Get background frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 200)  # Assuming frame 200 is a good background
    ret, background = cap.read()
    if not ret:
        print("Could not read background frame.")
        return df_interpolated
    background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    
    def process_frame(frame, background, roi1, roi2):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subtracted = cv2.subtract(background, frame_gray)
        blurred = cv2.GaussianBlur(subtracted, (3, 3), 0)
        _, thresholded = cv2.threshold(blurred, 20, 255, 0)
        inverted = cv2.bitwise_not(thresholded)
        median = cv2.medianBlur(thresholded, 7)
        
        # Create a mask to exclude the ROIs
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        if roi1:
            cv2.rectangle(mask, (int(roi1[0]), int(roi1[2])), (int(roi1[1]), int(roi1[3])), 0, -1)
        if roi2:
            cv2.rectangle(mask, (int(roi2[0]), int(roi2[2])), (int(roi2[1]), int(roi2[3])), 0, -1)
        
        # Apply the mask to the median image
        masked_median = cv2.bitwise_and(median, median, mask=mask)
        
        contours, _ = cv2.findContours(masked_median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours_and_areas = [(contour, cv2.contourArea(contour)) for contour in contours if cv2.contourArea(contour) > 12]
        
        cX, cY = None, None
        largest_contour = None
        if filtered_contours_and_areas:
            largest_contour = max(filtered_contours_and_areas, key=lambda x: x[1])[0]
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                cX = round(int(moments["m10"] / moments["m00"]), 2)
                cY = round(int(moments["m01"] / moments["m00"]), 2)
        
        processed_frame_color = None
        if visualize:
            # Create a color version of the processed frame for contour visualization
            processed_frame_color = cv2.cvtColor(masked_median, cv2.COLOR_GRAY2BGR)
            if largest_contour is not None:
                cv2.drawContours(processed_frame_color, [largest_contour], -1, (0, 255, 0), 2)
            
            # Draw the ROIs on the processed frame
            if roi1:
                cv2.rectangle(processed_frame_color, (int(roi1[0]), int(roi1[2])), (int(roi1[1]), int(roi1[3])), (255, 0, 0), 2)
            if roi2:
                cv2.rectangle(processed_frame_color, (int(roi2[0]), int(roi2[2])), (int(roi2[1]), int(roi2[3])), (0, 0, 255), 2)
        
        return cX, cY, processed_frame_color

    if visualize:
        # Set up the display
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        ax1.set_title("Original Frame")
        ax2.set_title("Processed Frame with Contours")
        im1 = ax1.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        im2 = ax2.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        orig_point, = ax1.plot([], [], 'bo', markersize=10)
        new_point, = ax1.plot([], [], 'ro', markersize=10)
        proc_orig_point, = ax2.plot([], [], 'bo', markersize=10)
        proc_new_point, = ax2.plot([], [], 'ro', markersize=10)
        plt.tight_layout()
    
    # Correct frames
    for frame_number in tqdm(frames_to_correct, desc="Processing frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if ret:
            cX, cY, processed_frame = process_frame(frame, background, roi1, roi2)
            
            if cX is not None and cY is not None:
                df_interpolated.loc[frame_number, 'x'] = cX
                df_interpolated.loc[frame_number, 'y'] = cY
            
            if visualize:
                # Update the display
                im1.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                im2.set_data(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                orig_point.set_data(df_interpolated.loc[frame_number, 'x'], df_interpolated.loc[frame_number, 'y'])
                proc_orig_point.set_data(df_interpolated.loc[frame_number, 'x'], df_interpolated.loc[frame_number, 'y'])
                
                if cX is not None and cY is not None:
                    new_point.set_data(cX, cY)
                    proc_new_point.set_data(cX, cY)
                else:
                    new_point.set_data([], [])
                    proc_new_point.set_data([], [])
                
                plt.draw()
                plt.pause(0.001)  # Add a small pause to allow the plot to update
        else:
            print(f"Could not read frame {frame_number}")
    
    if visualize:
        plt.ioff()  # Turn off interactive mode
        plt.close()
    
    cap.release()
    
    # Save corrected data
    corrected_file_path = h5_file_path.with_name(f"{h5_file_path.stem}_corrected.h5")
    with h5py.File(corrected_file_path, 'w') as f:
        f.create_dataset('center', data=df_interpolated.values)
    
    print(f"Corrected data saved to {corrected_file_path}")
    
    # Plot the original and corrected data
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    ax1.scatter(center[:, 0], center[:, 1], alpha=0.5)
    ax1.set_title('Original Data')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.invert_yaxis()
    
    ax2.scatter(df_interpolated['x'], df_interpolated['y'], alpha=0.5)
    ax2.set_title('Corrected Data')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.invert_yaxis()
    
    # Plot ROIs on both axes
    for ax in [ax1, ax2]:
        if roi1:
            ax.add_patch(plt.Rectangle((roi1[0], roi1[2]), 
                                       roi1[1] - roi1[0], 
                                       roi1[3] - roi1[2], 
                                       fill=False, edgecolor='r'))
        if roi2:
            ax.add_patch(plt.Rectangle((roi2[0], roi2[2]), 
                                       roi2[1] - roi2[0], 
                                       roi2[3] - roi2[2], 
                                       fill=False, edgecolor='b'))
    
    plt.tight_layout()
    plt.show()
    
    return df_interpolated

# Example usage
if __name__ == "__main__":
    
    data_directory = Path("~/OneDrive").expanduser().resolve()
    output_files = get_output_h5_files(data_directory)

    if output_files:
        # Correct fish position for the first file
        print(output_files[15])
        # df_corrected = correct_fish_position_with_two_rois(output_files[15], method='linear', visualize=False)
        # df_corrected = correct_fish_position_with_roi(Path("/home/jmdelahanty/OneDrive/lgfl/rawdata/sub-0013/20240722_sub-0013_OnePortEtohExperiment/20240722_sub-0013_OnePortEtohExperiment_output_corrected.h5"), method='linear', visualize=False)
        manual_correction_with_roi(Path("/home/jmdelahanty/OneDrive/lgfl/rawdata/sub-0016/20240722_sub-0016_OnePortEtohExperiment/20240722_sub-0016_OnePortEtohExperiment_output_corrected.h5"))
        print("Correction completed.")
    else:
        print("No output files found.")
