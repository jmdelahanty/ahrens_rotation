# import cv2
# from vimba import *
# import numpy as np
# import time
# from typing import Optional

# class FrameHandler:

#     def __init__(self, width: int, height: int):
#         self.frame_count = 0
#         self.start_time = time.time()
#         self.output_file = 'output.mp4'
#         fourcc = cv2.VideoWriter_fourcc(*'avc1')
#         self.video_writer = cv2.VideoWriter(self.output_file, fourcc, 30, (width, height), isColor=True)
#         if not self.video_writer.isOpened():
#             raise IOError("Failed to open video writer")
#         print("Video writer opened successfully")
#         self.last_frame_time = time.time()

#     def __call__(self, cam: Camera, frame: Frame):
#         try:
#             current_time = time.time()
#             print(f"Frame received at {current_time - self.start_time:.2f}s (Interval: {current_time - self.last_frame_time:.4f}s)")
#             self.last_frame_time = current_time

#             image = frame.as_numpy_ndarray()
#             print(f"Processing frame {self.frame_count} - Shape: {image.shape}, Dtype: {image.dtype}, Min: {np.min(image)}, Max: {np.max(image)}")
#             image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

#             # Write the frame without checking the return value
#             self.video_writer.write(image)
#             print(f"Frame {self.frame_count} processed")

#             self.frame_count += 1

#             if current_time - self.start_time >= 10:
#                 print("10 seconds elapsed, stopping stream")
#                 cam.stop_streaming()
#         except Exception as e:
#             print(f"Error in frame handler: {str(e)}")
#             import traceback
#             print(traceback.format_exc())
#         finally:
#             try:
#                 cam.queue_frame(frame)
#             except Exception as e:
#                 print(f"Error queuing frame: {str(e)}")

#     def print_stats(self):
#         print(f"Recording complete. Total frames: {self.frame_count}")
#         self.video_writer.release()

#         # Check file size and validity
#         import os
#         if os.path.exists(self.output_file):
#             file_size = os.path.getsize(self.output_file)
#             print(f"Output file size: {file_size / (1024 * 1024):.2f} MB")
            
#             # Try to open the video file to verify it's valid
#             cap = cv2.VideoCapture(self.output_file)
#             if cap.isOpened():
#                 ret, frame = cap.read()
#                 # Get total number of frames
#                 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 print(f"Total frames in video: {total_frames}")
#                 if ret:
#                     print("Video file is valid and contains at least one frame")
#                 else:
#                     print("Video file opened but couldn't read a frame")
#                 cap.release()
#             else:
#                 print("Failed to open the video file, it might be corrupted")

#         else:
#             print("Output file does not exist")
# def print_camera_features(cam):
#     print("\nCamera Features:")
#     for feature in cam.get_all_features():
#         try:
#             value = feature.get()
#             print(f"{feature.get_name()}: {value}")
#         except (AttributeError, VimbaFeatureError):
#             print(f"{feature.get_name()}: Unable to read")

# def main():
#     print('Vimba simple example started.')

#     with Vimba.get_instance() as vimba:
#         print('Vimba instance created.')

#         cameras = vimba.get_all_cameras()
#         if not cameras:
#             print('No cameras found.')
#             return

#         with cameras[0] as cam:
#             print(f'Using camera {cam.get_name()}')

#             # Print all camera features before setting
#             print_camera_features(cam)

#             # Set up camera
#             cam.ExposureAuto.set('Off')
#             cam.ExposureMode.set('Timed')
#             cam.ExposureTime.set(33333)  # ~30 fps (1/30 second)
#             cam.AcquisitionFrameRate.set(30)
#             cam.AcquisitionMode.set('Continuous')
#             cam.set_pixel_format(PixelFormat.Mono8)
#             # cam.get_feature_by_name("StreamAnnounceBufferMinimum").set(5)
#             # cam.get_feature_by_name("StreamAnnouncedBufferCount").set(5)

#             # Print camera features after setting
#             print("\nCamera Features after setup:")
#             print_camera_features(cam)

#             # Get camera dimensions
#             width = cam.get_feature_by_name('Width').get()
#             height = cam.get_feature_by_name('Height').get()
#             print(f'Camera dimensions: {width}x{height}')

#             # Create frame handler
#             handler = FrameHandler(width, height)

#             # Start streaming
#             print("Starting camera stream")
#             cam.start_streaming(handler)
            
#             # Wait for streaming to complete or timeout
#             timeout = time.time() + 10  # 10 seconds timeout
#             while cam.is_streaming() and time.time() < timeout:
#                 if not cam.is_streaming():
#                     print("Camera stopped streaming unexpectedly")
#                     break
#                 print(f"Camera is streaming: {cam.is_streaming()}")
#                 time.sleep(0.1)
#                 if time.time() - handler.last_frame_time > 2:  # No frame for 2 seconds
#                     print("No frames received for 2 seconds, stopping stream")
#                     cam.stop_streaming()
#                     break

#             if cam.is_streaming():
#                 print("Timeout reached, stopping stream")
#                 cam.stop_streaming()

#             handler.print_stats()

# if __name__ == '__main__':
#     main()

import cv2

def print_opencv_backend_info():
    print("OpenCV version:", cv2.__version__)
    print("Available backends:")
    backends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]
    print(backends)
    
    print("Checking codec support:")
    codecs_to_check = ['avc1', 'mp4v', 'mjpg', 'xvid', 'h264']
    for codec in codecs_to_check:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        test_file = f'test_{codec}.mp4'
        test_writer = cv2.VideoWriter(test_file, fourcc, 30, (640, 480), isColor=True)
        if test_writer.isOpened():
            print(f"Codec {codec} is supported")
            test_writer.release()
            import os
            os.remove(test_file)
        else:
            print(f"Codec {codec} is not supported")


import cv2
import numpy as np

def test_video_writer():
    output_file = 'test_output.mp4'
    fps = 30
    frame_size = (640, 480)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Try 'mp4v' first
    out = cv2.VideoWriter(output_file, fourcc, fps, frame_size, isColor=True)

    if not out.isOpened():
        print(f"Failed to open VideoWriter with mp4v, trying avc1")
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_file, fourcc, fps, frame_size, isColor=True)
        
    if not out.isOpened():
        print("Failed to open VideoWriter with both mp4v and avc1")
        return

    for i in range(100):  # Generate 100 frames
        frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        out.write(frame)
        print(f"Wrote frame {i+1}")

    out.release()
    print(f"Video saved to {output_file}")

    # Verify the file was created and has content
    import os
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file)
        print(f"File size: {file_size} bytes")
    else:
        print("File was not created")

test_video_writer()

# print_opencv_backend_info()