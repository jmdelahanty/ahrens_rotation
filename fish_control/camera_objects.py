from PyQt5.QtCore import QObject, pyqtSignal, QThread
from PyQt5.QtGui import QImage
from threading import Event
from time import perf_counter, sleep
from typing import Tuple
import os
import sys
import cv2
import numpy as np
from vimba import Vimba, Camera, Frame, VimbaCameraError, PixelFormat


class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)

    def __init__(self, duration, frame_rate, start_event, camera_ready_event, video_file):
        super().__init__()
        self.frame_rate = frame_rate
        self.duration = duration + (2 / self.frame_rate)
        self.start_event = start_event
        self.camera_ready_event = camera_ready_event
        self.video_file = video_file
        self.video_writer = None
        self.camera_width = 0
        self.camera_height = 0
        self.expected_frame_count = int(self.frame_rate * self.duration)
        self.frame_count = 0
        self.stop_flag = Event()
        self.start_time = None
        self.last_frame_time = 0

    def run(self):
        try:
            self.log_signal.emit("CameraThread: Starting run method")
            self.log_signal.emit(f"CameraThread: Initializing Vimba Camera and Video Writer")
            self.log_signal.emit(f"Expected frame count: {self.expected_frame_count}")

            with Vimba.get_instance() as vimba:
                self.log_signal.emit("CameraThread: Vimba instance created")

                cameras = vimba.get_all_cameras()
                self.log_signal.emit(f"CameraThread: Found {len(cameras)} cameras")

                if not cameras:
                    raise ValueError("No cameras found")

                with cameras[0] as cam:
                    self.log_signal.emit(f"CameraThread: Using camera {cam.get_name()}")

                    self.log_signal.emit("CameraThread: Setting camera features")
                    cam.ExposureAuto.set('Off')
                    cam.ExposureMode.set('Timed')
                    cam.ExposureTime.set(1000000 / self.frame_rate)
                    cam.AcquisitionFrameRate.set(self.frame_rate)
                    cam.AcquisitionMode.set('Continuous')
                    
                    actual_frame_rate = cam.AcquisitionFrameRate.get()
                    self.log_signal.emit(f"Actual frame rate set: {actual_frame_rate}")
                    
                    self.log_signal.emit("CameraThread: Setting pixel format")
                    cam.set_pixel_format(PixelFormat.Mono8)

                    self.log_signal.emit("CameraThread: Getting camera dimensions")
                    self.width = cam.get_feature_by_name('Width').get()
                    self.height = cam.get_feature_by_name('Height').get()
                    self.log_signal.emit(f"Camera dimensions: Width:{self.width} x Height:{self.height}")

                    self.log_signal.emit("CameraThread: Initializing video writer")
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    self.video_writer = cv2.VideoWriter(str(self.video_file), fourcc, self.frame_rate, (self.width, self.height), isColor=True)
                    if not self.video_writer.isOpened():
                        raise ValueError("Could not open video writer")
                    
                    self.log_signal.emit("Video file opened for writing")

                    self.log_signal.emit("CameraThread: Starting camera streaming")
                    cam.start_streaming(self.frame_handler)
                    self.log_signal.emit("CameraThread: Camera streaming started")

                    self.camera_ready_event.set()
                    self.log_signal.emit("CameraThread: Camera ready event set")
                    
                    self.log_signal.emit("CameraThread: Waiting for start event")
                    self.start_event.wait()
                    self.log_signal.emit(f"CameraThread: Start event received at {perf_counter():.6f}")

                    self.start_time = perf_counter()
                    self.log_signal.emit(f"CameraThread: Acquisition started at {self.start_time:.6f}")
    
                    # Filesize debugging
                    # last_size_check = self.start_time
                    # last_file_size = 0

                    while self.frame_count < self.expected_frame_count and not self.stop_flag.is_set():
                        
                        # Video file size debugging
                        # current_time = perf_counter()
                        
                        # # Check file size every 5 seconds
                        # if current_time - last_size_check >= 5:
                        #     if self.video_file.exists():
                        #         current_file_size = os.path.getsize(self.video_file)
                        #         size_increase = current_file_size - last_file_size
                        #         self.log_signal.emit(f"Video file size: {current_file_size / 1024 / 1024:.2f} MB "
                        #                             f"(increased by {size_increase / 1024 / 1024:.2f} MB)")
                        #         last_file_size = current_file_size
                        #     else:
                        #         self.log_signal.emit("Warning: Video file does not exist")
                        #     last_size_check = current_time

                        sleep(0.001)  # Small sleep to prevent busy waiting

                    self.log_signal.emit("CameraThread: Stopping camera streaming")
                    cam.stop_streaming()

                    end_time = perf_counter()
                    total_duration = end_time - self.start_time
                    
                    self.log_signal.emit(f"Image acquisition completed. Total frames: {self.frame_count}")
                    self.log_signal.emit(f"Total time elapsed: {total_duration:.2f}s")
                    self.log_signal.emit(f"Actual frame rate: {self.frame_count / total_duration:.2f} fps")
                    
                    if self.video_writer is not None:
                        self.video_writer.release()
                        self.log_signal.emit(f"Video saved to: {self.video_file}")

        except VimbaCameraError as e:
            self.log_signal.emit(f"VimbaCameraError: {str(e)}")
        except Exception as e:
            self.log_signal.emit(f"An unexpected error occurred in CameraThread: {str(e)}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
        finally:
            if self.video_writer is not None:
                self.video_writer.release()
            self.log_signal.emit("CameraThread: Exiting run method")

    def frame_handler(self, cam: Camera, frame: Frame):
        if not self.start_event.is_set():
            try:
                cam.queue_frame(frame)
            except Exception as e:
                self.log_signal.emit(f"Error queuing frame: {str(e)}")
            return
        
        if self.video_writer is None:
            try:
                self.log_signal.emit("Video writer not ready, queuing frame")
                cam.queue_frame(frame)
            except Exception as e:
                self.log_signal.emit(f"Error queuing frame: {str(e)}")
            return

        try:
            current_time = perf_counter()
            self.last_frame_time = current_time

            image = frame.as_numpy_ndarray()
            im = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            self.video_writer.write(im)

            if self.frame_count <= 5:
                self.log_signal.emit(f"Processing frame {self.frame_count} - Shape: {image.shape}, Dtype: {image.dtype}, Min: {np.min(image)}, Max: {np.max(image)}")

            h, w, ch = im.shape
            bytes_per_line = w * ch
            q_image = QImage(im.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.frame_ready.emit(q_image)

            self.frame_count += 1

            if current_time - self.start_time >= self.duration:
                self.log_signal.emit(f"{self.duration} seconds elapsed, stopping stream")
                self.stop_flag.set()

        except Exception as e:
            import traceback
            error_info = traceback.extract_tb(sys.exc_info()[2])
            line_number = error_info[-1].lineno
            self.log_signal.emit(f"Error in frame handler at line {line_number}: {str(e)}")
            self.log_signal.emit(traceback.format_exc())
        finally:
            try:
                cam.queue_frame(frame)
            except Exception as e:
                self.log_signal.emit(f"Error queuing frame: {str(e)}")

    def stop(self):
        self.log_signal.emit("Stopping camera thread...")
        self.stop_flag.set()
        self.log_signal.emit("Camera thread stopped!")

class DebugVideoWriter:
    def __init__(self, filename: str, fourcc: int, fps: float, frameSize: Tuple[int, int], isColor: bool = True):
        self.log_signal = print  # Replace this with your log_signal method
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor)
        self.frame_count = 0
        self.filename = filename
        self.fourcc = fourcc
        self.fps = fps
        self.frameSize = frameSize
        self.isColor = isColor

    def write(self, frame: np.ndarray) -> bool:
        self.frame_count += 1
        self.log_signal(f"Attempting to write frame {self.frame_count}")
        self.log_signal(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        self.log_signal(f"Frame min: {np.min(frame)}, max: {np.max(frame)}")
        
        try:
            result = self.writer.write(frame)
            if result:
                self.log_signal(f"Frame {self.frame_count} written successfully")
            else:
                self.log_signal(f"Failed to write frame {self.frame_count}")
            return result
        except Exception as e:
            self.log_signal(f"Error writing frame {self.frame_count}: {str(e)}")
            return False

    def release(self):
        self.writer.release()
        self.log_signal("VideoWriter released")

    def isOpened(self):
        return self.writer.isOpened()

    def __getattr__(self, name):
        return getattr(self.writer, name)