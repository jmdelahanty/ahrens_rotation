from PyQt5.QtCore import QObject, pyqtSignal, QThread, QElapsedTimer
from PyQt5.QtGui import QImage
from threading import Event
from time import perf_counter, sleep
import os
import sys
import cv2
import numpy as np
from vimba import Vimba, Camera, Frame, VimbaCameraError, PixelFormat
from abc import ABC, abstractmethod, ABCMeta
from queue import Queue, Empty
import skvideo.io

class DiskWriterThread(QThread):
    log_signal = pyqtSignal(str)

    def __init__(self, video_file, frame_rate, frame_size, disk_writer_ready_event, isColor=True, start_event=None):
        super().__init__()
        self.video_file = video_file
        self.frame_rate = frame_rate
        self.frame_size = frame_size
        self.isColor = isColor
        self.start_event = start_event
        self.disk_writer_ready_event = disk_writer_ready_event
        self.queue = Queue()
        self.running = True
        self.video_writer = None
        self.frames_written = 0
        self.frames_received = 0

    def run(self):
        try:
            self.log_signal.emit("DiskWriterThread: Starting run method")
            
            # Set up skvideo.io writer
            outputdict = {
                '-vcodec': 'libx264',
                '-crf': '23', 
                '-preset': 'ultrafast',
                '-pix_fmt': 'yuv420p',
                '-r': str(self.frame_rate)
            }
            
            self.video_writer = skvideo.io.FFmpegWriter(
                self.video_file,
                inputdict={'-r': str(self.frame_rate)},
                outputdict=outputdict
            )
            
            self.log_signal.emit("DiskWriterThread: skvideo.io writer initialized successfully")
            self.disk_writer_ready_event.set()
            self.log_signal.emit("DiskWriterThread: Ready event set, waiting for start event")
            
            if self.start_event:
                self.log_signal.emit("DiskWriterThread: Waiting for start event...")
                self.start_event.wait()
                self.log_signal.emit("DiskWriterThread: Start event received. Beginning video writing")
            else:
                self.log_signal.emit("DiskWriterThread: No start event provided, beginning video writing immediately")

            self.log_signal.emit("DiskWriterThread: Entering main processing loop")
            while self.running or not self.queue.empty():
                try:
                    frame = self.queue.get(timeout=1)
                    if frame is not None:
                        # If the frame is grayscale, convert to RGB
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        elif frame.shape[2] == 3 and not self.isColor:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        self.frames_received += 1
                        try:
                            self.video_writer.writeFrame(frame)
                            self.frames_written += 1
                        except Exception as e:
                            print(f"Error writing frame: {str(e)}")
                    
                    self.queue.task_done()
                except Empty:
                    if not self.running and self.queue.empty():
                        self.log_signal.emit("DiskWriterThread: Queue empty and not running, exiting loop")
                        break
                    self.log_signal.emit("DiskWriterThread: Queue is empty, waiting for more frames")
                    continue
        except Exception as e:
            self.log_signal.emit(f"DiskWriterThread: Error during execution - {str(e)}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
        finally:
            self.finalize()

    def add_frame(self, frame):
        self.queue.put(frame)
        if self.queue.qsize() % 10 == 0:
            self.log_signal.emit(f"DiskWriterThread: Queue size is now {self.queue.qsize()}")

    def stop(self):
        self.log_signal.emit("DiskWriterThread: Stop requested")
        self.running = False
        self.wait()

    def finalize(self):
        self.log_signal.emit("DiskWriterThread: Finalizing")
        frames_remaining = self.queue.qsize()
        self.log_signal.emit(f"DiskWriterThread: {frames_remaining} frames remaining in queue")
        while not self.queue.empty():
            try:
                frame = self.queue.get(timeout=1)
                if frame is not None and self.video_writer is not None:
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    elif frame.shape[2] == 3 and not self.isColor:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.video_writer.writeFrame(frame)
                    self.frames_written += 1
                self.queue.task_done()
            except Empty:
                break
        if self.video_writer is not None:
            self.video_writer.close()
            self.log_signal.emit(f"DiskWriterThread: Video writer closed. Total frames written: {self.frames_written}")
            self.log_signal.emit(f"DiskWriterThread: Total frames received: {self.frames_received}")
        else:
            self.log_signal.emit("DiskWriterThread: Video writer was not initialized")
        
        # Check if the video file was created and has content
        if os.path.exists(self.video_file):
            file_size = os.path.getsize(self.video_file)
            self.log_signal.emit(f"DiskWriterThread: Video file size: {file_size} bytes")
        else:
            self.log_signal.emit("DiskWriterThread: Video file was not created")


class ThreadABCMeta(type(QThread), ABCMeta):
    pass

class BaseCameraThread(QThread, ABC, metaclass=ThreadABCMeta):
    frame_ready = pyqtSignal(np.ndarray)
    log_signal = pyqtSignal(str)

    def __init__(self, duration, frame_rate, start_event, camera_ready_event, video_file):
        super().__init__()
        self.duration = duration
        self.frame_rate = frame_rate
        self.start_event = start_event
        self.camera_ready_event = camera_ready_event
        self.video_file = video_file
        self.video_writer = None
        self.expected_frame_count = int(self.frame_rate * self.duration)
        self.frame_count = 0
        self.stop_flag = Event()

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def stop(self):
        pass

class VimbaCameraThread(BaseCameraThread):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vimba import Vimba, Camera, Frame, VimbaCameraError, PixelFormat
        self.Vimba = Vimba
        self.PixelFormat = PixelFormat


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
                        self.validate_frame_count()

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

            self.frame_ready.emit(im)

            self.frame_count += 1

            if current_time - self.start_time >= self.duration:
                self.log_signal.emit(f"{self.duration} seconds elapsed, stopping stream")
                self.stop()

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

    def validate_frame_count(self):
        # Check written video file for frame count
        cap = cv2.VideoCapture(str(self.video_file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # Raise a warning if the frame count is not as expected
        if frame_count != self.expected_frame_count:
            self.log_signal.emit(f"Warning: Frame count mismatch. Expected: {self.expected_frame_count}, Actual: {frame_count}")

    def stop(self):
        self.log_signal.emit("Stopping camera thread...")
        self.stop_flag.set()
        self.log_signal.emit("Camera thread stopped!")

class BaslerCameraThread(BaseCameraThread):
    def __init__(self, duration, frame_rate, start_event, camera_ready_event, video_file):
        super().__init__(duration, frame_rate, start_event, camera_ready_event, video_file)
        import pypylon.pylon as pylon
        self.pylon = pylon
        self.disk_writer = None
        self.disk_writer_ready_event = Event()
        self.start_event = start_event
        self.frame_interval = 1.0 / self.frame_rate
        self.frame_buffer = Queue(maxsize=10)

    def run(self):
        try:
            self.log_signal.emit("BaslerCameraThread: Starting run method")
            self.log_signal.emit(f"Expected frame count: {self.expected_frame_count}")
            converter = self.pylon.ImageFormatConverter()

            # converting to opencv bgr format
            converter.OutputPixelFormat = self.pylon.PixelType_Mono8
            converter.OutputBitAlignment = self.pylon.OutputBitAlignment_MsbAligned

            tlf = self.pylon.TlFactory.GetInstance()
            cam = self.pylon.InstantCamera(tlf.CreateFirstDevice())
            self.log_signal.emit(f"BaslerCameraThread: Opening Camera {cam.GetDeviceInfo().GetModelName()}")
            cam.Open()
            self.log_signal.emit("BaslerCameraThread: Camera opened successfully")

            # Configure camera settings
            cam.Width.Value = cam.Width.Max
            cam.Height.Value = cam.Height.Max
            cam.CenterX.Value = True
            cam.CenterY.Value = True
            cam.ExposureTime.Value = 30000
            cam.AcquisitionFrameRateEnable.Value = True
            cam.AcquisitionFrameRate.Value = 30

            # Set up disk writer thread
            self.log_signal.emit("BaslerCameraThread: Initializing disk writer thread")
            self.disk_writer = DiskWriterThread(
                self.video_file,
                self.frame_rate,
                (
                    int(cam.Width.Value),
                    int(cam.Height.Value)
                ),
                self.disk_writer_ready_event,
                isColor=True,
                start_event=self.start_event
            )

            self.log_signal.emit("BaslerCameraThread: DiskWriterThread initialized")
            self.disk_writer.log_signal.connect(self.log_signal)
            self.log_signal.emit("BaslerCameraThread: Starting disk writer thread")
            self.disk_writer.start()
            self.log_signal.emit("BaslerCameraThread: DiskWriterThread started")

            self.disk_writer_ready_event.wait()
            self.log_signal.emit("BaslerCameraThread: DiskWriterThread ready")
            self.camera_ready_event.set()
            self.log_signal.emit("BaslerCameraThread: Camera ready event set")
            cam.StartGrabbing(self.pylon.GrabStrategy_LatestImageOnly)
            self.log_signal.emit("BaslerCameraThread: Camera grabbing started")
            self.start_event.wait()

            next_frame_time = 0

            while cam.IsGrabbing() and self.frame_count < self.expected_frame_count and not self.stop_flag.is_set():

                grab_result = cam.RetrieveResult(5000, self.pylon.TimeoutHandling_ThrowException)
                
                if grab_result.GrabSucceeded():
                    image = converter.Convert(grab_result)
                    img = image.GetArray()

                    self.frame_buffer.put(img, block=False)
                    self.process_frame_buffer(img)

                    self.frame_count += 1
                    next_frame_time += self.frame_interval
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    self.frame_ready.emit(img_bgr)
                else:
                    print("Error: GRAB FAILED")
                    sleep(0.001)
                grab_result.Release()
            
            self.log_signal.emit(f"BaslerCameraThread: Capture complete. Total frames: {self.frame_count}")

        except Exception as e:
            self.log_signal.emit(f"An unexpected error occurred in BaslerCameraThread: {str(e)}")
            import traceback
            self.log_signal.emit(traceback.format_exc())
            print("ERROR WITH CAMERA THREAD")
            print(traceback.format_exc())
        finally:
            cam.StopGrabbing()
            cam.Close()
            self.log_signal.emit("BaslerCameraThread: Camera closed successfully")

            if self.disk_writer:
                self.log_signal.emit("BaslerCameraThread: Stopping disk writer thread")
                self.disk_writer.stop()
                self.log_signal.emit("BaslerCameraThread: DiskWriterThread stopped")
            
            self.validate_frame_count()

    def stop(self):
        self.log_signal.emit("BaslerCameraThread: Stop requested")
        self.stop_flag.set()
        if self.disk_writer:
            self.disk_writer.stop()
        self.log_signal.emit("BaslerCameraThread: Stopped")

    def add_frame(self, frame):
        if self.disk_writer:
            self.disk_writer.add_frame(frame)
        else:
            self.log_signal.emit("BaslerCameraThread: DiskWriterThread not initialized, frame dropped")

    def process_frame_buffer(self, img):
        while not self.frame_buffer.empty():
            img = self.frame_buffer.get()
            if self.disk_writer:
                self.disk_writer.add_frame(img)

    def validate_frame_count(self):
        # Check written video file for frame count
        cap = cv2.VideoCapture(str(self.video_file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # Raise a warning if the frame count is not as expected
        if frame_count != self.expected_frame_count:
            self.log_signal.emit(f"Warning: Frame count mismatch. Expected: {self.expected_frame_count}, Actual: {frame_count}")
