from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer, QEventLoop
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage
import pyfirmata
from time import monotonic, sleep
import cv2
import queue
import h5py
import numpy as np

class StartEvent(QObject):
    started = pyqtSignal()

class ValveController(QObject):
    valve_operated = pyqtSignal(str, float)

    def __init__(self, board):
        super().__init__()
        self.valve1 = board.digital[4]
        self.valve1.mode = pyfirmata.OUTPUT
        self.valve1.write(0)
        self.timer = QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.close_valve)

    def operate_valve(self, duration):
        self.valve1.write(1)
        self.valve_operated.emit('open', monotonic())
        self.timer.start(int(duration * 1000))

    def close_valve(self):
        self.valve1.write(0)
        self.valve_operated.emit('close', monotonic())

class CameraThread(QThread):
    frame_ready = pyqtSignal(object)
    log_signal = pyqtSignal(str)

    def __init__(self, duration, frame_rate, start_event):
        super().__init__()
        self.duration = duration
        self.frame_rate = frame_rate
        self.stop_flag = False
        self.start_event = start_event
        self.video_file = 'experiment.avi'

    def run(self):
        self.log_signal.emit("CameraThread: Waiting for start signal")
        loop = QEventLoop()
        self.start_event.started.connect(loop.quit)
        loop.exec_()
        
        self.log_signal.emit("CameraThread: Starting video capture")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log_signal.emit("ERROR: Cannot open webcam")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log_signal.emit(f"Frame size: {frame_width}x{frame_height}")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.video_file, fourcc, self.frame_rate, (frame_width, frame_height))
        
        if not out.isOpened():
            self.log_signal.emit(f"ERROR: Could not open video file for writing: {self.video_file}")
            cap.release()
            return

        self.log_signal.emit(f"Video will be saved as: {os.path.abspath(self.video_file)}")

        start_time = monotonic()
        frame_count = 0

        while monotonic() - start_time < self.duration and not self.stop_flag:
            ret, frame = cap.read()
            if ret:
                out.write(frame)
                self.frame_ready.emit(frame)
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    self.log_signal.emit(f"Frames captured: {frame_count}")
            else:
                self.log_signal.emit("ERROR: Failed to capture frame")
                break

            # Small sleep to prevent CPU overuse
            self.msleep(10)

        cap.release()
        out.release()
        
        self.log_signal.emit(f"Video recording completed. Total frames: {frame_count}")
        self.log_signal.emit(f"Video saved to: {os.path.abspath(self.video_file)}")

    def stop(self):
        self.stop_flag = True

class H5WriterThread(QThread):
    def __init__(self, filename='valve_timestamps.h5', start_event=None):
        super().__init__()
        self.filename = filename
        self.queue = queue.Queue()
        self.stop_flag = False
        self.start_event = start_event

    def run(self):
        loop = QEventLoop()
        self.start_event.started.connect(loop.quit)
        loop.exec_()
        
        start_time = monotonic()
        self.add_event('start', start_time)
        
        with h5py.File(self.filename, 'w') as f:
            dt = np.dtype([('event', 'S10'), ('timestamp', 'f8')])
            dset = f.create_dataset('valve_events', (0,), maxshape=(None,), dtype=dt, chunks=True)
            f.attrs['intended_fps'] = 30
            
            buffer = []
            buffer_size = 10

            while not self.stop_flag or not self.queue.empty():
                try:
                    event, timestamp = self.queue.get(timeout=1)
                    buffer.append((event, timestamp))
                    
                    if len(buffer) >= buffer_size:
                        data = np.array(buffer, dtype=dt)
                        dset.resize((dset.shape[0] + len(buffer),))
                        dset[-len(buffer):] = data
                        buffer.clear()
                        f.flush()
                        for e, t in data:
                            print(f"Event written to H5 file: {e.decode('ascii')} at {t}")
                except queue.Empty:
                    continue
            
            # Write any remaining events in the buffer
            if buffer:
                data = np.array(buffer, dtype=dt)
                dset.resize((dset.shape[0] + len(buffer),))
                dset[-len(buffer):] = data
                f.flush()
                for e, t in data:
                    print(f"Event written to H5 file: {e.decode('ascii')} at {t}")
            
            self.add_event('end', monotonic())

    def add_event(self, event, timestamp):
        self.queue.put((event, timestamp))
        print(f"Event added to H5 queue: {event} at {timestamp}")

    def stop(self):
        self.stop_flag = True

class ExperimentThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    frame_ready = pyqtSignal(QImage)

    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.stop_flag = False
        self.start_event = StartEvent()
        self.h5_writer = None
        self.camera_thread = None
        self.board = None

    def run(self):
            try:
                self.update_signal.emit("Setting up Arduino connection...")
                board = pyfirmata.Arduino('/dev/ttyACM0')
                
                valve_controller = ValveController(board)
                camera_thread = CameraThread(self.experiment.recording_duration, 20, self.start_event)
                h5_writer = H5WriterThread('valve_timestamps.h5', self.start_event)

                valve_controller.valve_operated.connect(h5_writer.add_event)
                camera_thread.frame_ready.connect(self.frame_ready)

                camera_thread.start()
                h5_writer.start()

                self.update_signal.emit("Experiment ready. Starting...")
                self.start_event.started.emit()  # Signal all threads to start
                
                start_time = monotonic()

                # Pre-period
                self.update_signal.emit(f"Pre stimulus period ---- duration = {self.experiment.pre_period}")
                self.sleep(int(self.experiment.pre_period * 1000))

                # Experimental period
                exp_duration = self.experiment.num_stim * self.experiment.num_pulses * (self.experiment.ipi + self.experiment.isi)
                self.update_signal.emit(f"Experiment ---- duration = {exp_duration}")
                
                for _ in range(self.experiment.num_stim):
                    for _ in range(self.experiment.num_pulses):
                        self.update_signal.emit(f"Opening valve for {self.experiment.ipi} seconds")
                        valve_controller.operate_valve('open', self.experiment.ipi)
                        self.sleep(self.experiment.ipi)
                        self.update_signal.emit("Closing valve")
                        valve_controller.operate_valve('close', 0)
                    self.sleep(self.experiment.isi)

                # Post-period
                self.update_signal.emit(f"Post stimulus period ---- duration = {self.experiment.post_period}")
                valve_controller.operate_valve('close', self.experiment.post_period)
                self.sleep(int(self.experiment.post_period * 1000))

                self.update_signal.emit("Experiment completed successfully.")
            except Exception as e:
                self.update_signal.emit(f"An error occurred: {str(e)}")
            finally:
                camera_thread.stop()
                h5_writer.stop()
                camera_thread.wait()
                h5_writer.wait()
                if 'board' in locals():
                    board.exit()
                self.finished_signal.emit()

    def stop(self):
        self.stop_flag = True

    def sleep(self, seconds):
        start = monotonic()
        while monotonic() - start < seconds:
            QApplication.processEvents()  # Allow GUI updates
        actual_sleep = monotonic() - start
        self.update_signal.emit(f"Slept for {actual_sleep:.2f}s (intended: {seconds:.2f}s)")

class ExperimentRunner(QObject):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.experiment_thread = ExperimentThread(experiment)

    def start_experiment(self):
        self.experiment_thread.start()