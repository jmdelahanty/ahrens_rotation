# Experiment Running Threads and Objects
from PyQt5.QtCore import QObject, pyqtSignal, QThread
from time import monotonic, sleep, perf_counter
import pyfirmata
from datetime import datetime
from threading import Event
import queue
import h5py
import numpy as np
from PyQt5.QtGui import QImage
import subprocess as sp
from typing import List
from pathlib import Path
import sys
import json
from camera_objects import VimbaCameraThread, BaslerCameraThread, DiskWriterThread
# from experiment_objects import OnePortEtohExperiment, EtOHBathExperiment

STATIC_FIRMATA_PATH = Path("C:/Users/delahantyj/Documents/Arduino/libraries/Firmata/examples/StandardFirmata/StandardFirmata.ino")

class CleanupThread(QThread):
    update_signal = pyqtSignal(str)
    
    def __init__(self, data_directory):
        super().__init__()
        self.data_directory = Path(data_directory)
        self.valve_controller = None
        self.stop_flag = False

    def run(self):
        try:
            self.update_signal.emit("Initializing cleanup process...")
            self.setup_arduino()
            self.valve_controller.start_cleanup()
            self.update_signal.emit("Cleanup started. Valve opened.")
            
            while not self.stop_flag:
                self.msleep(100)  # Check stop flag every 100ms
            
            self.valve_controller.stop_cleanup()
            self.update_signal.emit("Cleanup stopped. Valve closed.")
        except Exception as e:
            self.update_signal.emit(f"Error during cleanup: {str(e)}")
        finally:
            if self.valve_controller:
                try:
                    self.valve_controller.cleanup_finished.disconnect()
                    self.valve_controller.disconnect()  # Disconnect from Arduino
                    self.update_signal.emit("Disconnected from Arduino.")
                except Exception as e:
                    self.update_signal.emit(f"Error disconnecting from Arduino: {str(e)}")
            self.update_signal.emit("Cleanup process finished.")

    def setup_arduino(self):
        self.update_signal.emit("Setting up Arduino for cleanup...")
        # You might need to adjust this part based on your Arduino setup
        board = pyfirmata.Arduino('COM3')  # Replace 'COM3' with your Arduino port
        self.valve_controller = ValveController(board)
        self.valve_controller.cleanup_finished.connect(self.quit)

    def stop_cleanup(self):
        self.stop_flag = True


class ValveController(QObject):
    valve_operated = pyqtSignal(str, float)
    cleanup_finished = pyqtSignal()

    def __init__(self, board):
        super().__init__()
        self.board = board
        self.valve1 = board.digital[4]
        self.valve1.mode = pyfirmata.OUTPUT
        self.valve1.write(0)
        self.cleanup_mode = False

    def operate_valve(self, duration, stim_num, pulse_num):
        self.valve1.write(1)
        self.valve_operated.emit(f'valve_open_{stim_num}_{pulse_num}', monotonic())
        sleep(duration)
        self.valve1.write(0)
        self.valve_operated.emit(f'valve_close_{stim_num}_{pulse_num}', monotonic())

    def start_cleanup(self):
        self.cleanup_mode = True
        self.valve1.write(1)

    def stop_cleanup(self):
        if self.cleanup_mode:
            self.cleanup_mode = False
            self.valve1.write(0)
            self.cleanup_finished.emit()  # Emit the cleanup_finished signal
    def disconnect(self):
        if self.board:
            self.board.exit()
            self.board = None

class ExperimentThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    frame_ready = pyqtSignal(QImage)
    request_confirmation = pyqtSignal(str, str) # Request confirmation from the user before starting the experiment (title, message)

    def __init__(self, experiment, experiment_dir):
        super().__init__()
        self.experiment = experiment
        self.experiment_dir = experiment_dir
        self.stop_flag = Event()
        self.board = None
        self.start_event = Event()
        self.camera_thread = None
        self.camera_ready_event = Event()
        # self.disk_writer_thread = None
        # self.disk_writer_ready_event = Event()
        self.h5_writer = None
        self.h5_ready_event = Event()
        self.recording_duration = self.experiment.recording_duration
        self.current_stim = -1
        self.current_pulse = -1
        self.user_confirmed = Event()
        self.video_filename = None
        self.h5_filename = None
        self.all_components_ready = Event()

    def setup_arduino(self):
        self.update_signal.emit("Setting up Arduino...")
        arduino_boards = Arduino.list_boards()
        if not arduino_boards:
            raise ValueError("No Arduino boards found. Please ensure the board is connected and the Arduino CLI is installed correctly.")
        if len(arduino_boards) > 1:
            self.update_signal.emit("Multiple Arduino boards detected. Using the first one.")
        
        selected_board = arduino_boards[0]
        self.update_signal.emit(f"Selected board: {selected_board[0]} on {selected_board[2]}")
        arduino = Arduino(sketch_path=STATIC_FIRMATA_PATH, idx=0)
        self.update_signal.emit("Compiling Firmata Sketch...")
        arduino.compile_sketch()
        self.update_signal.emit("Uploading Firmata Sketch...")
        # arduino.upload_sketch()
        self.update_signal.emit("Standard Firmata setup complete!")
        self.board = pyfirmata.Arduino(arduino.board_com)
        self.update_signal.emit(f"Arduino board initialized on {arduino.board_com}")

    def setup_camera(self):
        self.update_signal.emit("Setting up camera...")
        camera_class = self.get_camera_class()
        self.camera_thread = camera_class(
            self.recording_duration,
            30,  # Frame rate
            self.start_event,
            self.camera_ready_event,
            self.video_filename
        )
        self.camera_thread.frame_ready.connect(self.frame_ready)
        self.camera_thread.log_signal.connect(self.update_signal)
        self.camera_thread.start()
        self.update_signal.emit("Camera thread started.")

    def get_camera_class(self):
        if hasattr(self.experiment, 'camera_class'):
            return self.experiment.camera_class
        else:
            raise ValueError(f"Unknown experiment type: {type(self.experiment).__name__}")

    def setup_h5_writer(self):
        self.update_signal.emit("Setting up H5 writer...")
        self.h5_writer = H5WriterThread(self.h5_filename, self.start_event, self.h5_ready_event)
        self.h5_writer.start()

    def setup_components(self):
        if getattr(self.experiment, 'requires_camera', False):
            self.setup_camera()
        if getattr(self.experiment, 'requires_arduino', False):
            self.setup_arduino()
        if getattr(self.experiment, 'requires_h5_logging', False):
            self.setup_h5_writer()

    def wait_for_components(self):
        self.update_signal.emit("Waiting for all components to be ready...")
        events_to_wait = []
        
        if self.camera_thread:
            events_to_wait.append(self.camera_ready_event)
        if self.h5_writer:
            events_to_wait.append(self.h5_ready_event)
        # if self.disk_writer_thread:
        #     events_to_wait.append(self.disk_writer_ready_event)
        
        for event in events_to_wait:
            event.wait()
            self.update_signal.emit(f"Component ready: {event}")
        
        self.all_components_ready.set()
        self.update_signal.emit("All components are ready.")

    def cleanup(self):
        self.stop_flag.set()
        if self.h5_writer:
            self.update_signal.emit("Stopping H5 writer thread...")
            self.h5_writer.stop()
            self.h5_writer.wait_until_done()
        if self.board:
            self.update_signal.emit("Closing Arduino connection...")
            self.board.exit()
        self.finished_signal.emit()

    def stop(self):
        self.stop_flag.set()

    def sleep(self, duration):
        start_time = monotonic()
        while monotonic() - start_time < duration and not self.stop_flag.is_set():
            sleep(0.1)  # Sleep in short intervals to allow for stopping

    def run(self):
        try:
            # Generate filenames
            timestamp = datetime.now().strftime("%Y%m%d")
            self.video_filename = self.experiment_dir / f"{timestamp}_experiment.mp4"
            self.h5_filename = self.experiment_dir / f"{timestamp}_valve_timestamps.h5"

            self.update_signal.emit("Setting up experiment components...")

            # Setup the components based on experiment requirements
            self.setup_components()

            # Wait for all components to be ready
            self.wait_for_components()

            self.update_signal.emit("All components ready. Requesting user confirmation...")

            # Request user confirmation
            self.request_confirmation.emit("Start Experiment", "All components ready. Click OK to start!")

            # Wait for user confirmation
            self.user_confirmed.wait()

            self.update_signal.emit("All components ready. Starting experiment...")
            start_experiment_time = perf_counter()
            self.update_signal.emit(f"Setting start event at {start_experiment_time:.2f}s")
            self.start_event.set()
            self.update_signal.emit(f"Experiment started at {start_experiment_time:.2f}s")

            # Run the experiment
            self.experiment.run(self)

            self.update_signal.emit("Experiment completed successfully.")
        except Exception as e:
            self.update_signal.emit(f"An error occurred: {str(e)}")
            import traceback
            self.update_signal.emit(traceback.format_exc())
        finally:
            self.cleanup()

    def sleep(self, seconds):
        start = monotonic()
        while monotonic() - start < seconds and not self.stop_flag.is_set():
            QThread.msleep(10)  # Sleep for short intervals to allow stopping
        actual_sleep = monotonic() - start
        self.update_signal.emit(f"Slept for {actual_sleep:.2f}s (intended: {seconds:.2f}s)")

    def cleanup(self):
        self.stop_flag.set()

        if self.h5_writer:
            self.update_signal.emit("Stopping H5 writer thread...")
            self.h5_writer.stop()
            self.h5_writer.wait_until_done()
            self.update_signal.emit("H5 writer thread stopped.")

        if self.board:
            self.update_signal.emit("Closing Arduino connection...")
            self.board.exit()
            self.update_signal.emit("Arduino connection closed.")
        self.finished_signal.emit()

    def stop(self):
        self.stop_flag.set()

    def user_confirmation_received(self):
        self.user_confirmed.set()
        self.update_signal.emit("User confirmed experiment start.")

class ExperimentRunner(QObject):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    frame_ready = pyqtSignal(QImage)

    def __init__(self, experiment, experiment_dir):
        super().__init__()
        self.experiment = experiment
        self.experiment_dir = Path(experiment_dir)
        self.experiment_thread = ExperimentThread(experiment, self.experiment_dir)

    def start_experiment(self):
        try:
            self.experiment_thread.start()
            # self.experiment_thread.wait()
            # self.log_signal.emit("Experiment completed successfully!")
        except Exception as e:
            self.update_signal.emit(f"An error occurred: {str(e)}")
            import traceback
            self.update_signal.emit(traceback.format_exc())


class H5WriterThread(QThread):
    def __init__(self, filename, start_event=None, h5_ready_event=None):
        super().__init__()
        self.filename = filename
        self.queue = queue.Queue()
        self.stop_flag = Event()
        self.processing_done = Event()
        self.start_event = start_event
        self.h5_ready_event = h5_ready_event
        self.experiment_start_time = None
        self.event_count = 0

    def run(self):
        try:
            with h5py.File(self.filename, 'w') as f:
                print(f"H5WriterThread: File {self.filename} created")
                
                dt = np.dtype([
                    ('event', 'S20'),
                    ('stim_number', 'i4'),
                    ('pulse_number', 'i4'),
                    ('timestamp', 'f8')
                ])
                dset = f.create_dataset('valve_events', (0,), maxshape=(None,), dtype=dt, chunks=True)
                f.attrs['intended_fps'] = 30

                self.h5_ready_event.set()
                print("H5WriterThread: Ready event set, waiting for start event...")
                self.start_event.wait()
                
                self.experiment_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                f.attrs['experiment_start_time'] = self.experiment_start_time
                print(f"H5WriterThread: Experiment started at {self.experiment_start_time}")

                while not self.stop_flag.is_set() or not self.queue.empty():
                    try:
                        event_data = self.queue.get(timeout=1)
                        self.event_count += 1
                        print(f"H5WriterThread: Writing event to file: {event_data}")
                        
                        dset.resize((dset.shape[0] + 1,))
                        dset[-1] = event_data
                        f.flush()
                        
                        print(f"H5WriterThread: Event written. Total events: {dset.shape[0]}")
                    except queue.Empty:
                        if self.stop_flag.is_set():
                            print("H5WriterThread: Stop flag set and queue empty, exiting...")
                            break
                        print("H5WriterThread: Queue empty, waiting for more events...")
                        continue

                print(f"H5WriterThread: Total events processed: {self.event_count}")
                print(f"H5WriterThread: Final dataset size: {dset.shape[0]}")

        except Exception as e:
            print(f"H5WriterThread: An error occurred while writing to the HDF5 file: {str(e)}")
        finally:
            self.processing_done.set()
            print("H5WriterThread: HDF5 file closed")

    def add_event_to_queue(self, event, stim_number, pulse_number, timestamp):
        self.queue.put((event, stim_number, pulse_number, timestamp))
        print(f"H5WriterThread: Event added to queue: {event} (Stim: {stim_number}, Pulse: {pulse_number}) at {timestamp}")

    def stop(self):
        self.stop_flag.set()
        print("H5WriterThread: Stop flag set, thread will stop after processing remaining events")

    def wait_until_done(self):
        self.processing_done.wait()
        print("H5WriterThread: Processing done")

class Arduino:
    """
    Generic Arduino class for interacting with arbitrary Arduino boards.

    Class for discovering boards available on the system with ability
    to select arbitrary Arduino boards discovered on the machine. Compiles
    and uploads sketches to the board before the experiment starts.
    """

    def __init__(self, sketch_path:Path=None, idx:int=0):
        self.sketch_path = sketch_path

        properties = self.list_boards()

        if not properties:
            raise ValueError("No Arduino boards found. Please ensure the board is connected and the CLI is installed.")

        if idx > len(properties):
            raise ValueError(f"Requested board with index {idx}, but {len(properties)} boards found")
        
        self.board_name = properties[idx][0]
        self.fqbn = properties[idx][1]
        self.board_com = properties[idx][2]

        print(f"Selected board: {self.board_name} on {self.board_com}")


    @classmethod
    def list_boards(cls) -> List[tuple]:
        """
        Query CLI for finding available Arduinos on the machine.
        """

        print("Determining Board Properties...")

        # Query the CLI with a subprocess
        com_list = sp.run(
            [
                "arduino-cli",
                "board",
                "list",
                "--format",
                "json"
            ],
            capture_output=True
        ).stdout.decode("utf-8")

        # Load json formatted output for parsing
        decoded_com_list = json.loads(com_list)

        # For each address found in the com_list (the CLI)
        # will report all noted COM addresses even if its
        # not sure what it is), find its name, fully-qualified
        # board name (fqbn) and the COM ports associated with it
        boards = []
        if 'detected_ports' in decoded_com_list:
            for port_info in decoded_com_list['detected_ports']:
                if 'matching_boards' in port_info:
                    for board in port_info['matching_boards']:
                        boards.append((
                            board['name'],
                            board['fqbn'],
                            port_info['port']['address']
                        ))
        if not boards:
            print("No Arduino boards found.")
        else:
            print(f"Found {len(boards)} Arduino board(s).")

        return boards

    def compile_sketch(self):
        """
        Use the CLI to compile the project's Arduino sketch
        """

        print("Compiling Sketch...")

        compile_sketch = sp.run(
            [
                "arduino-cli",
                "compile",
                "--fqbn",
                self.fqbn,
                str(self.sketch_path),
                "-v",
                "--format",
                "json"
            ],
            capture_output=True
        )

        compile_output = json.loads(compile_sketch.stdout.decode())

        if compile_output["success"]:
            print("Sketch compiled successfully!")
        
        # TODO: Things like this should be logged...
        else:
            print("COMPILATION ERROR!!! Error in Arduino Script!")
            print(compile_output)
            print(compile_sketch.stderr.decode())   
            sys.exit()

    def upload_sketch(self):
        """
        Use the CLI to upload the sketch to the Arduino.
        """

        print("Uploading Sketch...")

        upload_sketch = sp.run(
            [
                "arduino-cli",
                "upload",
                "-p",
                self.board_com,
                "--fqbn",
                self.fqbn,
                str(self.sketch_path),
                "--format",
                "json",
                "--verbose"
            ],
            capture_output=True
        )

        # TODO: Another thing that should be logged
        if upload_sketch.returncode:
            print("UPLOAD FAILURE! Serial Monitor Open Elsewhere?")
            print(upload_sketch.stdout.decode())
            print(upload_sketch.stderr.decode())
            sys.exit()
        else:
            print("Upload successful!")