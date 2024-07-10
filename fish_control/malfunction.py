import sys
import re
import time
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,
    QFormLayout, QPushButton, QDateEdit, QTextEdit, QFileDialog, QMessageBox,
    QComboBox, QMainWindow, QDialog, QListWidget, QFileDialog, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QDate, pyqtSlot, pyqtSignal
from datetime import datetime
import json
from pydantic import ValidationError, BaseModel
from equipment import Incubator
import inspect
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QTimer, QEventLoop
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QImage
import pyfirmata
from time import monotonic, sleep
import cv2
import queue
import os
import h5py
import numpy as np
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QPushButton, QDateEdit, QTextEdit, QFormLayout,
    QListWidget, QDialog, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from datetime import datetime
import inspect
from pathlib import Path
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QDate, pyqtSlot
import experiment as experiment_module
from threading import Event
import threading

# Pydantic models (from fish_updater.py)
class LightCycle(BaseModel):
    light_duration: str
    is_stable: bool
    dawn_dusk: str

class IncubatorProperties(BaseModel):
    temperature: float
    light_cycle: LightCycle
    room: str

class IncubatorModel(BaseModel):
    Left_Lab_Incubator: IncubatorProperties
    Right_Lab_Incubator: IncubatorProperties

class Breeding(BaseModel):
    parents: list[str]

class Metadata(BaseModel):
    class Config:
        arbitrary_types_allowed = True
    subject_id: str
    cross_id: str
    dish_id: str
    dof: str
    genotype: str
    sex: str
    species: str
    responsible: str
    breeding: Breeding
    enclosure: IncubatorProperties

CONFIG_FILE = "config.json"

class MetadataWindow(QWidget):
    def __init__(self, data_directory):
        super().__init__()
        self.data_directory = data_directory
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Metadata Input")
        layout = QFormLayout()

        # Subject Information
        self.subject_id_input = QLineEdit(self)
        self.cross_id_input = QLineEdit(self)
        self.dish_id = QLineEdit(self)
        self.dof_input = QDateEdit(self)
        self.dof_input.setDisplayFormat("yyyyMMdd")
        self.dof_input.setDate(datetime.today())
        self.dof_input.setCalendarPopup(True)
        self.genotype_input = QLineEdit(self)
        self.sex_input = QComboBox(self)
        self.sex_input.addItems(["M", "F", "U", "O"])
        self.species_input = QLineEdit(self)
        self.species_input.setText("Danio rerio")
        self.responsible = QLineEdit(self)
        self.enclosure_input = QComboBox(self)
        self.enclosure_input.addItems(["Left Lab Incubator", "Right Lab Incubator"])

        # Automatically set subject alias
        next_subject_number = self.get_next_subject_number(self.data_directory)
        self.subject_id_input.setText(f"sub-{next_subject_number:04d}")

        layout.addRow("Subject ID:", self.subject_id_input)
        layout.addRow("Cross ID:", self.cross_id_input)
        layout.addRow("Dish ID:", self.dish_id)
        layout.addRow("Date of Fertilization:", self.dof_input)
        layout.addRow("Genotype:", self.genotype_input)
        layout.addRow("Sex:", self.sex_input)
        layout.addRow("Species:", self.species_input)
        layout.addRow("Enclosure:", self.enclosure_input)
        layout.addRow("Responsible:", self.responsible)

        # Breeding Information
        self.parents_input = QLineEdit(self)
        layout.addRow("Parents (comma-separated):", self.parents_input)

        # Save Button
        save_button = QPushButton("Save Metadata", self)
        save_button.clicked.connect(self.save_metadata)
        layout.addRow(save_button)

        self.setLayout(layout)
    def get_next_subject_number(self, data_directory):
        data_directory = Path(data_directory)
        pattern = re.compile(r"sub-(\d{4})")
        max_number = 0

        for filename in data_directory.iterdir():
            match = pattern.search(str(filename))
            if match:
                number = int(match.group(1))
                if number > max_number:
                    max_number = number
        return max_number + 1

    def save_metadata(self):
        try:
            incubator = Incubator()
            selected_enclosure = self.enclosure_input.currentText()
            enclosure_properties = incubator.to_dict(selected_enclosure)

            metadata = Metadata(
                subject_id=self.subject_id_input.text(),
                cross_id=self.cross_id_input.text(),
                dish_id=self.dish_id.text(),
                dof=self.dof_input.text(),
                genotype=self.genotype_input.text(),
                sex=self.sex_input.currentText(),
                species=self.species_input.text(),
                responsible=self.responsible.text(),
                breeding=Breeding(
                    parents=self.parents_input.text().split(','),
                ),
                enclosure=IncubatorProperties(**enclosure_properties)
            )

            # Convert to dictionary and save to JSON
            metadata_dict = metadata.dict()
            # Define the path of the new directory
            new_sub_directory = Path(f'{self.data_directory}/{metadata_dict["subject_id"]}')

            # Create the directory
            new_sub_directory.mkdir(parents=True, exist_ok=True)
            with open(f"{new_sub_directory}/{metadata_dict['subject_id']}_metadata.json", "w") as file:
                json.dump(metadata_dict, file, indent=4)

            QMessageBox.information(self, "Success", "Metadata saved successfully.")

            # Reset the GUI state
            self.reset_gui_state()

        except ValidationError as e:
            QMessageBox.warning(self, "Validation Error", str(e))

    def reset_gui_state(self):
        self.cross_id_input.clear()
        self.dish_id.clear()
        self.dof_input.setDate(datetime.today())
        self.genotype_input.clear()
        self.sex_input.setCurrentIndex(0)
        self.species_input.setText("Danio rerio")
        self.responsible.clear()
        self.enclosure_input.setCurrentIndex(0)
        self.parents_input.clear()
        
        # Automatically set the next subject ID
        next_subject_number = self.get_next_subject_number(self.data_directory)
        self.subject_id_input.setText(f"sub-{next_subject_number:04d}")

class ExperimentSelectorWindow(QDialog):
    experiment_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Experiment Type")
        self.setGeometry(200, 200, 300, 200)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        select_button = QPushButton("Select")
        select_button.clicked.connect(self.on_select)
        layout.addWidget(select_button)

        self.setLayout(layout)

        self.populate_experiment_list()

    def populate_experiment_list(self):
        # Get all classes from the experiment module
        classes = inspect.getmembers(experiment_module, inspect.isclass)
        for name, cls in classes:
            # Add the class name to the list if it's defined in the experiment module
            if cls.__module__ == 'experiment':
                self.list_widget.addItem(name)

    def on_select(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            class_name = selected_item.text()
            selected_class = getattr(experiment_module, class_name)
            self.experiment_selected.emit(selected_class)
            self.accept()

class ExperimentConfigWindow(QWidget):
    def __init__(self, data_directory, experiment_class):
        super().__init__()
        self.data_directory = data_directory
        self.experiment = experiment_class()
        self.experiment_runner = None
        print(f"Created experiment with:")
        print(f"  Pre-period: {self.experiment.pre_period}s")
        print(f"  Post-period: {self.experiment.post_period}s")
        print(f"  Num stimuli: {self.experiment.num_stim}")
        print(f"  Num pulses: {self.experiment.num_pulses}")
        print(f"  IPI: {self.experiment.ipi}s")
        print(f"  ISI: {self.experiment.isi}s")
        print(f"  Recording duration: {self.experiment.recording_duration:.2f}s")
        self.initUI()
    def initUI(self):
        main_layout = QHBoxLayout()
        left_layout = QFormLayout()
        right_layout = QVBoxLayout()

        # Left side (experiment configuration)
        self.subject_combo = QComboBox(self)
        self.populate_subject_combo()
        left_layout.addRow("Subject:", self.subject_combo)

        self.input_fields = {}
        attributes_to_display = ['pre_period', 'post_period', 'ipi', 'isi', 'num_stim', 'num_pulses', 'recording_duration']
        for attr in attributes_to_display:
            value = getattr(self.experiment, attr)
            input_field = QLineEdit(self)
            input_field.setText(str(value))
            self.input_fields[attr] = input_field
            left_layout.addRow(f"{attr.replace('_', ' ').capitalize()}:", input_field)
            input_field.textChanged.connect(lambda _, a=attr: self.update_experiment(a))

        self.input_fields['recording_duration'].setReadOnly(True)

        self.experiment_name_input = QLineEdit(self)
        self.experiment_date_input = QDateEdit(self)
        self.experiment_date_input.setDate(QDate.currentDate())
        left_layout.addRow("Experiment Name:", self.experiment_name_input)
        left_layout.addRow("Experiment Date:", self.experiment_date_input)

        self.run_button = QPushButton("Run Experiment", self)
        self.run_button.clicked.connect(self.run_experiment)
        left_layout.addRow(self.run_button)

        self.status_label = QLabel("Ready", self)
        left_layout.addRow("Status:", self.status_label)

        # Right side (video display and log)
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Video feed will appear here")
        right_layout.addWidget(self.video_label)

        self.log_display = QTextEdit(self)
        self.log_display.setReadOnly(True)
        right_layout.addWidget(self.log_display)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)
        self.setLayout(main_layout)
        self.update_experiment_name()
    def populate_subject_combo(self):
        for d in Path(self.data_directory).iterdir():
            if d.is_dir():
                self.subject_combo.addItem(d.name)
        self.subject_combo.model().sort(0)

    def update_experiment(self, attr):
        try:
            value = self.input_fields[attr].text()
            if attr != 'recording_duration':
                if isinstance(getattr(self.experiment, attr), int):
                    value = int(value)
                elif isinstance(getattr(self.experiment, attr), float):
                    value = float(value)
                print(f"Updating {attr} to {value}")
                setattr(self.experiment, attr, value)
                self.experiment._update_recording_duration()  # Call this after updating any parameter
            
            # Update the recording_duration field in the GUI
            new_duration = self.experiment.recording_duration
            print(f"New recording duration: {new_duration:.2f}s")
            self.input_fields['recording_duration'].setText(str(new_duration))
        except ValueError:
            print(f"Invalid input for {attr}")
    
    def update_experiment_name(self):
        subject = self.subject_combo.currentText()
        experiment_type = type(self.experiment).__name__
        date = datetime.now().strftime("%Y%m%d")
        self.experiment_name_input.setText(f"{date}_{subject}_{experiment_type}")

    def run_experiment(self):
        try:
            print("About to verify duration:")
            print(f"  Current recording duration: {self.experiment.recording_duration:.2f}s")
            self.experiment.verify_duration()
            self.save_experiment_config()
            self.experiment_runner = ExperimentRunner(self.experiment)
            self.experiment_runner.experiment_thread.update_signal.connect(self.update_log)
            self.experiment_runner.experiment_thread.finished_signal.connect(self.experiment_finished)
            self.experiment_runner.experiment_thread.frame_ready.connect(self.update_video_feed)
            self.run_button.setEnabled(False)
            self.experiment_runner.start_experiment()
        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", str(e))
            return

    @pyqtSlot(str)
    def update_log(self, message):
        self.log_display.append(message)
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())
        print(f"Log: {message}")  # Print to console for debugging


    @pyqtSlot(QImage)
    def update_video_feed(self, image):
        pixmap = QPixmap.fromImage(image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # print("Video frame updated")  # Add this line for debugging

    @pyqtSlot(str)
    def update_status(self, message):
        self.status_label.setText(message)

    def experiment_finished(self):
        self.run_button.setEnabled(True)
        QMessageBox.information(self, "Experiment Completed", "The experiment has finished.")
        self.status_label.setText("Ready")

    def save_experiment_config(self):
        subject_id = self.subject_combo.currentText()
        experiment_name = self.experiment_name_input.text()
        experiment_date = self.experiment_date_input.date().toString(Qt.ISODate)
        
        config = {
            "subject_id": subject_id,
            "experiment_name": experiment_name,
            "experiment_date": experiment_date,
        }

        # Add all experiment parameters to the config
        for attr, field in self.input_fields.items():
            config[attr] = field.text()

        # Ensure the subject directory exists
        subject_dir = Path(self.data_directory) / subject_id
        subject_dir.mkdir(parents=True, exist_ok=True)

        # Save the config file
        config_path = subject_dir / f"{experiment_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        print(f"Experiment configuration saved to {config_path}")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.data_directory = self.load_config()
        if not self.data_directory:
            self.data_directory = self.prompt_for_data_directory()
            self.save_config(self.data_directory)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Fish Experiment Manager")
        self.setGeometry(100, 100, 300, 200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        
        metadata_button = QPushButton("Open Metadata Input", self)
        metadata_button.clicked.connect(self.open_metadata_window)
        layout.addWidget(metadata_button)

        experiment_button = QPushButton("Open Experiment Configuration", self)
        experiment_button.clicked.connect(self.open_experiment_selector)
        layout.addWidget(experiment_button)
        
        central_widget.setLayout(layout)

    def open_metadata_window(self):
        self.metadata_window = MetadataWindow(self.data_directory)
        self.metadata_window.show()

    def open_experiment_selector(self):
        selector = ExperimentSelectorWindow(self)
        selector.experiment_selected.connect(self.open_experiment_config)
        selector.exec_()

    def open_experiment_config(self, experiment_class):
        self.experiment_window = ExperimentConfigWindow(self.data_directory, experiment_class)
        self.experiment_window.show()

    def load_config(self):
        if Path(CONFIG_FILE).exists():
            with open(CONFIG_FILE, "r") as file:
                config = json.load(file)
            return config.get("data_directory")
        return None

    def save_config(self, data_directory):
        config = {"data_directory": data_directory}
        with open(CONFIG_FILE, "w") as file:
            json.dump(config, file, indent=4)

    def prompt_for_data_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if not dir_path:
            QMessageBox.warning(self, "No Directory Selected", "No directory selected. Exiting.")
            sys.exit(1)
        return dir_path

class ExperimentSelectorPopup(QDialog):
    def __init__(self, subjects_directory: Path):
        super().__init__()
        self.subjects_directory = Path(subjects_directory)
    experiment_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Experiment Type")
        self.setGeometry(200, 200, 300, 200)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        select_button = QPushButton("Select")
        select_button.clicked.connect(self.on_select)
        layout.addWidget(select_button)

        self.setLayout(layout)

        self.populate_experiment_list()

    def populate_experiment_list(self):
        # Get all classes from the experiment module
        classes = inspect.getmembers(experiment_module, inspect.isclass)
        for name, cls in classes:
            # Add the class name to the list if it's defined in the experiment module
            if cls.__module__ == 'experiment':
                self.list_widget.addItem(name)

    def on_select(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            class_name = selected_item.text()
            selected_class = getattr(experiment_module, class_name)
            self.experiment_selected.emit(selected_class)
            self.accept()

class StartEvent(QObject):
    started = pyqtSignal()

class ValveController(QObject):
    valve_operated = pyqtSignal(str, float)

    def __init__(self, board):
        super().__init__()
        self.valve1 = board.digital[4]
        self.valve1.mode = pyfirmata.OUTPUT
        self.valve1.write(0)

    def operate_valve(self, duration, stim_num, pulse_num):
        self.valve1.write(1)
        self.valve_operated.emit(f'valve_open_{stim_num}_{pulse_num}', monotonic())
        sleep(duration)
        self.valve1.write(0)
        self.valve_operated.emit(f'valve_close_{stim_num}_{pulse_num}', monotonic())

class CameraThread(QThread):
    frame_ready = pyqtSignal(QImage)
    log_signal = pyqtSignal(str)

    def __init__(self, duration, frame_rate, start_event, camera_ready_event, video_file):
        super().__init__()
        self.frame_rate = frame_rate
        self.duration = duration + (2/ self.frame_rate)
        self.start_event = start_event
        self.camera_ready_event = camera_ready_event
        self.video_file = video_file
        self.expected_frame_count = int(self.frame_rate * self.duration)
        self.frame_count = 0
        self.stop_flag = threading.Event()

    def run(self):
        self.log_signal.emit("CameraThread: Initializing camera")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.log_signal.emit("ERROR: Cannot open Camera")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.log_signal.emit(f"Frame size: {frame_width}x{frame_height}")

        fourcc = cv2.VideoWriter_fourcc(*'AVC1')
        out = cv2.VideoWriter(self.video_file, fourcc, self.frame_rate, (frame_width, frame_height))
        
        self.log_signal.emit(f"Checking if video file is opened: {out.isOpened()}")
        if not out.isOpened():
            self.log_signal.emit(f"ERROR: Could not open video file for writing: {self.video_file}")
            cap.release()
            return
        self.log_signal.emit("Video file opened for writing")
        self.log_signal.emit(f"Video will be saved as: {os.path.abspath(self.video_file)}")

        expected_frame_count = int(self.frame_rate * self.duration) + 1
        self.log_signal.emit(f"Expected frame count: {self.expected_frame_count}")

        self.log_signal.emit("Camera is ready, setting ready event")
        self.camera_ready_event.set()
        self.log_signal.emit("Waiting for start event...")

        self.start_event.wait()
        self.log_signal.emit("CameraThread: Starting video capture")
        start_capture_time = time.perf_counter()
        next_frame_time = start_capture_time
        frame_interval = 1.0 / self.frame_rate
        self.log_signal.emit(f"CameraThread: Capture start time: {start_capture_time}")
        
        def timing_thread():
            next_frame_time = start_capture_time
            while not self.stop_flag and self.frame_count < self.expected_frame_count:
                current_time = time.perf_counter()
                if current_time >= next_frame_time:
                    self.frame_count += 1
                    next_frame_time += frame_interval
                time.sleep(0.001)  # Small sleep to prevent busy-waiting
        
        timing_thread = threading.Thread(target=timing_thread)
        timing_thread.start()

        while self.frame_count < self.expected_frame_count and not self.stop_flag.is_set():
            ret, frame = cap.read()
            if ret:
                self.frame_count += 1
                out.write(frame)
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.frame_ready.emit(qt_image)
                next_frame_time += start_capture_time + (self.frame_count / self.frame_rate)
                if self.frame_count % self.frame_count == 0:
                    elapsed_time = time.perf_counter() - start_capture_time
                    self.log_signal.emit(f"Frames captured: {self.frame_count}, Time: {elapsed_time:.4f}s")
            else:
                self.log_signal.emit("ERROR: Failed to capture frame")
                break
        else:
            time.sleep(0.0005)

        self.stop_flag.set()
        timing_thread.join()
        cap.release()
        out.release()
        
        end_capture_time = time.perf_counter()
        total_duration = end_capture_time - start_capture_time

        self.log_signal.emit(f"Video recording completed. Total frames: {self.frame_count}")
        self.log_signal.emit(f"Total time elapsed: {total_duration:.2f}s")
        # Verify the frame count in the saved video
        saved_cap = cv2.VideoCapture(self.video_file)
        saved_frame_count = int(saved_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        saved_cap.release()
        self.log_signal.emit(f"Frames in saved video: {saved_frame_count}")

        if self.frame_count == saved_frame_count == self.expected_frame_count:
            self.log_signal.emit("Video recording completed as expected!")
        else:
            self.log_signal.emit("WARNING: Inconsistent frame count detected")
            self.log_signal.emit(f"Expected: {self.expected_frame_count}, Captured: {self.frame_count}, Saved: {saved_frame_count}")

        self.log_signal.emit(f"Video saved to: {os.path.abspath(self.video_file)}")

    def stop(self):
        self.log_signal.emit("Stopping camera thread...")
        self.stop_flag.set()
        self.log_signal.emit("Camera thread stopped!")


class H5WriterThread(QThread):
    def __init__(self, filename='valve_timestamps.h5', start_event=None, h5_ready_event=None):
        super().__init__()
        self.filename = filename
        self.queue = queue.Queue()
        self.stop_flag = threading.Event()
        self.processing_done = threading.Event()
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

    def get_experiment_start_time(self):
        return self.experiment_start_time

class ExperimentThread(QThread):
    update_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()
    frame_ready = pyqtSignal(QImage)

    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.stop_flag = Event()
        self.board = None
        self.start_event = Event()
        self.camera_thread = None
        self.camera_ready_event = Event()
        self.h5_writer = None
        self.h5_ready_event = Event()
        self.recording_duration = self.experiment.recording_duration
        self.current_stim = -1
        self.current_pulse = -1

    def run(self):
        try:
            # Generate filenames
            timestamp = datetime.now().strftime("%Y%m%d")
            video_filename = f"{timestamp}_experiment.mp4"
            h5_filename = f"{timestamp}_valve_timestamps.h5"
            timestamp = datetime.now().strftime("%Y%m%d")

            self.update_signal.emit("Setting up Arduino connection...")
            self.board = pyfirmata.Arduino('/dev/ttyACM1')
            self.update_signal.emit(f"Experiment ---- duration = {self.experiment.recording_duration}")
            
            valve_controller = ValveController(self.board)
            valve_controller.valve_operated.connect(self.handle_valve_event)
            self.h5_writer = H5WriterThread('valve_timestamps.h5', self.start_event, self.h5_ready_event)
            self.h5_writer.start()

            self.camera_thread = CameraThread(self.recording_duration, 30, self.start_event, self.camera_ready_event, video_filename)
            self.camera_thread.frame_ready.connect(self.frame_ready)
            self.camera_thread.log_signal.connect(self.update_signal)
            self.camera_thread.start()

            # Wait for both the camera and H5 writer to be ready before starting the experiment
            self.update_signal.emit("Waiting for camera and H5 writer to be ready...")
            self.camera_ready_event.wait()
            self.h5_ready_event.wait()
            self.update_signal.emit("Camera and H5 writer ready.")

            self.update_signal.emit("All threads started. Sending start event...")
            self.start_event.set()
            start_experiment_time = time.perf_counter()
            self.update_signal.emit(f"Experiment started at {start_experiment_time:.2f}s")

            # Pre-period
            self.update_signal.emit(f"Pre stimulus period ---- duration = {self.experiment.pre_period}")
            self.h5_writer.add_event_to_queue('prestimulus_start', -1, -1, time.monotonic())
            self.sleep(self.experiment.pre_period)
            self.h5_writer.add_event_to_queue('pre_stimulus_end', -1, -1, monotonic())
            
            for stim in range(self.experiment.num_stim):
                if self.stop_flag.is_set():
                    break
                self.current_stim = stim
                self.update_signal.emit(f"Stimulus {stim+1} start")
                for pulse in range(self.experiment.num_pulses):
                    if self.stop_flag.is_set():
                        break
                    self.current_pulse = pulse
                    self.update_signal.emit(f"Opening valve for stimulus {stim+1}, pulse {pulse+1}")
                    valve_controller.operate_valve(self.experiment.ipi, stim, pulse)
                    self.sleep(self.experiment.ipi)
                    self.update_signal.emit(f"Closing valve for stimulus {stim+1}, pulse {pulse+1}")
                self.sleep(self.experiment.isi)
                self.update_signal.emit(f"Stimulus {stim+1} end")

            # Post-period
            if not self.stop_flag.is_set():
                self.update_signal.emit(f"Post stimulus period ---- duration = {self.experiment.post_period}")
                self.h5_writer.add_event_to_queue('post_stimulus_start', -1, -1, monotonic())
                self.sleep(self.experiment.post_period)
                self.h5_writer.add_event_to_queue('post_stimulus_end', -1, -1, monotonic())

            self.h5_writer.add_event_to_queue('experiment_end', -1, -1, monotonic())
            self.update_signal.emit("Experiment completed successfully.")
        except Exception as e:
            self.update_signal.emit(f"An error occurred: {str(e)}")
        finally:
            self.cleanup()

    def handle_valve_event(self, event, timestamp):
        print(f"Event: {event}, Stim: {self.current_stim}, Pulse: {self.current_pulse}, Time: {timestamp}")
        if event.startswith('valve_open'):
            self.h5_writer.add_event_to_queue('valve_open', self.current_stim, self.current_pulse, timestamp)
        elif event.startswith('valve_close'):
            self.h5_writer.add_event_to_queue('valve_close', self.current_stim, self.current_pulse, timestamp)
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

class ExperimentRunner(QObject):
    def __init__(self, experiment):
        super().__init__()
        self.experiment = experiment
        self.experiment_thread = ExperimentThread(experiment)

    def start_experiment(self):
        self.experiment_thread.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())