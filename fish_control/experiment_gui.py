# Experiment GUI

import inspect
import json
from datetime import datetime
from pathlib import Path
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QDate
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,
    QFormLayout, QPushButton, QDateEdit, QTextEdit, QMessageBox,
    QComboBox, QDialog, QListWidget, 
)

from experiment_objects import ExperimentRunner
import experiment

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
        classes = inspect.getmembers(experiment, inspect.isclass)
        for name, cls in classes:
            # Add the class name to the list if it's defined in the experiment module
            if cls.__module__ == 'experiment':
                self.list_widget.addItem(name)

    def on_select(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            class_name = selected_item.text()
            selected_class = getattr(experiment, class_name)
            self.experiment_selected.emit(selected_class)
            self.accept()

class ExperimentConfigWindow(QWidget):
    def __init__(self, remote_dir=None, raw_data_directory= None, experiment_class=None):
        super().__init__()
        self.data_directory = Path(raw_data_directory)
        self.remote_dir = Path(remote_dir)
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
        attributes_to_display = [
            'pre_period', 'post_period', 'ipi', 'isi', 'num_stim', 'num_pulses', "left_syringe", "etoh_concentration",
            'right_syringe', 'recording_duration'
            ]
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
        for d in Path(self.remote_dir).iterdir():
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
            self.check_vimba_import()
            print("About to verify duration:")
            print(f"Current recording duration: {self.experiment.recording_duration:.2f}s")
            self.experiment.verify_duration()
            self.save_experiment_config()
            experiment_dir = self.data_directory / self.subject_combo.currentText() / self.experiment_name_input.text()
            experiment_dir.mkdir(parents=True, exist_ok=True)
            self.experiment_runner = ExperimentRunner(self.experiment, experiment_dir)
            self.experiment_runner.experiment_thread.update_signal.connect(self.update_log)
            self.experiment_runner.experiment_thread.finished_signal.connect(self.experiment_finished)
            self.experiment_runner.experiment_thread.frame_ready.connect(self.update_video_feed)

            self.log_display.append("Signal connections established")
            
            self.run_button.setEnabled(False)
            self.experiment_runner.start_experiment()
        except ValueError as e:
            QMessageBox.warning(self, "Validation Error", str(e))
            return
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")
            return

    def check_vimba_import(self):
        try:
            from vimba import Vimba
            self.log_display.append("Vimba import successful")
        except ImportError as e:
            self.log_display.append(f"Error importing Vimba: {str(e)}")
        except Exception as e:
            self.log_display.append(f"Unexpected error when importing Vimba: {str(e)}")

    @pyqtSlot(str)
    def update_log(self, message):
        self.log_display.append(message)
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())
        print(f"Log: {message}")  # Print to console for debugging

    @pyqtSlot(QImage)
    def update_video_feed(self, image):

        try:

            pixmap = QPixmap.fromImage(image)
            
            if image.isNull():
                self.log_display.append("Received null QImage")
                return
            
            pixmap = QPixmap.fromImage(image)
            if pixmap.isNull():
                self.log_display.append("Failed to create QPixmap from QImage")
                return
            self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.log_display.append(f"Error updating video feed, Line 224: {str(e)}")

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