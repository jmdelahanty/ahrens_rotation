import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QPushButton, QDateEdit, QTextEdit, QFormLayout,
    QListWidget, QDialog, QComboBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from datetime import datetime
import experiment
import inspect
from pathlib import Path
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QDate, pyqtSlot
from experiment_thread import ValveController

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
    def __init__(self, subjects_directory: Path, parent=None):
        super().__init__()
        self.subjects_directory = Path(subjects_directory)
        self.experiment = None
        self.subject_paths = {}
        self.selected_experiment_name = ""
        self.experiment_name_input = None
        self.select_experiment()

    def select_experiment(self):
        selector = ExperimentSelectorPopup(self)
        selector.experiment_selected.connect(self.on_experiment_selected)
        selector.exec_()

    def on_experiment_selected(self, experiment_class):
        valve_controller = ValveController()  # Initialize the ValveController
        self.experiment = experiment_class(valve_controller=valve_controller)  # Pass the valve_controller instance
        self.selected_experiment_name = experiment_class.__name__
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Experiment Configuration")
        self.setGeometry(100, 100, 400, 400)

        # Main layout
        layout = QVBoxLayout()

        # Form layout for better organization
        form_layout = QFormLayout()

        # Subject selection
        self.subject_combo = QComboBox(self)
        self.populate_subject_combo()
        self.subject_combo.currentTextChanged.connect(self.update_experiment_name)
        form_layout.addRow("Subject:", self.subject_combo)


        # Dictionary to hold attribute names and their corresponding input fields
        self.input_fields = {}

        # List of attributes to be displayed in the GUI
        attributes_to_display = [
            'pre_period', 'post_period', 'ipi', 'isi', 'num_stim', 'num_pulses', 'recording_duration'
        ]

        # Dynamically create input fields based on the attributes of the Experiment instance
        for attr in attributes_to_display:
            value = getattr(self.experiment, attr)
            input_field = QLineEdit(self)
            input_field.setText(str(value))
            self.input_fields[attr] = input_field
            form_layout.addRow(f"{attr.replace('_', ' ').capitalize()}:", input_field)
            # Connect textChanged signal to update_experiment method
            input_field.textChanged.connect(lambda _, a=attr: self.update_experiment(a))

        # Make recording_duration read-only
        self.input_fields['recording_duration'].setReadOnly(True)

        # Special handling for non-standard fields like experiment name, date, and notes
        self.experiment_name_input = QLineEdit(self)
        self.experiment_date_input = QDateEdit(self)
        self.experiment_date_input.setDisplayFormat("yyyyMMdd")
        self.experiment_date_input.setDate(datetime.today())
        self.experiment_date_input.setCalendarPopup(True)
        self.experiment_notes_input = QTextEdit(self)

        form_layout.addRow("Experiment Name:", self.experiment_name_input)
        form_layout.addRow("Experiment Date:", self.experiment_date_input)
        form_layout.addRow("Experiment Notes:", self.experiment_notes_input)

        # Save button
        save_button = QPushButton("Save Experiment", self)
        save_button.clicked.connect(self.save_experiment)
        form_layout.addRow(save_button)

        layout.addLayout(form_layout)
        self.setLayout(layout)

    def update_experiment(self, attr):
        value = self.input_fields[attr].text()
        try:
            setattr(self.experiment, attr, value)
            
            # Update the recording_duration field
            self.input_fields['recording_duration'].setText(str(self.experiment.recording_duration))
            
            # Debug: Print all attributes
            self.print_all_attributes()
        except ValueError as e:
            print(f"Invalid input for {attr}: {value}")
            print(f"Error: {str(e)}")

    def populate_subject_combo(self):
        self.subject_paths.clear()  # Clear existing entries
        for d in self.subjects_directory.iterdir():
            if d.is_dir():
                subject_name = d.name
                self.subject_paths[subject_name] = d
                self.subject_combo.addItem(subject_name)
        self.subject_combo.model().sort(0)

    def update_experiment_name(self):
        if self.experiment_name_input is None:
            return  # Exit if experiment_name_input is not yet created
        today = datetime.now().strftime("%Y%m%d")
        subject = self.subject_combo.currentText()
        experiment_type = self.selected_experiment_name
        default_name = f"{today}_{subject}_{experiment_type}"
        self.experiment_name_input.setText(default_name)

    def print_all_attributes(self):
        print("\nCurrent Experiment Attributes:")
        for attr in ['pre_period', 'post_period', 'ipi', 'isi', 'num_stim', 'num_pulses', 'recording_duration']:
            value = getattr(self.experiment, attr)
            print(f"{attr}: {value} (GUI: {self.input_fields[attr].text()})")
        print()  # Add a blank line for readability

    def save_experiment(self):
        experiment_name = self.experiment_name_input.text()
        experiment_date = self.experiment_date_input.text()
        experiment_notes = self.experiment_notes_input.toPlainText()

        print("\nSaving Experiment:")
        print(f"Experiment Name: {experiment_name}")
        print(f"Experiment Date: {experiment_date}")
        print(f"Experiment Notes: {experiment_notes}")
        self.print_all_attributes()

        print("Experiment configuration saved successfully.")
        
        # Close the window after saving
        self.close()