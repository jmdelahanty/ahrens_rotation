# Built in Imports for the GUI
# sys for sys calls like exiting the program
import sys
# pathlib for easy path manipulation
from pathlib import Path
# json for saving/loading configuration
import json
# inspect for dynamic dropdown population
import inspect

# PyQt5 imports for the GUI
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog,
    QMessageBox, QMainWindow, QDialog, QListWidget,
)
from PyQt5.QtCore import pyqtSignal, QObject

# Import metadata gui and experiment guis
from metadata_gui import MetadataWindow
from experiment_gui import ExperimentSelectorWindow, ExperimentConfigWindow

# Vimba is a camera control system. This is a no longer supported version of the
# software because this vimba is quite old. Unfortunately, the only camera immediately
# available is the old camera I have from the older rig.
# The procedure for installing this old Vimba version is to download the old software from
# here: https://www.alliedvision.com/en/products/vimba-sdk/
# Once downloaded, install the drivers with the driver installation helper,
# Vimba Viewer, and update any firmware that the software deems necessary.
# From there, you should be able to grab frames just fine!
# The python bindings require you to update an environment variable
# This is the "VIMBA_HOME" Variable. You do this by:
# Open a Command Prompt as an Admin. You can sign in locally with:
# .\local_admin_acct_name ; passwd: password
# In the command prompt, type:
# setx VIMBA_HOME "C:\Program Files\Allied Vision\Vimba_6.0" /M
# The setx Creates an environment variable, /M sets this machine wide (hence the admin rights)
# You can check this works in python through:
# import os
# print(os.environ.get('VIMBA_HOME'))
# It should show the updated path to this file
# Look at the Vimba Python Manual in the docs folder for installing
# and do so in your mamba environment

### CUSTOM CLASSES ###
# Import the experiment module
import experiments

# Set static config file location
CONFIG_FILE = "config.json"

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        config = self.load_config()
        if not config:
            self.raw_data_directory, self.remote_data_directory = self.prompt_for_data_directories()
            self.save_config(self.raw_data_directory, self.remote_data_directory)
        else:
            self.remote_data_directory = config.get("remote_data_directory")
            self.raw_data_directory = config.get("raw_data_directory")
            print(f"Raw Data Directory: {self.raw_data_directory}")
            print(f"Remote Data Directory: {self.remote_data_directory}")
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
        self.metadata_window = MetadataWindow(self.remote_data_directory)
        self.metadata_window.show()

    def open_experiment_selector(self):
        selector = ExperimentSelectorWindow(self)
        selector.experiment_selected.connect(self.open_experiment_config)
        selector.exec_()

    def open_experiment_config(self, experiment_class):
        self.experiment_window = ExperimentConfigWindow(self.remote_data_directory, self.raw_data_directory, experiment_class)
        self.experiment_window.show()

    def load_config(self):
        # Determine if the config file exists
        if Path(CONFIG_FILE).exists():
            # Load the config file
            with open(CONFIG_FILE, "r") as file:
                config = json.load(file)
                if config.get("raw_data_directory") == None:
                    print("Config not found! Select directories...")
                    return None
                else:
                    return config
        else:
            raise FileNotFoundError("Config file not found! Create one...")

    def save_config(self, local_data_path, remote_data_path):
        config = {
            "raw_data_directory": local_data_path,
            "remote_data_directory": remote_data_path
            }
        with open(CONFIG_FILE, "w") as file:
            json.dump(config, file, indent=4)

    def prompt_for_data_directories(self):
        local_data_path = QFileDialog.getExistingDirectory(self, "Select Raw Data Directory")
        if not local_data_path:
            QMessageBox.warning(self, "No Directory Selected", "No directory selected. Exiting.")
            sys.exit(1)
        remote_data_path = QFileDialog.getExistingDirectory(self, "Select Remote Data Directory")
        return local_data_path, remote_data_path

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
        classes = inspect.getmembers(experiments, inspect.isclass)
        for name, cls in classes:
            # Add the class name to the list if it's defined in the experiment module
            if cls.__module__ == 'experiment':
                self.list_widget.addItem(name)

    def on_select(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            class_name = selected_item.text()
            selected_class = getattr(experiments, class_name)
            self.experiment_selected.emit(selected_class)
            self.accept()

class StartEvent(QObject):
    started = pyqtSignal()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())