import sys
import os
import re
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,
    QFormLayout, QPushButton, QDateEdit, QTextEdit, QFileDialog, QMessageBox,
    QComboBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from datetime import datetime
import json
from pydantic import ValidationError, BaseModel
from equipment import Incubator

class Breeding(BaseModel):
    parents: list[str]

class Metadata(BaseModel):
    subject_id: str
    cage_id: str
    cage_alias: str
    dof: str
    genotype: str
    sex: str
    species: str
    breeding: Breeding

CONFIG_FILE = "config.json"

class MetadataGUI(QWidget):
    def __init__(self):
        super().__init__()
        
        self.data_directory = self.load_config()
        if not self.data_directory:
            self.data_directory = self.prompt_for_data_directory()
            self.save_config(self.data_directory)

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Lets Get Fish Lit")

        # Main layout
        main_layout = QHBoxLayout()

        # Form layout for better organization
        form_layout = QFormLayout()

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
        self.strain_input = QLineEdit(self)
        self.responsible = QLineEdit(self)
        self.enclosure_input = QComboBox(self)
        self.enclosure_input.addItems(["Left Lab Incubator", "Right Lab Incubator"])
        # Automatically set subject alias
        next_subject_number = self.get_next_subject_number(self.data_directory)
        self.subject_id_input.setText(f"sub-{next_subject_number:03d}")
        form_layout.addRow("Subject ID:", self.subject_id_input)
        form_layout.addRow("Dish ID:", self.dish_id)
        form_layout.addRow("Date of Fertilization:", self.dof_input)
        form_layout.addRow("Genotype:", self.genotype_input)
        form_layout.addRow("Sex:", self.sex_input)
        form_layout.addRow("Species:", self.species_input)

        # Breeding Information
        self.parents_input = QLineEdit(self)
        self.litter_input = QLineEdit(self)

        form_layout.addRow("Parents (comma-separated):", self.parents_input)
        form_layout.addRow("Litter ID:", self.litter_input)

        # Save Button
        save_button = QPushButton("Save Metadata", self)
        save_button.clicked.connect(self.save_metadata)
        form_layout.addRow(save_button)

        # Add form layout to the main layout
        vbox = QVBoxLayout()
        vbox.addLayout(form_layout)
        main_layout.addLayout(vbox)

        # Load and display the image
        image_label = QLabel(self)
        pixmap = QPixmap("/Users/jmdelahanty/gitrepos/ahrens_rotation/fish_control/zebrafish_cartoon.png")
        image_label.setPixmap(pixmap)
        main_layout.addWidget(image_label)

        self.setLayout(main_layout)

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
            metadata = Metadata(
                subject_id=self.subject_id_input.text(),
                cross_id=self.cross_id_input.text(),
                dof=self.dof_input.text(),
                genotype=self.genotype_input.text(),
                sex=self.sex_input.text(),
                species=self.species_input.text(),
                breeding=Breeding(
                    parents=self.parents_input.text().split(','),
                    litter=self.litter_input.text()
                )
            )

            # Convert to dictionary and save to JSON
            metadata_dict = metadata.model_dump()
            with open(f"{metadata_dict['subject_id']}_metadata.json", "w") as file:
                json.dump(metadata_dict, file, indent=4)

            print("Metadata saved successfully.")

        except ValidationError as e:
            print("Validation error:", e)
        
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MetadataGUI()
    ex.show()
    sys.exit(app.exec())