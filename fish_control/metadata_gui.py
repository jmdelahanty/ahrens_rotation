# Metadata GUI/Fish Updater
# PyQt5 imports for the GUI
from PyQt5.QtWidgets import (
    QWidget, QLineEdit, QFormLayout, QPushButton, QDateEdit, QMessageBox,
    QComboBox
)
from datetime import datetime
import json
from pathlib import Path
import re
from pydantic import ValidationError
# Custom classes for equipment (ie. Incubator)
from equipment import Incubator
# Custom classes for metadata (ie. Metadata, IncubatorProperties, Breeding)
from metadata import Metadata, IncubatorProperties, Breeding
from pathlib import Path

class MetadataWindow(QWidget):
    def __init__(self, remote_dir=None):
        super().__init__()
        self.remote_dir = Path(remote_dir)
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
        next_subject_number = self.get_next_subject_number(self.remote_dir)
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
            metadata_dict = metadata.model_dump()
            # Define the path of the new directory
            new_sub_directory = Path(f'{self.remote_dir}/{metadata_dict["subject_id"]}')

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
        next_subject_number = self.get_next_subject_number(self.remote_dir)
        self.subject_id_input.setText(f"sub-{next_subject_number:04d}")