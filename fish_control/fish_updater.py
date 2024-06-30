import sys
import os
import re
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QVBoxLayout, QHBoxLayout,
    QFormLayout, QPushButton, QDateEdit, QTextEdit, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from datetime import datetime
import json
from pydantic import ValidationError, BaseModel

class Breeding(BaseModel):
    parents: list[str]
    litter: str

class Metadata(BaseModel):
    subject_id: str
    subject_alias: str
    cage_id: str
    cage_alias: str
    dob: str
    description: str
    genotype: str
    sex: str
    species: str
    strain: str
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
        self.setWindowTitle("Metadata Creator")

        # Main layout
        main_layout = QHBoxLayout()

        # Form layout for better organization
        form_layout = QFormLayout()

        # Subject Information
        self.subject_id_input = QLineEdit(self)
        self.cross_id_input = QLineEdit(self)
        self.cage_alias_input = QLineEdit(self)
        self.dof_input = QDateEdit(self)
        self.dof_input.setDisplayFormat("yyyyMMdd")
        self.dof_input.setCalendarPopup(True)
        self.genotype_input = QLineEdit(self)
        self.sex_input = QLineEdit(self)
        self.species_input = QLineEdit(self)
        self.strain_input = QLineEdit(self)

        # Automatically set subject alias
        next_subject_number = self.get_next_subject_number(self.data_directory)
        self.subject_id_input.setText(f"sub-{next_subject_number:03d}")
        form_layout.addRow("Subject ID:", self.subject_id_input)
        form_layout.addRow("Cage Alias:", self.cage_alias_input)
        form_layout.addRow("Date of Fertilization:", self.dof_input)
        form_layout.addRow("Genotype:", self.genotype_input)
        form_layout.addRow("Sex:", self.sex_input)
        form_layout.addRow("Species:", self.species_input)
        form_layout.addRow("Strain:", self.strain_input)

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
        pixmap = QPixmap("/home/jmdelahanty/gitrepos/ahrens_rotation/zebrafish_cartoon.png")
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
                subject_alias=self.subject_alias_input.text(),
                cage_id=self.cage_id_input.text(),
                cage_alias=self.cage_alias_input.text(),
                dob=self.dob_input.date().toString("yyyyMMdd"),
                description=self.description_input.toPlainText(),
                genotype=self.genotype_input.text(),
                sex=self.sex_input.text(),
                species=self.species_input.text(),
                strain=self.strain_input.text(),
                breeding=Breeding(
                    parents=self.parents_input.text().split(','),
                    litter=self.litter_input.text()
                )
            )

            # Convert to dictionary and save to JSON
            metadata_dict = metadata.dict()
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



# subject_id: "A000"
# subject_alias: "best-mouse"
# ear_tag:
#   l_ear: true
#   r_ear: false
# ear_notch:
#   l_ear: false
#   r_ear: false
# # Cage IDs MUST be the barcode for the cage
# cage_id: 0123456
# # Cage Alias is whatever the experimenter likes to refer the cage as
# cage_alias: "cage-alias"
# # Date of birth in YYYMMDD format
# dob: "20220205"
# # Descriptions
# description: "Mouse Description"
# genotype: "HET DAT Cre"
# sex: "M"
# species: "Mus Musculus"
# strain: "C57BL/6J"
# # Surgeries are organized by date
# surgery:
#   "20210924":
#     surgeon: "Jeremy Delahanty"
#     start_time: "10:00"
#     end_time: "11:00"
#     stereotax: "Delta"
#     anesthetic: "Isoflurane"
#     skull_position:
#       bregma: -0.02
#       lambda: -0.02
#       level_left: -0.01
#       level_right: -0.01
#     headplate:
#       num_screws: 1
#       skull_hashing: true
#     grin_implant:
#       target: "mPFC"
#       hemisphere: "R"
#       type: "Proview"
#       dims: [1, 4]
#       supplier: "Inscopix"
#       ap: 1.90
#       ml: 0.45
#       dv: -2.18
#       angle: null
#       notes: "None"
#     brain_injections:
#       gcamp:
#         target: "mPFC"
#         hemisphere: "R"
#         virus: "AAV8"
#         promoter: "promoter"
#         opsin: null
#         fluorophore: "jGCaMP7f"
#         fluorophore_excitation_lambda: 482.5
#         fluorophore_emission_lambda: 513.0
#         supplier: "The Salk Institute for Biological Studies"
#         description: "Calcium Sensitive Green Fluorescent Indicator"
#         lot: null
#         ap: 1.90
#         ml: 0.45
#         dv: -2.20
#         angle: null
#         # Volume in nL
#         volume: 300
#         # Rate in nL/min
#         rate: 100.0
#         # Bevel Direction: Away from you is 0 degrees
#         bevel: 90
#       chr:
#         target: "VTA"
#         hemisphere: "R"
#         virus: "AAV8"
#         promoter: "hSyn"
#         cre: "FLEX"
#         flp: null
#         opsin: "chrimsonR"
#         opsin_excitation_lambda: 100.0
#         fluorophore: "tdTomato"
#         fluorophore_excitation_lambda: 482.5
#         fluorophore_emission_lambda: 581.0
#         supplier: "Supplier"
#         description: "description"
#         lot: null
#         ap: -3.30
#         ml: 0.35
#         dv: -4.05
#         angle: null
#         # Volume in nL
#         volume: 700.0
#         # Rate in nL/min
#         rate: 100.0
#         # Bevel Direction
#         bevel: 180
#     analgesics:
#       buprinex:
#         route: "IP"
#         location: "abdomen"
#         # Dosages are in mg/kg
#         dose: 1.0
#         # Volumes here in mL
#         volume: 2.0
#       lidocaine:
#         route: "SQ"
#         location: "scalp"
#         dose: null
#         volume: 0.5
#       loxicam:
#         route: "IP"
#         location: "abdomen"
#         # Dosages are in mg/kg
#         dose: 5.0
#         volume: 0.5
#     fluids:
#       ringers:
#         route: "IP"
#         location: "abdomen"
#         percent: 20.0
#         volume: 1.0

# status:
#   alive: true
#   time_of_death: null
#   death_notes: null