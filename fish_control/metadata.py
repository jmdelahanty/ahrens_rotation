# Module for pydantic validation of metadata
from pydantic import BaseModel

# Pydantic models (from fish_updater.py)
class LightCycle(BaseModel):
    light_duration: str
    is_stable: bool
    dawn_dusk: str

class IncubatorProperties(BaseModel):
    temperature: float
    light_cycle: LightCycle
    room: str

class Dish(BaseModel):
    dish_number: int
    subdish_number: int

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
    dish_id: Dish
    dof: str
    genotype: str
    sex: str
    species: str
    responsible: str
    breeding: Breeding
    enclosure: IncubatorProperties