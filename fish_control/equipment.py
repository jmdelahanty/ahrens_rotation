class Incubator:
    def __init__(self):
        self.incubator_properties = {
            "Left Lab Incubator": {
                "temperature": 28.5,
                "light_cycle": {
                    "light_duration": "8am - 10pm",
                    "is_stable": True,
                    "dawn_dusk": "20 minutes"
                },
                "room": "2E.260"
            },
            "Right Lab Incubator": {
                "temperature": 29.0,
                "light_cycle": {
                    "light_duration": "7am - 9pm",
                    "is_stable": True,
                    "dawn_dusk": "25 minutes"
                },
                "room": "2E.260"
            }
        }

    # def get_properties(self):
    #     return self.incubator_properties
    def to_dict(self, selection: str):
        if selection == "Left Lab Incubator":
            return self.incubator_properties.get("Left Lab Incubator", {})
        elif selection == "Right Lab Incubator":
            return self.incubator_properties.get("Right Lab Incubator", {})
        else:
            raise ValueError("Invalid selection. Choose 'left' or 'right'.")
        