import warnings
from time import sleep

class OnePortEtohExperiment:
    def __init__(self, valve_controller=None, pre_period=300, post_period=300, ipi=0.50, isi=60, num_stim=5, num_pulses=1):
        self.valve_controller = None
        self._pre_period = pre_period
        self._post_period = post_period
        self._ipi = ipi
        self._isi = isi
        self._num_stim = num_stim
        self._num_pulses = num_pulses
        self._recording_duration = 0  # Initialize to 0
        self.frame_count = 0
        self.valve1_flag = False
        self._left_syringe = None
        self._right_syringe = None
        self._etoh_concentration = None
        self._update_recording_duration()  # Calculate initial recording duration
        self.total_duration = self._recording_duration
        print(f"Initial recording duration: {self._recording_duration:.2f}s")

    @property
    def left_syringe(self):
        return self._left_syringe

    @left_syringe.setter
    def left_syringe(self, value):
        if value not in ['H2O', 'EtOH']:
            warnings.warn("Invalid value for left_syringe. Must be 'H2O' or 'EtOH'. Setting left_syringe to None.")
            self._left_syringe = None
        else:
            print(f"Setting left_syringe to {value}")
            self._left_syringe = value

    @property
    def right_syringe(self):
        return self._right_syringe

    @right_syringe.setter
    def right_syringe(self, value):
        if value not in ['H2O', 'EtOH']:
            warnings.warn("Invalid value for right_syringe. Must be 'H2O' or 'EtOH'. Setting right_syringe to None.")
            self._right_syringe = None
        else:
            print(f"Setting right_syringe to {value}")
            self._right_syringe = value

    @property
    def etoh_concentration(self):
        return self._etoh_concentration

    @etoh_concentration.setter
    def etoh_concentration(self, value):
        try:
            if not (0 <= int(value) <= 100):
                warnings.warn("Invalid value for etoh_concentration. Must be between 0 and 100. Setting etoh_concentration to None.")
                self._etoh_concentration = None
            else:
                print(f"Setting etoh_concentration to {value}%")
                self._etoh_concentration = value
        except ValueError:
            warnings.warn("Invalid value for etoh_concentration. Must be an integer. Setting etoh_concentration to None.")
            self._etoh_concentration = None

    @property
    def pre_period(self):
        return self._pre_period

    @pre_period.setter
    def pre_period(self, value):
        print(f"Setting pre_period to {value}")
        self._pre_period = int(value)
        self._update_recording_duration()

    @property
    def post_period(self):
        return self._post_period

    @post_period.setter
    def post_period(self, value):
        print(f"Setting post_period to {value}")
        self._post_period = int(value)
        self._update_recording_duration()

    @property
    def ipi(self):
        return self._ipi

    @ipi.setter
    def ipi(self, value):
        print(f"Setting ipi to {value}")
        self._ipi = float(value)
        self._update_recording_duration()

    @property
    def isi(self):
        return self._isi

    @isi.setter
    def isi(self, value):
        print(f"Setting isi to {value}")
        self._isi = float(value)
        self._update_recording_duration()

    @property
    def num_stim(self):
        return self._num_stim

    @num_stim.setter
    def num_stim(self, value):
        print(f"Setting num_stim to {value}")
        self._num_stim = int(value)
        self._update_recording_duration()

    @property
    def num_pulses(self):
        return self._num_pulses

    @num_pulses.setter
    def num_pulses(self, value):
        print(f"Setting num_pulses to {value}")
        self._num_pulses = int(value)
        self._update_recording_duration()

    @property
    def recording_duration(self):
        return self._recording_duration

    def calculate_recording_duration(self):
        return (
            self._pre_period +
            self._num_stim * (self._ipi * 2 + self._isi) +
            self._post_period
        )

    def _update_recording_duration(self):
        old_duration = self._recording_duration
        self._recording_duration = self.calculate_recording_duration()
        print(f"Updating recording_duration:")
        print(f"  Pre-period: {self._pre_period}s")
        print(f"  Post-period: {self._post_period}s")
        print(f"  Num stimuli: {self._num_stim}")
        print(f"  Num pulses: {self._num_pulses}")
        print(f"  IPI: {self._ipi}s")
        print(f"  ISI: {self._isi}s")
        print(f"  Old duration: {old_duration:.2f}s")
        print(f"  New duration: {self._recording_duration:.2f}s")

    def verify_duration(self):
        calculated_duration = (
            self._pre_period +
            self._num_stim * (self._ipi * 2 + self._isi) +
            self._post_period
        )
        if abs(calculated_duration - self._recording_duration) > 0.001:  # Allow for small floating-point discrepancies
            raise ValueError(f"Calculated duration ({calculated_duration:.2f}s) does not match recording_duration ({self._recording_duration:.2f}s)\n"
                             f"Pre-period: {self._pre_period}s\n"
                             f"Post-period: {self._post_period}s\n"
                             f"Number of stimuli: {self._num_stim}\n"
                             f"Number of pulses: {self._num_pulses}\n"
                             f"Inter-pulse interval (IPI): {self._ipi}s\n"
                             f"Inter-stimulus interval (ISI): {self._isi}s")