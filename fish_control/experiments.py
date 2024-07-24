import warnings
from time import sleep, monotonic
from experiment_objects import ValveController
from camera_objects import VimbaCameraThread, BaslerCameraThread

class OnePortEtohExperiment:
    always_display = ['pre_period', 'post_period', 'ipi', 'isi', 'num_stim', 'num_pulses', 
                      'left_syringe', 'right_syringe', 'etoh_concentration']
    requires_camera = True
    requires_arduino = True
    requires_h5_logging = True
    camera_class = VimbaCameraThread

    def __init__(self, valve_controller=None, pre_period=300, post_period=300, ipi=0.50, isi=60, num_stim=5, num_pulses=1):
        self.valve_controller = None
        self._pre_period = pre_period
        self._post_period = post_period
        self._ipi = ipi
        self._isi = isi
        self._num_stim = num_stim
        self._num_pulses = num_pulses
        self._recording_duration = 0
        self._left_syringe = None
        self._right_syringe = None
        self._etoh_concentration = None
        self._update_recording_duration()

    @property
    def pre_period(self):
        return self._pre_period

    @pre_period.setter
    def pre_period(self, value):
        self._pre_period = int(value)
        self._update_recording_duration()

    @property
    def post_period(self):
        return self._post_period

    @post_period.setter
    def post_period(self, value):
        self._post_period = int(value)
        self._update_recording_duration()

    @property
    def ipi(self):
        return self._ipi

    @ipi.setter
    def ipi(self, value):
        self._ipi = float(value)
        self._update_recording_duration()

    @property
    def isi(self):
        return self._isi

    @isi.setter
    def isi(self, value):
        self._isi = float(value)
        self._update_recording_duration()

    @property
    def num_stim(self):
        return self._num_stim

    @num_stim.setter
    def num_stim(self, value):
        self._num_stim = int(value)
        self._update_recording_duration()

    @property
    def num_pulses(self):
        return self._num_pulses

    @num_pulses.setter
    def num_pulses(self, value):
        self._num_pulses = int(value)
        self._update_recording_duration()

    @property
    def left_syringe(self):
        return self._left_syringe

    @left_syringe.setter
    def left_syringe(self, value):
        if value in ['H2O', 'EtOH']:
            self._left_syringe = value
        else:
            raise ValueError("Invalid value for left_syringe. Must be 'H2O' or 'EtOH'.")

    @property
    def right_syringe(self):
        return self._right_syringe

    @right_syringe.setter
    def right_syringe(self, value):
        if value in ['H2O', 'EtOH']:
            self._right_syringe = value
        else:
            raise ValueError("Invalid value for right_syringe. Must be 'H2O' or 'EtOH'.")

    @property
    def etoh_concentration(self):
        return self._etoh_concentration

    @etoh_concentration.setter
    def etoh_concentration(self, value):
        if 0 <= int(value) <= 100:
            self._etoh_concentration = int(value)
        else:
            raise ValueError("Invalid value for etoh_concentration. Must be between 0 and 100.")

    @property
    def recording_duration(self):
        return self._recording_duration

    def _update_recording_duration(self):
        self._recording_duration = (
            self._pre_period +
            self._num_stim * (self._ipi * self._num_pulses + self._isi) +
            self._post_period
        )
        print(f"Recording duration: {self._recording_duration}")
        print(f"Pre-period: {self._pre_period}")
        print(f"Post-period: {self._post_period}")
        print(f"Number of stimuli: {self._num_stim}")
        print(f"Number of pulses: {self._num_pulses}")
        print(f"Inter-pulse interval (IPI): {self._ipi}")
        print(f"Inter-stimulus interval (ISI): {self._isi}")

    def verify_duration(self):
        calculated_duration = (
            self._pre_period +
            self._num_stim * (self._ipi * self._num_pulses + self._isi) +
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

    def handle_valve_event(self, thread, event, time):
            event_parts = event.split('_')
            if len(event_parts) == 4:  # Should be 'valve', 'open/close', stim_num, pulse_num
                valve_action, stim_num, pulse_num = event_parts[1], int(event_parts[2]), int(event_parts[3])
                thread.update_signal.emit(f"Valve event: {valve_action} for stimulus {stim_num}, pulse {pulse_num} at time {time}")
                self.add_event_to_h5(thread, f'valve_{valve_action}', stim_num, pulse_num, time)
            else:
                thread.update_signal.emit(f"Unexpected valve event format: {event}")

    def add_event_to_h5(self, thread, event, stim_number, pulse_number, timestamp):
        thread.h5_writer.add_event_to_queue(event, stim_number, pulse_number, timestamp)
        thread.update_signal.emit(f"Added event to H5: {event} (Stim: {stim_number}, Pulse: {pulse_number}) at {timestamp}")

    def run(self, thread):
        thread.update_signal.emit("Running OnePortEtohExperiment...")
        
        # Initialize ValveController here
        if not hasattr(thread, 'valve_controller'):
            thread.update_signal.emit("Initializing ValveController...")
            thread.valve_controller = ValveController(thread.board)

        # Connect valve_operated signal to a method that will add events to the H5 queue
        thread.valve_controller.valve_operated.connect(lambda event, time: self.handle_valve_event(thread, event, time))

        # Pre-period
        thread.update_signal.emit(f"Pre stimulus period ---- duration = {self.pre_period}")
        thread.h5_writer.add_event_to_queue('prestimulus_start', -1, -1, monotonic())
        thread.sleep(self.pre_period)
        thread.h5_writer.add_event_to_queue('pre_stimulus_end', -1, -1, monotonic())
        
        for stim in range(self.num_stim):
            if thread.stop_flag.is_set():
                break
            thread.update_signal.emit(f"Stimulus {stim+1} start")
            for pulse in range(self.num_pulses):
                if thread.stop_flag.is_set():
                    break
                thread.update_signal.emit(f"Operating valve for stimulus {stim+1}, pulse {pulse+1}")
                thread.valve_controller.operate_valve(self.ipi, stim, pulse)
            thread.sleep(self.isi)
            thread.update_signal.emit(f"Stimulus {stim+1} end")

        # Post-period
        if not thread.stop_flag.is_set():
            thread.update_signal.emit(f"Post stimulus period ---- duration = {self.post_period}")
            thread.h5_writer.add_event_to_queue('post_stimulus_start', -1, -1, monotonic())
            thread.sleep(self.post_period)
            thread.h5_writer.add_event_to_queue('post_stimulus_end', -1, -1, monotonic())

        thread.h5_writer.add_event_to_queue('experiment_end', -1, -1, monotonic())
        thread.update_signal.emit("Experiment run completed")

class EtOHBathExperiment:
    always_display = ['pre_period', 'experiment_period', 'num_lanes']
    requires_camera = True
    requires_arduino = False
    requires_h5_logging = True
    camera_class = BaslerCameraThread

    def __init__(self, pre_period=300, experiment_period=900, num_lanes=4):
        self._pre_period = pre_period
        self._experiment_period = experiment_period
        self._num_lanes = num_lanes
        self._recording_duration = 0
        self._subjects = [''] * num_lanes
        self._update_recording_duration()

    @property
    def pre_period(self):
        return self._pre_period
    
    @pre_period.setter
    def pre_period(self, value):
        self._pre_period = int(value)
        self._update_recording_duration()

    @property
    def experiment_period(self):
        return self._experiment_period
    
    @experiment_period.setter
    def experiment_period(self, value):
        self._experiment_period = int(value)
        self._update_recording_duration()

    @property
    def num_lanes(self):
        return self._num_lanes

    @num_lanes.setter
    def num_lanes(self, value):
        self._num_lanes = int(value)
        self._subjects = [''] * self._num_lanes

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, value):
        if len(value) != self._num_lanes:
            raise ValueError(f"Number of subjects must match number of lanes ({self._num_lanes})")
        self._subjects = value

    @property
    def recording_duration(self):
        return self._recording_duration
    
    def _update_recording_duration(self):
        self._recording_duration = self._pre_period + self._experiment_period

    def verify_duration(self):
        calculated_duration = self._pre_period + self._experiment_period
        if abs(calculated_duration - self._recording_duration) > 0.001:
            raise ValueError(f"Calculated duration ({calculated_duration:.2f}s) does not match recording_duration ({self._recording_duration:.2f}s)\n"
                             f"Pre-period: {self._pre_period}s\n"
                             f"Experiment period: {self._experiment_period}s")

    def run(self, thread):
        thread.update_signal.emit("Running EtOHBathExperiment...")
        
        # Pre-period
        thread.update_signal.emit(f"Pre bath period ---- duration = {self.pre_period}")
        thread.h5_writer.add_event_to_queue('pre_bath_start', -1, -1, monotonic())
        thread.sleep(self.pre_period)
        thread.h5_writer.add_event_to_queue('pre_bath_end', -1, -1, monotonic())
        
        # Experiment period
        thread.update_signal.emit(f"Bath period ---- duration = {self.experiment_period}")
        thread.h5_writer.add_event_to_queue('bath_start', -1, -1, monotonic())
        thread.sleep(self.experiment_period)
        thread.h5_writer.add_event_to_queue('bath_end', -1, -1, monotonic())

        thread.h5_writer.add_event_to_queue('experiment_end', -1, -1, monotonic())
        thread.update_signal.emit("Experiment completed")