"""Analysis program made for USN ultrasound lab.

Investigate and save traces from single-element ultrasound transducers using
Picoscope 5000-series osciloscopes.
GUI interface made in Qt Designer, ver. 5.
Based on earlier NI LabWindows, LabVIEW and Matlab programs. Result file format
is compatible with these, but smaller modifications may be
required in some cases.

Operation
    Sets up a GUI to control the system
    Continously reads traces from the oscilloscope
    Includes an arbitrary waveform generator to transmit shaped pulses

Lars Hoff, USN, Sep 2022
Modified July 2026
    - Follow PEP-8 and numpy docstring style guides.
    - Tested on Ubuntu
    - GUI updated to QT6
    - Massive cleanup in code

Remaining
    - Hardware-control of pulser, pad with zeros for correct rep. rate
"""

import sys
import numpy as np
import matplotlib
from PySide6 import QtWidgets
from PySide6.QtUiTools import loadUiType
import ultrasound_utilities as us         # USN ultrasound lab specific
import ps5000a_ultrasound_wrappers as ps  # Interface to Pico c-library

# Constants
COLOR_WARNING = ['#78281F', '#FADBD8']
COLOR_OK = ['#145A32', '#D4EFDF']
COLOR_NEUTRAL = ['#000000', '#FFFFFF']
COLOR_CH = ['#004B93', '#D32F2F', '#388E3C', '#FBC02D']
COLOR_OFF = '#708090'
COLOR_AWG = '#388E3C'
COLOR_ZOOM = '#C0FFFF'
COLOR_ZOOM_BACKGROUND = '#F0FFFF'
COLOR_AWG_BACKGROUND = '#F5FFFA'

TIMESCALE = 1E-6      # Display scales for time and frequency
FREQUENCYSCALE = 1E6
V_MAX = 20           # Absolute maximum voltage scale

# Set up GUI from Qt5
matplotlib.use('Qt5Agg')
oscilloscope_main_window, QtBaseClass = loadUiType('aquire_ultrasound_gui.ui')


class Display:
    """Settings for display on screen during runtime.

    Attributes
    ----------
    t_min : float
        Start time of part of trace to be analysed.
    t_max : float
        End time of part of trace to be analysed.
    channel : list of bool
        Channels to display on screen.
    """

    def __init__(self) -> None:
        self.t_min = 0.0
        self.t_max = 10.0
        self.channel = [True, True]


class AcquisitionControl:
    """Flags to control running of program.

    Attributes
    ----------
    oscilloscope_ready :  bool
        Osciloscope connected and ready to acquire.
    stop_acquisition :  bool
        Stop data acquisition, do not quit program.
    sampling_changed :  bool
        Sampling updated.
    """

    def __init__(self) -> None:
        self.oscilloscope_ready = False
        self.stop_acquisition = False
        self.sampling_changed = True


class ReadUltrasound(QtBaseClass, oscilloscope_main_window):
    """Start GUI and initialise system."""

    def __init__(self) -> None:
        """Set up GUI and initialise classes and variables."""

        super().__init__()
        self.setupUi(self)

        # Initialise instrument variables.
        self.runstate = AcquisitionControl()
        self.display = Display()         # Scaling and display options
        self.dso = ps.Communication()    # Instrument connection and status.
        self.channel = [ps.Channel(0), ps.Channel(1)]    # Vertical channels
        self.trigger = ps.Trigger()      # Trigger configuration
        self.sampling = ps.Horizontal()  # Horisontal configuration (time)
        self.wfm = us.Waveform()         # Result, storing acquired traces
        self.pulse = us.Pulse()          # Pulse for function generator output
        self.rf_filter = us.WaveformFilter()  # Filtering, for display only
        self.pulse.dt = 1/ps.DAC_SAMPLERATE

        # Open gui and connect initailse graphs
        self.connect_gui()
        fig, axis, graph = self.define_graphs()
        self.fig = fig
        self.axis = axis
        self.graph = graph

        self.update_connected_box(False)
        self.acquireButton.setEnabled(False)
        self.transmitButton.setEnabled(False)
        self.saveButton.setEnabled(False)

    def connect_dso(self) -> int:
        """Connect, configure and start instrument."""
        self.statusBar.showMessage('Connecting instrument')
        errorcode = 0

        # Try to close old handle if resident. May not work
        try:
            if 'openunit' in self.dso.status:
                if not ('close' in self.dso.status):
                    ps.stop_adc(self.dso, self.dso.status)
                    ps.close_adc(self.dso, self.dso.status)
            self.dso.status = {}
        except AttributeError:
            self.dso.status = {}

        # Connect and initialise instrument
        self.dso = ps.open_adc(self.dso)
        if self.dso.connected:
            # Check for signal generator
            self.dso = ps.check_awg(self.dso)

            # Send initial configuration to oscilloscope
            self.dso.status = self.update_vertical()
            self.dso.status = self.update_trigger()

            # Update configuration parameters
            self.update_sampling()
            self.update_pulser()
            self.update_rf_filter()
            self.update_display()

            # Update GUI status
            self.acquireButton.setEnabled(True)
            self.saveButton.setEnabled(False)
            self.connectButton.setEnabled(False)
            self.transmitButton.setEnabled(False)

            self.statusBar.showMessage('Instrument connected')
            self.update_connected_box(True)
            self.update_status_box(self.acquireButton.isChecked())
            self.runstate.oscilloscope_ready = True
            self.runstate.stop_acquisition = False

            errorcode = 0
        else:
            self.statusBar.showMessage('Instrument not connected')
            self.update_connected_box(False)
            self.runstate.oscilloscope_ready = False
            self.runstate.stop_acquisition = False
            errorcode = -1
        return errorcode

    def close_connection(self):
        """Close instrument connection, does not stop program."""
        self.statusBar.showMessage('Closing')
        matplotlib.pyplot.close(self.fig)
        try:
            self.dso.status = ps.close_adc(self.dso)
            errorcode = 0
        except ValueError:
            errorcode = -1
        finally:
            self.close()

        self.statusBar.showMessage('Closed')
        return self.dso.status, errorcode

    # Update oscilloscope settings
    def update_vertical(self):
        """Read vertical settings from GUI and send to instrument."""
        self.channel[0].enabled = True  # Both traces are always aquired
        self.channel[1].enabled = True

        # for k in range(len(self.channel)):
        for channel_no, channel in enumerate(self.channel):

            channel.v_range = us.read_scaled_value(
                self.rangeComboBox[channel_no].currentText())
            channel.v_range = self.channel[channel_no].v_max()
            channel.coupling = self.couplingComboBox[channel_no].currentText()
            channel.offset = self.offsetSpinBox[channel_no].value()

            bwl = self.bwlComboBox[channel_no].currentText()
            channel.bwl = not bwl.casefold().startswith('none')

        if self.dso.connected:
            for channel in self.channel:
                # self.channel[k].no = k
                self.dso.status = ps.set_vertical(self.dso, channel)
                self.dso.status = ps.set_bwl(self.dso, channel)

        return self.dso.status

    def update_trigger(self):
        """Read trigger settings from GUI and send to instrument."""
        self.trigger.source = self.triggerSourceComboBox.currentText()
        self.trigger.direction = self.triggerModeComboBox.currentText()
        self.trigger.level = self.triggerLevelSpinBox.value()
        self.trigger.delay = self.triggerDelaySpinBox.value()*TIMESCALE
        self.trigger.autodelay = self.triggerAutoDelaySpinBox.value()*1e-3

        self.sampling.trigger_position = self.triggerPositionSpinBox.value()

        if self.dso.connected:
            self.dso.status = ps.set_trigger(self.dso, self.trigger,
                                             self.channel, self.sampling)
        return self.dso.status

    def update_sampling(self):
        """Read trace length from GUI and set sample rate."""
        fs_requested = int(self.samplerateSpinBox.value()*FREQUENCYSCALE)
        self.sampling.timebase, fs_actual = ps.find_timebase(fs_requested)
        self.sampling.dt = 1/fs_actual
        self.sampling.n_samples = int(self.nSamplesSpinBox.value()*1e3)

        if self.dso.connected:
            self.sampling.dt = ps.get_sample_interval(self.dso, self.sampling)

        self.samplerateSpinBox.setValue(self.sampling.fs()/FREQUENCYSCALE)

        self.runstate.sampling_changed = True
        return 0

    def update_pulser(self):
        """Send pulse to arbitrary waveform generator.

        Read settings for arbitrary waveform generator (awg)
        Plot pulse and send to instrument
        Graph is updated by replacing data
        """
        print(f'Pulser connected: {self.dso.signal_generator}')
        if not self.dso.signal_generator:
            self.update_transmit_box(available=False)
            return 0    # Does nothing signal genarator not available

        else:
            # Read GUI
            self.pulse.on = self.transmitButton.isChecked()
            self.pulse.envelope = self.pulseEnvelopeComboBox.currentText()
            self.pulse.shape = self.pulseShapeComboBox.currentText()
            self.pulse.f0 = self.pulseFrequencySpinBox.value()*FREQUENCYSCALE
            self.pulse.n_cycles = self.pulseDurationSpinBox.value()
            self.pulse.phase = self.pulsePhaseSpinBox.value()
            self.pulse.a = self.pulseAmplitudeSpinBox.value()

            # Update pulse display
            self.graph['awg'].set_data(self.pulse.t()/TIMESCALE,
                                       self.pulse.y())
            vlim = 1.1 * self.pulse.a
            self.axis['awg'].set(xlim=(0, self.pulse.duration() / TIMESCALE),
                                 ylim=(-vlim, vlim))

            # Calculate and plot pulse spectrum
            f, psd = self.pulse.powerspectrum()
            self.graph['awgspec'].set_data(f/FREQUENCYSCALE, psd)

            # Select line display depending on pulser status
            awg_line = 'solid' if self.pulse.on else 'dotted'
            for g in ['awg', 'awgspec']:
                self.graph[g].set_linestyle(awg_line)

            # Send data to pulser
            self.dso = ps.set_signal(self.dso, self.sampling, self.pulse)
            self.update_display()
            self.update_transmit_box(available=True, on=self.pulse.on)

        return 0

    def update_rf_filter(self):
        """Read RF noise filter settings from GUI."""
        self.rf_filter.sample_rate = self.sampling.fs()
        self.rf_filter.type = self.filterComboBox.currentText()
        self.rf_filter.f_min = self.fminSpinBox.value()*FREQUENCYSCALE
        self.rf_filter.f_max = self.fmaxSpinBox.value()*FREQUENCYSCALE
        self.rf_filter.order = self.filterOrderSpinBox.value()
        return 0

    def control_acquisition(self):
        """Control data acquisition from oscilloscope."""
        if self.acquireButton.isChecked():
            self.acquire_trace()
        else:
            self.stop_acquisition()
        return 0

    def acquire_trace(self):
        """Acquire trace from instrument."""
        if self.runstate.oscilloscope_ready:
            self.runstate.oscilloscope_ready = False
            self.runstate_sampling_changed = True

            self.saveButton.setEnabled(True)
            self.closeButton.setEnabled(False)
            self.transmitButton.setEnabled(self.dso.signal_generator)

            self.update_status_box(True)
            self.statusBar.showMessage('Acquiring data ...')
            while not (self.runstate.stop_acquisition):
                if self.runstate.sampling_changed:
                    self.dso = ps.configure_acquisition(self.dso,
                                                        self.sampling)
                    self.runstate.sampling_changed = False
                self.dso, self.wfm.y = ps.acquire_trace(self.dso,
                                                        self.sampling,
                                                        self.channel)
                self.wfm.dt = self.sampling.dt
                self.wfm.t0 = self.sampling.t0()
                self.plot_result()
        self.update_status_box(False)
        self.statusBar.showMessage('Ready')
        self.runstate.stop_acquisition = False
        return 0

    def stop_acquisition(self):
        """Stop acquisition of traces without closing instrument connection."""
        if not (self.runstate.stop_acquisition):
            self.statusBar.showMessage('Stopping')
            self.update_status_box(False)
        self.runstate.stop_acquisition = True
        self.runstate.oscilloscope_ready = True
        self.closeButton.setEnabled(True)
        self.saveButton.setEnabled(False)
        return 0

    def plot_result(self, time_unit: str = 'us') -> None:
        """Plot measured trace on screen.

        Results are plotted by changing data in line objects representing
        the graphs. All other axis elements are kept

        Parameters
        ----------
        time_unit : str, optional
            Unit to use for the time axis. Must be 's', 'ms', or 'us'.
            The default is 'us'.

        Raises
        ------
        ValueError
            If an unsupported `time_unit` string is provided.
        """
        valid_units = {'s', 'ms', 'us'}
        if time_unit not in valid_units:
            raise ValueError(f"Invalid time_unit '{time_unit}'. "
                             f"Expected one of {valid_units}.")

        wfm_filtered = self.wfm.filtered(self.rf_filter)
        wfm_zoomed = wfm_filtered.zoomed(self.display.t_lim)
        f, psd = wfm_zoomed.powerspectrum(scale='dB', normalise='True')

        x_data = [wfm_filtered.t() / TIMESCALE,
                  wfm_zoomed.t() / TIMESCALE,
                  f / FREQUENCYSCALE]

        for ch_no, ch_name in enumerate(ps.CH_NAMES):
            lines = [self.graph[key][ch_no]
                     for key in ['trace', 'zoom', 'spectrum']]

            if self.display.channel[ch_no]:
                y_data = [wfm_filtered.y[:, ch_no],
                          wfm_zoomed.y[:, ch_no], psd[:, ch_no]]

                for line, x, y in zip(lines, x_data, y_data):
                    line.set_data(x, y)
            else:
                for line in lines:
                    line.set_data([], [])

        # self.fig.canvas.draw()       # --- TRY: Probably necessary
        self.fig.canvas.flush_events()    # --- TRY: Probably unnecessary
        self.update_display()
        return 0

    def save_result(self):
        """Save measured traces and parameters to binary file.

        The filename is generated automatically from a short descriptive
        prefix, the date, and a counter.
        """
        self.statusBar.showMessage('Saving results ...')

        resultfile = us.find_filename(prefix='us',
                                      ext='trc',
                                      resultdir='results')

        self.wfm.save(resultfile.path)
        self.filecounterSpinBox.setValue(resultfile.counter)
        self.resultfileEdit.setText(resultfile.name)
        self.resultpathEdit.setText(resultfile.path)
        self.statusBar.showMessage(f'Result saved to {resultfile.name}')
        return 0

    def _update_ui_element(self, element, message, color) -> str:
        """Helper to set text and stylesheet colors for a QWidget element."""
        element.setText(message)
        element.setStyleSheet(
            f'color: {color[0]}; background-color: {color[1]}')
        return message

    def update_status_box(self, acquiring=False) -> str:
        """Write message to status box, with text and background colors."""
        if not acquiring:
            message, color = 'Stopped', COLOR_WARNING
        else:
            message, color = 'Acquiring', COLOR_OK
        return self._update_ui_element(self.statusEdit, message, color)

    def update_connected_box(self, connected=False) -> str:
        """Write status of the instrument connection."""
        if not connected:
            message, color = 'Not connected', COLOR_WARNING
        else:
            message, color = 'Connected', COLOR_OK
        return self._update_ui_element(self.connectedEdit, message, color)

    def update_transmit_box(self, available=False, on=False) -> str:
        """Write whether the waveform generator is transmitting pulses."""
        if not available:
            message, color = 'Not available', COLOR_NEUTRAL
        elif on:
            message, color = 'Transmitting', COLOR_OK
        else:
            message, color = 'Off', COLOR_WARNING

        return self._update_ui_element(self.transmitStatusEdit, message, color)

    def update_status(self, message, append=False) -> str:
        """Update status message field at bottom of window."""
        if append:
            message = f'{self.status_textEdit.toPlainText()}{message}'

        self.status_textEdit.setText(message)
        return message

    def find_voltagescale(self, vmax) -> tuple[float, str]:
        """Find scale for voltage axis based on maximum value.

        Parameters
        ----------
        vmax : float
            Maximum voltage value in Volts.

        Returns
        -------
        voltage_scale : float
            Voltage scale multiplier (e.g., 1e-6, 1e-3, or 1).
        unit : str
            Voltage scale unit text (e.g., 'uV', 'mV', or 'V').
        """
        if vmax < 1e-3:
            voltage_scale, unit = 1e-6, 'uV'
        elif vmax < 1:
            voltage_scale, unit = 1e-3, 'mV'
        else:
            voltage_scale, unit = 1, 'V'

        return voltage_scale, unit

    def update_display(self, time_unit='us') -> int:
        """Update values and markers on screen.

        Parameters
        ----------
        time_unit : str, optional
            Unit to use on time axis, 's', 'ms', or 'us'. The default is 'us'.

        Returns
        -------
        int
            Returns 0 upon successful completion.
        """
        # Full trace
        t0_scaled = self.sampling.t0() / TIMESCALE
        tmax_scaled = self.sampling.t_max() / TIMESCALE
        self.axis['trace'][0].set_xlim(t0_scaled, tmax_scaled)

        # Selected interval, 'zoom'
        zoom_range = [self.zoomStartSpinBox.value(),
                      self.zoomEndSpinBox.value()]
        t_lim = us.find_limits(zoom_range, min_diff=0.1)

        self.graph['zoom_area'].set_x(t_lim[0])
        self.graph['zoom_area'].set_width(t_lim[1] - t_lim[0])
        self.axis['zoom'][0].set_xlim(t_lim)
        self.display.t_lim = t_lim * TIMESCALE

        # Vertical scale & Channels
        db_range = [self.dbMinSpinBox.value(), self.dbMaxSpinBox.value()]
        db_lim = us.find_limits(db_range)

        self.display.channel = [btn.isChecked() for btn in self.chButton]
        channel_data = zip(self.display.channel,
                           self.chButton,
                           self.chLabel,
                           self.displayrangeComboBox,
                           self.axis['zoom'],
                           self.axis['trace'],
                           self.axis['spectrum'],
                           self.channel)
        for k, data in enumerate(channel_data):
            (is_on, btn, label, vrange,
             ax_zoom, ax_trace, ax_spectrum, ch) = data

            bg_color = COLOR_CH[k] if is_on else COLOR_OFF
            btn.setStyleSheet(f"color: white; background-color: {bg_color};")
            label.setStyleSheet(f"color: white; background-color: {bg_color};")

            vzoom = us.read_scaled_value(vrange.currentText())
            ax_zoom.set_ylim(-vzoom, vzoom)

            v_max = ch.v_max()
            ax_trace.set_ylim(-v_max, v_max)
            ax_spectrum.set_ylim(db_lim)

        # Frequency axis
        f_range = [self.zoomFminSpinBox.value(), self.zoomFmaxSpinBox.value()]
        f_lim = us.find_limits(f_range, min_diff=0.1)
        self.axis['spectrum'][0].set_xlim(f_lim)
        self.axis['awgspec'].set_ylim(db_lim)
        self.axis['awgspec'].set_xlim(f_lim)

        self.fig.canvas.draw()
        return 0

    def connect_gui(self) -> int:
        """Connect GUI to functions"""
        # Display scales
        for spin_box in (self.zoomStartSpinBox, self.zoomEndSpinBox,
                         self.zoomFminSpinBox, self.zoomFmaxSpinBox,
                         self.dbMinSpinBox, self.dbMaxSpinBox):
            spin_box.valueChanged.connect(self.update_display)

        # RF-filter
        self.filterComboBox.activated.connect(self.update_rf_filter)
        for spin_box in (self.fminSpinBox, self.fmaxSpinBox,
                         self.filterOrderSpinBox):
            spin_box.valueChanged.connect(self.update_rf_filter)

        # Trigger
        for combo_box in (self.triggerSourceComboBox,
                          self.triggerModeComboBox):
            combo_box.activated.connect(self.update_trigger)

        for spin_box in (
                self.triggerPositionSpinBox, self.triggerLevelSpinBox,
                self.triggerDelaySpinBox, self.triggerAutoDelaySpinBox):
            spin_box.valueChanged.connect(self.update_trigger)

        # Horizontal (Time, sampling)
        self.samplerateSpinBox.valueChanged.connect(self.update_sampling)
        self.nSamplesSpinBox.valueChanged.connect(self.update_sampling)

        # Pulse generator (AWG)
        self.transmitButton.clicked.connect(self.update_pulser)
        for combo_box in (self.pulseEnvelopeComboBox,
                          self.pulseShapeComboBox):
            combo_box.activated.connect(self.update_pulser)

        for spin_box in (
                self.pulseFrequencySpinBox, self.pulseDurationSpinBox,
                self.pulsePhaseSpinBox, self.pulseAmplitudeSpinBox):
            spin_box.valueChanged.connect(self.update_pulser)

        # Program flow
        self.connectButton.clicked.connect(self.connect_dso)
        self.acquireButton.clicked.connect(self.control_acquisition)
        self.saveButton.clicked.connect(self.save_result)
        self.closeButton.clicked.connect(self.close_connection)

        # Vertical channels, voltage A and B
        self.chButton = [self.chAButton, self.chBButton]
        self.chLabel = [self.chALabel, self.chBLabel]
        self.displayrangeComboBox = [self.displayrangeAComboBox,
                                     self.displayrangeBComboBox]
        self.rangeComboBox = [self.rangeAComboBox, self.rangeBComboBox]
        self.couplingComboBox = [self.couplingAComboBox,
                                 self.couplingBComboBox]
        self.offsetSpinBox = [self.offsetASpinBox, self.offsetBSpinBox]
        self.bwlComboBox = [self.bwlAComboBox, self.bwlBComboBox]

        zipped_channels = zip(self.chButton,
                              self.chLabel,
                              self.displayrangeComboBox,
                              self.rangeComboBox,
                              self.couplingComboBox,
                              self.offsetSpinBox,
                              self.bwlComboBox,
                              COLOR_CH)

        for (button, label, disp_range, acq_range,
             coupling, offset, bwl, ch_color) in zipped_channels:

            button.clicked.connect(self.update_display)
            disp_range.activated.connect(self.update_display)
            acq_range.activated.connect(self.update_vertical)
            coupling.activated.connect(self.update_vertical)
            offset.valueChanged.connect(self.update_vertical)
            bwl.activated.connect(self.update_vertical)

            button.setStyleSheet(f'color: white; background-color: {ch_color}')
            label.setStyleSheet(f'color: white; background-color: {ch_color}')
            disp_range.setStyleSheet(
                f'color: white; background-color: {ch_color}')

        return 0

    def define_graphs(self):
        """
        Initialise result graphs, layout, titles, scales, colours etc.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle to the result figure.
        axis : list of matplotlib.axes.Axes
            Handles to graphs (subplots) in the figure.
        graph : list of matplotlib.lines.Line2D
            Handles to plots inside the subplots.
        """

        matplotlib.pyplot.ion()   # Does not seem to make any difference?

        # Figure layout
        axgrid = [['trace']*3,
                  ['awg'] + ['zoom']*2,
                  ['awgspec'] + ['spectrum']*2]
        fig, axis = matplotlib.pyplot.subplot_mosaic(axgrid,
                                                     figsize=(12, 8),
                                                     layout='constrained')

        axis['trace'].set_title('Acquired traces', loc='left')
        axis['zoom'].set_title('Selected interval', loc='left')
        axis['awg'].set_title('Pulser', loc='left')

        # Time graphs
        for key in ['trace', 'zoom', 'awg']:
            axis[key].set_xlabel('Time [us]')
            axis[key].set_ylabel('Voltage [V]')
            axis[key].set_xlim(0, 1)
            axis[key].grid(True)

        # Frequency graphs
        for key in ['spectrum', 'awgspec']:
            axis[key].set_xlabel('Frequency [MHz]')
            axis[key].set_ylabel('Power [dB re. max]')
            axis[key].set_xlim(0, 1)
            axis[key].grid(True)

        # Color backgrounds to identify axes
        for key in ['zoom', 'spectrum']:
            axis[key].set_facecolor(COLOR_ZOOM_BACKGROUND)
        for key in ['awg', 'awgspec']:
            axis[key].set_facecolor(COLOR_AWG_BACKGROUND)

        # Dual y-axis for two channels
        for key in ['trace', 'zoom', 'spectrum']:
            axis[key] = [axis[key], axis[key].twinx()]

        for ch_idx, ch in enumerate(ps.CH_NAMES):
            color = COLOR_CH[ch_idx]

            for g in ['trace', 'zoom']:
                ax = axis[g][ch_idx]
                ax.set_ylabel('Voltage [V]', color=color)
                ax.tick_params(axis='y', colors=color)

            ax_spec = axis['spectrum'][ch_idx]
            ax_spec.set_ylabel('Power [dB re. max]', color=color)
            ax_spec.tick_params(axis='y', colors=color)

        # Create empty gaphs, updated with data during measurement
        graph = {}

        graph['zoom_area'] = axis['trace'][0].axvspan(0, 0, color=COLOR_ZOOM)
        graph['awg'], = axis['awg'].plot([], [], color=COLOR_AWG)
        graph['awgspec'], = axis['awgspec'].plot([], [], color=COLOR_AWG)

        for key in ['trace', 'zoom', 'spectrum']:
            graph[key] = [ax.plot([], [], color=color)[0]
                          for ax, color in zip(axis[key], COLOR_CH)]

        fig.show()
        return fig, axis, graph


# Main function
if __name__ == '__main__':
    app = QtWidgets.QApplication.instance()

    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    app.setStyle('Fusion')
    window = ReadUltrasound()
    window.show()

    sys.exit(app.exec())
