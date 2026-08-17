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
    - Massive cleanup in code with help from Gemini

Remaining
    - Hardware-control of pulser, pad with zeros for correct rep. rate
"""
from PySide6 import QtWidgets
from PySide6.QtWidgets import QApplication
from PySide6.QtUiTools import loadUiType

# import time  # For profiling
import sys
import numpy as np
import matplotlib
from dataclasses import dataclass

import ultrasound_utilities as us         # USN ultrasound lab specific
import ps5000a_ultrasound_wrappers as ps  # Interface to Pico c-library

# Constants
COLOR = {'warning': ('#78281F', '#FADBD8'),
         'ok': ('#145A32', '#D4EFDF'),
         'neutral': ('#000000', '#FFFFFF'),
         'off': '#708090',
         'channel': ('#004B93', '#D32F2F', '#388E3C', '#FBC02D'),
         'awg': ('#388E3C', '#F5FFFA'),
         'zoom': ('#B0E0E6', '#E0FFFF'),
         }

TIMESCALE = 1E-6      # Display scales for time and frequency
FREQUENCYSCALE = 1E6
V_MAX = 20           # Absolute maximum voltage scale

matplotlib.use('QtAgg')
oscilloscope_main_window, QtBaseClass = loadUiType('aquire_ultrasound_gui.ui')


@dataclass
class Display:
    """Settings for display on screen during runtime."""
    t_min: float = 0.0      # Start time of part of trace to be analysed.
    t_max: float = 10.0     # End time of part of trace to be analysed.
    channel = [True, True]  # Channels to display on screen.


@dataclass
class AcquisitionControl:
    """Flags to control running of program."""
    oscilloscope_ready: bool = False  # Osciloscope connected and ready
    stop_acquisition: bool = False    # Stop acquisition, do not quit program.
    sampling_changed: bool = True     # Sampling parameters changed
    scales_changed: bool = True       # display scales have changed


class ReadUltrasound(QtBaseClass, oscilloscope_main_window):
    """Main ultrasound acquisition GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self._position_window()
        self._initialize_components()
        self._initialize_graphs()
        self._configure_gui()

    def _position_window(self) -> None:
        """Set GUI window position on screen."""
        geometry = QApplication.primaryScreen().geometry()
        self.move(int(0.02 * geometry.width()),
                  int(0.05 * geometry.height()))
        return

    def _initialize_components(self) -> None:
        """Initialize acquisition, processing, and display."""
        self.runstate = AcquisitionControl()

        # Display and oscilloscope
        self.dso = ps.Picoscope5000A()
        self.display = Display()
        self.sampling = ps.Horizontal()
        self.channel = [ps.Channel(i) for i in range(2)]
        self.trigger = ps.Trigger()

        # Waveform processing
        self.wfm = us.Waveform()
        self.pulse = us.Pulse()
        self.rf_filter = us.WaveformFilter()
        self.pulse.dt = 1 / ps.DAC_SAMPLERATE

        return

    def _initialize_graphs(self) -> None:
        """Create result graphs."""
        self.fig, self.axis, self.graph = self.define_graphs()
        return

    def _configure_gui(self) -> None:
        """Connect signals and initialize widget states."""
        self.connect_gui()
        self.update_connected_box(False)

        # Disable controls until oscilloscope is connected
        for button in (self.acquireButton,
                       self.transmitButton,
                       self.saveButton):
            button.setEnabled(False)

        return

    def connect_dso(self) -> int:
        """Connect, configure, and start the digital storage oscilloscope.

        First, attempts to clean up any existing active connections.
        Then, opens a new session with the instrumen, and applies the initial
        configuration (vertical settings, trigger, sampling, pulser,
        RF filter, and display).
        Also updates GUI buttons and status messages bars based on the
        connection result.

        Returns
        -------
        int
            Status code of the connection attempt.
             0 : Connection successful and instrument configured.
            -1 : Connection failed.

        Raises
        ------
        AttributeError
            Handled internally if `self.dso.status` does not exist during
            the initial cleanup phase.

        """
        self.statusBar.showMessage('Connecting instrument')
        errorcode = 0

        # Try to close old handle if resident. Unreliable
        try:
            if 'openunit' in self.dso.status:
                if not ('close' in self.dso.status):
                    self.dso.stop_adc()
                    self.dso.close_adc()
            self.dso.status = {}
        except AttributeError:
            self.dso.status = {}

        # Connect and initialise instrument
        self.dso.open_adc()
        if self.dso.connected:
            # Check for signal generator, remove graphs if not present
            self.dso.check_awg()
            if not self.dso.signal_generator:
                self.axis['awg'].remove()
                self.axis['awgspec'].remove()

            # Send initial configuration to oscilloscope
            self.update_vertical()
            self.update_trigger()
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

    def close_connection(self) -> tuple[str, int]:
        """Close instrument connection without stopping the program.

        Attempts to close the connection to the oscilloscope.
        If the instrument is not connected or fails to respond, the function
        catches the error to prevent the user interface from crashing.

        Returns
        -------
        status : str
            The connection status of the DSO
        errorcode : int
             0 : Successful closure
            -1 : Error, e.g., instrument was not connected
        """
        self.statusBar.showMessage("Closing")

        if hasattr(self, "fig") and self.fig is not None:
            matplotlib.pyplot.close(self.fig)

        errorcode = 0
        try:
            if hasattr(self, "dso") and self.dso is not None:
                self.dso.close_adc()
            else:
                errorcode = -1
        except Exception:
            errorcode = -1
        finally:
            self.close()

        self.statusBar.showMessage("Closed")
        return self.dso, errorcode

    def update_vertical(self):
        """Read vertical settings from GUI and apply them to oscilloscope.

        Synchronizes the software channel configurations with current states
        of the GUI widgets (range, coupling, offset, bandwidth limits) for
        all available channels. Transmits settings to the oscilloscope if it
        is connected.

        Side Effects
        ------------
        - Enables both `self.channel[0]` and `self.channel[1]` unconditionally.
        - Modifies properties (`v_range`, `coupling`, `offset`, `bwl`) on all
          channel objects within `self.channel`.
        - Transmits configurations to `self.dso` if a hardware connection is
          active.

        Notes
        -----
        Both input traces are hardcoded to always be acquired, regardless of
        individual GUI enable/disable states, they are only for display.
        """
        for channel_no, channel in enumerate(self.channel):
            channel.enabled = True  # All traces are always aquired

            channel.v_range = us.read_scaled_value(
                self.rangeComboBox[channel_no].currentText())

            channel.v_range = channel.v_max
            channel.coupling = self.couplingComboBox[channel_no].currentText()
            channel.offset = self.offsetSpinBox[channel_no].value()

            bwl = self.bwlComboBox[channel_no].currentText()
            channel.bwl = not bwl.casefold().startswith('none')

        if self.dso.connected:
            for channel in self.channel:
                self.dso.set_vertical(channel)
                self.dso.set_bwl(channel)

        self.update_display()
        return

    def update_trigger(self):
        """Read trigger settings from the GUI and send them to the instrument.

        Reads user inputs from the trigger-related GUI controls, scales values,
        updates internal trigger , and transfers configuration to the
        oscilloscope.
        Updates the 'sampling_changed' flag to notify time scale may
        have changed.

        """
        # Read settings from GUI
        self.trigger.source = self.triggerSourceComboBox.currentText()
        self.trigger.direction = self.triggerModeComboBox.currentText()
        self.trigger.level = self.triggerLevelSpinBox.value()
        self.trigger.delay = self.triggerDelaySpinBox.value()*TIMESCALE
        self.trigger.autodelay = self.triggerAutoDelaySpinBox.value()*1e-3
        self.sampling.trigger_position = self.triggerPositionSpinBox.value()

        # Transmit configuration to oscilloscope
        if self.dso.connected:
            self.dso.set_trigger(self.trigger, self.channel, self.sampling)

        self.runstate.sampling_changed = True
        return

    def update_sampling(self):
        """Read sampling settings from THE GUI and configure the timebase.

        Reads requested sample rate and number of samples from GUI,
        queries the instrument hardware to find the closest matching
        hardware timebase, and updates internal sampling parameters.
        Verifies the actual hardware sampling interval and updates the GUI
        value to display the actual hardware-supported sample rate.

        Side Effects
        ------------
        - Modifies properties (`timebase`, `dt`, `n_samples`) on the
          `self.sampling` object.
        - Updates `self.samplerateSpinBox` to reflect the actual,
          hardware-limited sampling frequency.
        - Sets the boolean flag `self.runstate.sampling_changed` to notify
          that configuratiohas changed.
        - Calls methods on `self.dso` to query timebase and sample interval.
        """
        fs_requested = int(self.samplerateSpinBox.value()*FREQUENCYSCALE)
        self.sampling.n_samples = int(self.nSamplesSpinBox.value()*1e3)

        if self.dso.connected:
            self.sampling.timebase, fs_actual = self.dso.find_timebase(
                fs_requested)
            self.sampling.dt = self.dso.get_sample_interval(self.sampling)

            self.samplerateSpinBox.setValue(
                self.sampling.sample_rate/FREQUENCYSCALE)

            self.runstate.sampling_changed = True
        return

    def update_pulser(self):
        """Read pulser settings from GUI, update plots, and program the AWG.

        Reads arbitrary waveform generator (AWG) parameters from
        the GUI (envelope, shape, frequency, duration, phase, and amplitude)
        and transfers them to the AWG in the oscilloscope.
        Computes and updates time-domain pulse and power spectrum graphs.
        Transfers the pulse data the oscilloscope If an AWG is supported.
        Aborts early if no AWG is present.

        Returns
        -------
        int
            Status code of the pulser update attempt.
             0 : Successfully processed settings, updated plots, and sent data.
            -1 : Aborted because the hardware does not support a signal
                 generator.

        Side Effects
        ------------
        - Updates data and axis limits for the `self.graph['awg']` plot.
        - Alters the line style ('solid' vs 'dotted') depending on whether
          the pulser output is active.
        """
        if not self.dso.signal_generator:
            self.update_transmit_box(available=False)
            return -1    # Does nothing, signal genarator not available

        else:
            # Read GUI
            self.pulse.on = self.transmitButton.isChecked()
            self.pulse.envelope = self.pulseEnvelopeComboBox.currentText()
            self.pulse.shape = self.pulseShapeComboBox.currentText()
            self.pulse.f0 = self.pulseFrequencySpinBox.value()*FREQUENCYSCALE
            self.pulse.n_cycles = self.pulseDurationSpinBox.value()
            self.pulse.phase = self.pulsePhaseSpinBox.value()
            self.pulse.a = self.pulseAmplitudeSpinBox.value()

            # Update pulse display, rescale axes
            self.graph['awg'].set_data(self.pulse.t/TIMESCALE,
                                       self.pulse.y)
            vlim = 1.1 * self.pulse.a
            self.axis['awg'].set(xlim=(0, self.pulse.duration / TIMESCALE),
                                 ylim=(-vlim, vlim))

            # Calculate and plot pulse spectrum
            f, psd = self.pulse.powerspectrum()
            self.graph['awgspec'].set_data(f/FREQUENCYSCALE, psd)

            # Select line type depending on pulser status
            awg_line = 'solid' if self.pulse.on else 'dotted'
            for g in ['awg', 'awgspec']:
                self.graph[g].set_linestyle(awg_line)

            # Send data to pulser
            self.dso.set_signal(self.sampling, self.pulse)
            self.update_display()
            self.update_transmit_box(available=True, on=self.pulse.on)

        return 0

    def update_rf_filter(self):
        """Read RF noise filter settings from GUI and update filter state.

        Reads filter type, cutoff frequencies, and filter order from GUI.
        Reads the sample rate from the system to scale the filter.
        """
        self.rf_filter.sample_rate = self.sampling.sample_rate
        self.rf_filter.type = self.filterComboBox.currentText()
        self.rf_filter.f_min = self.fminSpinBox.value()*FREQUENCYSCALE
        self.rf_filter.f_max = self.fmaxSpinBox.value()*FREQUENCYSCALE
        self.rf_filter.order = self.filterOrderSpinBox.value()
        return

    def control_acquisition(self):
        """Control data acquisition from oscilloscope.

        Checks state of the 'Acquire' button on the GUI.
        Button checked: Starts a new acquisition loop.
        Button unchecked: Requests the instrument to stop ongoing data
        acquistion.
        """
        if self.acquireButton.isChecked():
            self.acquire_trace()
        else:
            self.stop_acquisition()
        return

    def acquire_trace(self):
        """Acquire waveforms from the oscilloscope in a continuous loop.

        Locks the acquisition state and enters a polling loop that
        captures data traces from the hardware as long as the shutdown flag
        (`stop_acquisition`) is false. Forces the hardware to reconfigure if
        sampling parameters were modified prior to or during the loop.


        Side Effects
        ------------
        - Disables `Close` and enables `Save` buttons during execution.

        Notes
        -----
        .. warning::
           This method contains a blocking `while` loop. If this class runs on
           the main GUI thread (e.g., PyQt/PySide main thread), running this
           loop will freeze the user interface unless
           `QCoreApplication.processEvents()` is called inside the loop,
           or this entire method is executed within a separate `QThread`.
        """
        if self.runstate.oscilloscope_ready:
            # Update GUI controls and messages
            self.saveButton.setEnabled(True)
            self.closeButton.setEnabled(False)
            self.transmitButton.setEnabled(self.dso.signal_generator)
            self.update_status_box(True)
            self.statusBar.showMessage('Acquiring data ...')

            # Initialise status flags
            self.runstate.oscilloscope_ready = False
            self.runstate.sampling_changed = True
            self.runstate.scales_changed = True

            while not (self.runstate.stop_acquisition):
                # Reconfigure if sapling parameteres changed
                if self.runstate.sampling_changed:
                    self.dso.configure_acquisition(self.sampling)
                    self.wfm.dt = self.sampling.dt
                    self.wfm.t0 = self.sampling.start_time

                    self.runstate.sampling_changed = False
                    self.runstate.scales_changed = True
                    self.update_display()

                # Read and interpret result from osciloscope
                self.wfm.y = self.dso.acquire_trace(self.sampling,
                                                    self.channel)

                self.plot_result()

        self.update_status_box(False)
        self.statusBar.showMessage('Ready')
        self.runstate.stop_acquisition = False
        return

    def stop_acquisition(self):
        """Stop the continuous waveform acquisition loop safely.

        Flags active acquisition loop to stop after next iteration by setting
        the 'stop_acquisition' flag.
        Updates user interface status indicators and restores states of the
        GUI buttons.

        Side Effects
        ------------
        - Resets the `oscilloscope_ready` flag to True.
        - Enables the GUI `Close' button, disables the `Save' button.
        """
        self.runstate.stop_acquisition = True
        self.runstate.oscilloscope_ready = True

        self.closeButton.setEnabled(True)
        self.saveButton.setEnabled(False)

        self.statusBar.showMessage('Stopping')
        self.update_status_box(False)

        return

    def plot_result(self):
        """Process and plot the measured waveform trace on the screen.

        Applies the RF noise filter to the current waveform, crops the signal
        to the display time limits, and calculates its power spectrum
        in decibels (dB). It then updates the data of existing
        Matplotlib line objects for each active channel across three plots:
        raw trace, zoomed trace, and spectrum. Inactive channels are cleared
        from the display.

        Side Effects
        ------------
        - Extracts and transforms data from `self.wfm` via filtering and
          zooming.
        - Modifies the data arrays of Matplotlib line objects inside
          `self.graph`
          by calling `set_data()`.
        - Empties lines for channels that are disabled in
          `self.display.channel`.
        - Forces Matplotlib to process pending events via
          `self.fig.canvas.flush_events()`.
        - Triggers an interface redraw by invoking `self.update_display()`.

        Notes
        -----
        - Updating plot data using `line.set_data()` is significantly faster
          than clearing and redrawing the entire axes. This is critical for
          rapid continuous acquisition.
        - *Developer note:* If the plots fail to visually refresh on screen,
          uncommenting `self.fig.canvas.draw()` right before `flush_events()`
          is usually required in Matplotlib interactive modes.
        """
        # Filter, zoom, and find power spectrum
        wfm_filtered = self.wfm.filtered(self.rf_filter)
        wfm_zoomed = wfm_filtered.zoomed(self.display.t_lim)
        f, psd = wfm_zoomed.powerspectrum(scale='dB',
                                          normalise='True',
                                          upsample=0)

        # Decimate traces for fast rendering
        q = int(self.sampling.n_samples // 2000)
        t_full = wfm_filtered.t[::q] / TIMESCALE
        t_zoom = wfm_zoomed.t / TIMESCALE
        y_full = wfm_filtered.y[::q, :]
        y_zoomed = wfm_zoomed.y

        x_data = [t_full,
                  t_zoom,
                  f / FREQUENCYSCALE]

        for ch_no, ch_name in enumerate(ps.CHANNEL_NAMES):
            lines = [self.graph[key][ch_no]
                     for key in ['trace', 'zoom', 'spectrum']]

            if self.display.channel[ch_no]:
                y_data = [y_full[:, ch_no],
                          y_zoomed[:, ch_no],
                          psd[:, ch_no]]

                for line, x, y in zip(lines, x_data, y_data):
                    line.set_data(x, y)
            else:
                for line in lines:
                    line.set_data([], [])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        return

    def save_result(self) -> int:
        """Save measured traces and parameters to a binary file.

        The filename is generated automatically using a short descriptive
        prefix, the current system date, and an auto-incrementing counter
        to prevent overwriting existing measurements.

        Returns
        -------
        int
            Status code of the file save operation.
             0 : File saved successfully and GUI components updated.
            -1 : File save failed due to an I/O or system error.

        Side Effects
        ------------
        - Creates a binary `.trc` file inside the 'results' directory.
        - Updates the `self.statusBar` message with progress and results.
        - Synchronizes GUI elements: updates the File counter value,
          and writes the filename and full fill path on the GUI.
        """
        self.statusBar.showMessage('Saving results ...')

        try:
            resultfile = us.find_filename(prefix='us',
                                          ext='trc',
                                          resultdir='results')

            self.wfm.save(resultfile.path)

            self.filecounterSpinBox.setValue(resultfile.counter)
            self.resultfileEdit.setText(resultfile.name)
            self.resultpathEdit.setText(resultfile.path)

            self.statusBar.showMessage(f'Result saved to {resultfile.name}')
            return 0

        except (OSError, IOError) as e:
            # Fanger opp fulle disker, manglende mapper eller skriveforbud
            self.statusBar.showMessage(f'Error saving file: {e}')
            self.resultfileEdit.setText('ERROR')
            self.resultpathEdit.setText(str(e))
            return -1

    def _update_ui_element(self, element, message: str, color: tuple) -> str:
        """Helper to set text and stylesheet colors for a QWidget element.

        Parameters
        ----------
        element : QWidget
            The Qt UI widget to update
        message : str
            The text message to display inside the widget.
        color : tuple of str
            A tuple or list containing exactly two color strings, one for
            text and one for background.

        Returns
        -------
        str
            The message string that was applied to the UI element.

        """
        text_color, bg_color = color
        element.setText(message)
        element.setStyleSheet(
            f'color: {text_color}; background-color: {bg_color}')
        return message

    def update_status_box(self, acquiring: bool = False) -> str:
        """Write the system acquisition state to the status box.

        Parameters
        ----------
        acquiring : bool, default False
            Flag indicating if the oscilloscope is actively capturing data.

        Returns
        -------
        str
            The status message string applied ('Acquiring' or 'Stopped').
        """
        if not acquiring:
            message, color = 'Stopped', COLOR['warning']
        else:
            message, color = 'Acquiring', COLOR['ok']

        return self._update_ui_element(self.statusEdit, message, color)

    def update_connected_box(self, connected: bool = False) -> str:
        """Write the connection state of the hardware instrument to the GUI.

        Parameters
        ----------
        connected : bool, default False
            Flag indicating if a valid connection to the PicoScope is open.

        Returns
        -------
        str
            The connection message string ('Connected' or 'Not connected').
        """
        if not connected:
            message, color = 'Not connected', COLOR['warning']
        else:
            message, color = 'Connected', COLOR['ok']

        return self._update_ui_element(self.connectedEdit, message, color)

    def update_transmit_box(self,
                            available: bool = False,
                            on: bool = False) -> str:
        """Write operational state of the arbitrary waveform generator (AWG).

        Parameters
        ----------
        available : bool, default False
            Flag indicating if the connected PicoScope model features an AWG.
        on : bool, default False
            Flag indicating if the pulse generator is actively transmitting.

        Returns
        -------
        str
            The transmit message string applied ('Not available',
            'Transmitting', 'Off').
        """
        if not available:
            message, color = 'Not available', COLOR['warning']
        elif on:
            message, color = 'Transmitting', COLOR['ok']
        else:
            message, color = 'Off', COLOR['warning']

        return self._update_ui_element(self.transmitStatusEdit, message, color)

    def update_status(self, message: str, append: bool = False) -> str:
        """Update the status log field at the bottom of the window.

        Parameters
        ----------
        message : str
            The text to write or append to the status log.
        append : bool, default False
            If True, the new message is appended to the end of the existing
            text. If False, the field is overwritten with the new message.

        Returns
        -------
        str
            The text string that was written to the text edit widget.
        """
        if append:
            message = f'{self.status_textEdit.toPlainText()}{message}'

        self.status_textEdit.setText(message)
        return message

    def find_voltagescale(self, vmax: float) -> tuple[float, str]:
        """Find the appropriate scaling factor and unit for the voltage axis.

        Find maximum voltage value, determine whether it is best represented
        in microvolts, millivolts, or Volts.

        Parameters
        ----------
        vmax : float
            The peak or maximum voltage value in Volts.

        Returns
        -------
        voltage_scale : float
            Scaling factor to convert Volts into the target unit.
            Divide raw Volt value by this factor (raw_v / voltage_scale).
        unit : str
            Short representation of the voltage unit ('uV', 'mV', or 'V').
        """
        vmax_abs = abs(vmax)

        if vmax_abs < 1e-3:
            voltage_scale, unit = 1e-6, 'uV'
        elif vmax_abs < 1:
            voltage_scale, unit = 1e-3, 'mV'
        else:
            voltage_scale, unit = 1.0, 'V'

        return voltage_scale, unit

    def update_display(self, time_unit: str = 'us'):
        """Update values, graph limits, colors, and markers on the screen.

        This method synchronizes the Matplotlib axes limits (time, voltage,
        and frequency) with the current values in the GUI spinboxes. It also
        updates the style and background colors of the channel toggle buttons
        to visually reflect whether a channel is active, before forcing a
        canvas redraw.

        Parameters
        ----------
        time_unit : str, default 'us'
            Unit to use on the time axis. Must be one of {'s', 'ms', 'us'}.

        """
        scale_map = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6}
        if time_unit not in scale_map:
            raise ValueError(f"Invalid time_unit '{time_unit}'")

        current_timescale = scale_map[time_unit]

        # Full trace
        t_lim = np.array([self.sampling.start_time,
                         self.sampling.end_time]) / current_timescale
        self.axis['trace'][0].set_xlim(t_lim)

        # Selected interval, 'zoom'
        t_lim = us.find_limits([self.zoomStartSpinBox.value(),
                                self.zoomEndSpinBox.value()],
                               min_diff=0.1)
        self.display.t_lim = t_lim * current_timescale
        self.axis['zoom'][0].set_xlim(t_lim)

        self.graph['zoom_area'].set_x(t_lim[0])
        self.graph['zoom_area'].set_width(t_lim[1] - t_lim[0])

        # Vertical scale
        db_lim = us.find_limits([self.dbMinSpinBox.value(),
                                 self.dbMaxSpinBox.value()])

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

            bg_color = COLOR['channel'][k] if is_on else COLOR['off']
            text_color = "white" if is_on else "#808080"
            btn.setStyleSheet(
                f"color: {text_color}; background-color: {bg_color};")
            label.setStyleSheet(
                f"color: {text_color}; background-color: {bg_color};")

            vzoom = us.read_scaled_value(vrange.currentText())
            ax_zoom.set_ylim(-vzoom, vzoom)

            v_max = ch.v_max
            ax_trace.set_ylim(-v_max, v_max)

            ax_spectrum.set_ylim(db_lim)

        # Frequency axis
        f_lim = us.find_limits([self.zoomFminSpinBox.value(),
                                self.zoomFmaxSpinBox.value()],
                               min_diff=0.1)

        self.axis['spectrum'][0].set_xlim(*f_lim)
        self.axis['awgspec'].set_xlim(*f_lim)
        self.axis['awgspec'].set_ylim(db_lim)

        self.fig.canvas.draw()
        return

    def connect_gui(self):
        """Connect GUI signals to slot functions, initialize channels.

        Binds all interactive PyQt/PySide widgets (SpinBoxes, ComboBoxes,
        and Buttons) to the appropriate hardware control and display update
        functions.
        Structures multi-channel elements into iterable formats and applies
        the initial stylesheet coloring.
        """
        # Display scales
        for spin_box in (self.zoomStartSpinBox, self.zoomEndSpinBox,
                         self.zoomFminSpinBox, self.zoomFmaxSpinBox,
                         self.dbMinSpinBox, self.dbMaxSpinBox):
            spin_box.valueChanged.connect(lambda: self.update_display())

        # RF filter
        self.filterComboBox.activated.connect(lambda: self.update_rf_filter())
        for spin_box in (self.fminSpinBox,
                         self.fmaxSpinBox,
                         self.filterOrderSpinBox):
            spin_box.valueChanged.connect(lambda: self.update_rf_filter())

        # Trigger
        for combo_box in (self.triggerSourceComboBox,
                          self.triggerModeComboBox):
            combo_box.activated.connect(lambda: self.update_trigger())

        for spin_box in (self.triggerPositionSpinBox,
                         self.triggerLevelSpinBox,
                         self.triggerDelaySpinBox,
                         self.triggerAutoDelaySpinBox):
            spin_box.valueChanged.connect(lambda: self.update_trigger())

        # Horizontal (Sampling)
        self.samplerateSpinBox.valueChanged.connect(
            lambda: self.update_sampling())
        self.nSamplesSpinBox.valueChanged.connect(
            lambda: self.update_sampling())

        # Pulse generator (AWG)
        self.transmitButton.clicked.connect(lambda: self.update_pulser())
        for combo_box in (self.pulseEnvelopeComboBox, self.pulseShapeComboBox):
            combo_box.activated.connect(lambda: self.update_pulser())

        for spin_box in (self.pulseFrequencySpinBox, self.pulseDurationSpinBox,
                         self.pulsePhaseSpinBox, self.pulseAmplitudeSpinBox):
            spin_box.valueChanged.connect(lambda: self.update_pulser())

        # Program flow
        self.connectButton.clicked.connect(self.connect_dso)
        self.acquireButton.clicked.connect(self.control_acquisition)
        self.saveButton.clicked.connect(self.save_result)
        self.closeButton.clicked.connect(self.close_connection)

        # Vertical channels
        self.chButton = [self.chAButton, self.chBButton]
        self.chLabel = [self.chALabel, self.chBLabel]
        self.displayrangeComboBox = [
            self.displayrangeAComboBox, self.displayrangeBComboBox]
        self.rangeComboBox = [self.rangeAComboBox, self.rangeBComboBox]
        self.couplingComboBox = [
            self.couplingAComboBox, self.couplingBComboBox]
        self.offsetSpinBox = [self.offsetASpinBox, self.offsetBSpinBox]
        self.bwlComboBox = [self.bwlAComboBox, self.bwlBComboBox]

        channels = [
            {
                "btn": self.chAButton,
                "lbl": self.chALabel,
                "disp": self.displayrangeAComboBox,
                "acq": self.rangeAComboBox,
                "cpl": self.couplingAComboBox,
                "offset": self.offsetASpinBox,
                "bwl": self.bwlAComboBox,
                "color": COLOR['channel'][0]
            },
            {
                "btn": self.chBButton,
                "lbl": self.chBLabel,
                "disp": self.displayrangeBComboBox,
                "acq": self.rangeBComboBox,
                "cpl": self.couplingBComboBox,
                "offset": self.offsetBSpinBox,
                "bwl": self.bwlBComboBox,
                "color": COLOR['channel'][1]
            }
        ]

        for ch in channels:
            ch["btn"].clicked.connect(lambda: self.update_display())
            ch["disp"].activated.connect(lambda: self.update_display())

            ch["acq"].activated.connect(lambda: self.update_vertical())
            ch["cpl"].activated.connect(lambda: self.update_vertical())
            ch["offset"].valueChanged.connect(lambda: self.update_vertical())
            ch["bwl"].activated.connect(lambda: self.update_vertical())

            style = f"color: white; background-color: {ch['color']};"
            ch["btn"].setStyleSheet(style)
            ch["lbl"].setStyleSheet(style)
            ch["disp"].setStyleSheet(style)

        self.update_display()
        return

    def define_graphs(self) -> tuple[dict, dict, dict]:
        """Initialize result graphs, layout, titles, scales, and colors.

        Sets up plot layout, configures dual y-axes (twinx) for
        multi-channel RF data, applies background theme coloring to
        distinguish between raw traces, zoomed intervals, and pulse generator
        responses, and creates empty line objects for high-speed updates.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle to the created result figure.
        axis : dict of list of matplotlib.axes.Axes
            Dictionary where keys are plot names ('trace', 'zoom', etc.)
            and values are lists of axes handles (index 0 for ChA, 1 for ChB).
        graph : dict
            Dictionary containing the fast-update line objects (`Line2D`)
            and visual indicators, e.g. `axvspan` zoom area.
        """
        # Figure layout
        axgrid = [['trace'] * 3,
                  ['awg'] + ['zoom'] * 2,
                  ['awgspec'] + ['spectrum'] * 2]

        fig, axis = matplotlib.pyplot.subplot_mosaic(axgrid,
                                                     figsize=(9, 6),
                                                     layout='constrained')

        axis['trace'].set_title('Acquired traces', loc='left')
        axis['zoom'].set_title('Selected interval', loc='left')
        axis['awg'].set_title('Pulser', loc='left')

        # Configure time graphs
        for key in ['trace', 'zoom', 'awg']:
            axis[key].set_xlabel(r'Time [$\mu$s]')
            axis[key].set_ylabel('Voltage [V]')
            axis[key].set_xlim(0, 1)
            axis[key].grid(True)

        # Configure frequency graphs
        for key in ['spectrum', 'awgspec']:
            axis[key].set_xlabel('Frequency [MHz]')
            axis[key].set_ylabel('Power [dB re. max]')
            axis[key].set_xlim(0, 1)
            axis[key].grid(True)

        # Colour backgrounds for identification
        for key in ['zoom', 'spectrum']:
            axis[key].set_facecolor(COLOR['zoom'][1])
        for key in ['awg', 'awgspec']:
            axis[key].set_facecolor(COLOR['awg'][1])

        # Create dual y-axes and apply channel colours
        for key in ['trace', 'zoom', 'spectrum']:
            axis[key] = [axis[key], axis[key].twinx()]

        for ch_idx, ch_name in enumerate(ps.CHANNEL_NAMES):
            color = COLOR['channel'][ch_idx]

            # Time domain
            for g in ['trace', 'zoom']:
                ax = axis[g][ch_idx]
                ax.set_ylabel('Voltage [V]', color=color)
                ax.tick_params(axis='y', colors=color)
                if ch_idx > 0:
                    ax.set_xlabel('')

            # Frequency domain
            ax_spec = axis['spectrum'][ch_idx]
            ax_spec.set_ylabel('Power [dB re. max]', color=color)
            ax_spec.tick_params(axis='y', colors=color)
            if ch_idx > 0:
                ax_spec.set_xlabel('')

        # Create empty graphs to be filled with results, for quick updates
        graph = {}
        for key in ['trace', 'zoom', 'spectrum']:
            graph[key] = [ax.plot([], [], color=color)[0]
                          for ax, color in zip(axis[key], COLOR['channel'])]

        graph['zoom_area'] = axis['trace'][0].axvspan(0, 0,
                                                      color=COLOR['zoom'][0])
        graph['awg'], = axis['awg'].plot([], [],
                                         color=COLOR['awg'][0])
        graph['awgspec'], = axis['awgspec'].plot([], [],
                                                 color=COLOR['awg'][0])

        fig.show()
        return fig, axis, graph


if __name__ == '__main__':
    """Main entry point for the ReadUltrasound application.

    Initializes the QApplication instance, ensures it handles pre-existing
    instances correctly, applies cross-platform 'Fusion' visual theme,
    and manages clean destruction of the window upon exit.

    Side Effects
    ------------
    - Spawns a graphical user interface thread via `QtWidgets.QApplication`.
    - Modifies the global application style theme to 'Fusion'.
    - Instantiates and displays the `ReadUltrasound` main window.
    """
    # Check for running Qt-applications
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    app.setStyle('Fusion')

    # Run Qt application, with clean exit
    try:
        window = ReadUltrasound()
        window.show()
        sys.exit(app.exec())

    except Exception as e:
        print(f"Application crashed at start, : {e}")
        sys.exit(1)
