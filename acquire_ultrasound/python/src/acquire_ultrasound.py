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

import sys
import matplotlib

import ultrasound_utilities as us         # USN ultrasound lab specific
import ps5000a_ultrasound_wrappers as ps  # Interface to Pico c-library

# Constants
COLOR_WARNING = ['#78281F', '#FADBD8']
COLOR_OK = ['#145A32', '#D4EFDF']
COLOR_NEUTRAL = ['#000000', '#FFFFFF']
COLOR_OFF = '#708090'
COLOR_CH = ['#004B93', '#D32F2F', '#388E3C', '#FBC02D']
COLOR_AWG = '#388E3C'
COLOR_ZOOM = '#B0E0E6'
COLOR_AWG_BACKGROUND = '#F5FFFA'
COLOR_ZOOM_BACKGROUND = '#E0FFFF'

TIMESCALE = 1E-6      # Display scales for time and frequency
FREQUENCYSCALE = 1E6
V_MAX = 20           # Absolute maximum voltage scale

matplotlib.use('QtAgg')
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
    """Main ultrasound acquisition GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.setupUi(self)
        self._position_window()
        self._initialize_components()
        self._initialize_graphs()
        self._configure_gui()

    def _position_window(self) -> None:
        """Place GUI window near top-left corner of the screen."""
        geometry = QApplication.primaryScreen().geometry()
        self.move(int(0.02 * geometry.width()),
                  int(0.05 * geometry.height()),
                  )

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

    def _initialize_graphs(self) -> None:
        """Create result graphs."""
        self.fig, self.axis, self.graph = self.define_graphs()

    def _configure_gui(self) -> None:
        """Connect signals and initialize widget states."""
        self.connect_gui()
        self.update_connected_box(False)

        for button in (self.acquireButton,
                       self.transmitButton,
                       self.saveButton,
                       ):
            button.setEnabled(False)

    def connect_dso(self) -> int:
        """Connect, configure, and start the digital storage oscilloscope (DSO).

        This method attempts to clean up any existing active connections, 
        opens a new session with the instrument, and applies the initial 
        configuration (vertical settings, trigger, sampling, pulser, 
        RF filter, and display). It also updates the GUI buttons and status 
        bars based on the connection success state.

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

        # Try to close old handle if resident. May not work
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
            # Check for signal generator
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

        Attempts to gracefully close the connection to the digitizer/DSO.
        If the instrument is not connected or fails to respond, the function
        catches the error to prevent the user interface from crashing.

        Returns
        -------
        status : str
            The connection status of the DSO
        errorcode : int
            Returns 0 upon successful closure, or -1 if an error occurred
            (e.g., instrument not connected).
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
        """Read vertical settings from the GUI and apply them to the instrument.

        This method synchronizes the software channel configurations with the 
        current states of the GUI widgets (range, coupling, offset, and 
        bandwidth limits) for all available channels. If the instrument is 
        connected, these settings are transmitted to the hardware.

        Side Effects
        ------------
        - Enables both `self.channel[0]` and `self.channel[1]` unconditionally.
        - Modifies properties (`v_range`, `coupling`, `offset`, `bwl`) on all 
          channel objects within `self.channel`.
        - Transmits configurations to `self.dso` if a hardware connection is active.

        Notes
        -----
        Both input traces are hardcoded to be acquired always, regardless of 
        individual GUI enable/disable states.
        """
        self.channel[0].enabled = True  # Both traces are always aquired
        self.channel[1].enabled = True

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
                self.dso.set_vertical(channel)
                self.dso.set_bwl(channel)

        return

    def update_trigger(self):
        """Read trigger settings from the GUI and send them to the instrument.

        This method extracts the current user inputs from the trigger-related 
        GUI controls, scales the time-based values appropriately, updates 
        the internal trigger and sampling states, and applies the 
        configuration to the oscilloscope.


        Side Effects
        ------------
        - Updates properties (`source`, `direction`, `level`, `delay`, `autodelay`) 
          on the `self.trigger` object.
        - Updates the `trigger_position` property on the `self.sampling` object.
        - Transmits the new trigger configuration to `self.dso` using the active 
          channels and sampling settings, provided a hardware connection exists.

        Notes
        -----
        - The trigger delay is automatically scaled using the global 
          `TIMESCALE` constant.
        """
        self.trigger.source = self.triggerSourceComboBox.currentText()
        self.trigger.direction = self.triggerModeComboBox.currentText()
        self.trigger.level = self.triggerLevelSpinBox.value()
        self.trigger.delay = self.triggerDelaySpinBox.value()*TIMESCALE
        self.trigger.autodelay = self.triggerAutoDelaySpinBox.value()*1e-3
        self.sampling.trigger_position = self.triggerPositionSpinBox.value()

        if self.dso.connected:
            self.dso.set_trigger(self.trigger, self.channel, self.sampling)

        return

    def update_sampling(self):
        """Read sampling settings from the GUI and configure the timebase.

        Retrieves requested sample rate and number of samples from GUI, 
        queries the instrument hardware to find the closest matching 
        hardware timebase, and updates internal sampling parameters. If the 
        ocilloscope is connected, it verifies the actual hardware sampling 
        interval and updates the GUI value to display the actual 
        hardware-supported sample rate.

        Side Effects
        ------------
        - Modifies properties (`timebase`, `dt`, `n_samples`) on the 
          `self.sampling` object.
        - Updates `self.samplerateSpinBox` to reflect the actual, 
          hardware-limited sampling frequency.
        - Sets the boolean flag `self.runstate.sampling_changed` to True.
        - Calls methods on `self.dso` to query timebase and sample interval.

        Notes
        -----
        - The requested sample rate is scaled using the global `FREQUENCYSCALE` 
          constant before being passed to the DSO driver.
        """
        fs_requested = int(self.samplerateSpinBox.value()*FREQUENCYSCALE)
        self.sampling.timebase, fs_actual = self.dso.find_timebase(
            fs_requested)
        self.sampling.n_samples = int(self.nSamplesSpinBox.value()*1e3)

        self.sampling.dt = 1/fs_actual
        if self.dso.connected:
            self.sampling.dt = self.dso.get_sample_interval(self.sampling)

        self.samplerateSpinBox.setValue(self.sampling.fs()/FREQUENCYSCALE)
        self.runstate.sampling_changed = True
        return

    def update_pulser(self):
        """Read pulse settings from the GUI, update plots, and program the AWG.

        This method reads arbitrary waveform generator (AWG) parameters from 
        the GUI (envelope, shape, frequency, duration, phase, and amplitude) 
        and transfers them to the AWG in the oscilloscope. It computes and 
        updates both the time-domain pulse signal graph and its corresponding 
        power spectrum graph. If an AWG is supported by the connected hardware, 
        the waveform data is transmitted to the instrument; otherwise, the 
        execution aborts early.

        Returns
        -------
        int
            Status code of the pulser update attempt.
            -1 : Aborted because the hardware does not support a signal generator.
             0 : Successfully processed settings, updated plots, and sent data.

        Side Effects
        ------------
        - Modifies properties (`on`, `envelope`, `shape`, `f0`, `n_cycles`,
          `phase`, `a`) on the `self.pulse` object.
        - Updates data and axis limits for the `self.graph['awg']` plot.
        - Computes the pulse power spectrum and updates `self.graph['awgspec']`.
        - Alters the line style ('solid' vs 'dotted') based on whether
          the pulser output is currently toggled active.
        - Updates the GUI transmit panel state.

        """
        if not self.dso.signal_generator:
            self.update_transmit_box(available=False)
            return -1    # Does nothing signal genarator not available

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
        """Read RF noise filter settings from the GUI and update filter state.

        Retrieves the requested filter type, cutoff frequencies, and 
        filter order from the GUI widgets. Reads the sampling rate
        from the system to scale the filter.

        Side Effects
        ------------
        - Modifies properties (`sample_rate`, `type`, `f_min`, `f_max`, 
          `order`) on the `self.rf_filter` object.
        """
        self.rf_filter.sample_rate = self.sampling.fs()
        self.rf_filter.type = self.filterComboBox.currentText()
        self.rf_filter.f_min = self.fminSpinBox.value()*FREQUENCYSCALE
        self.rf_filter.f_max = self.fmaxSpinBox.value()*FREQUENCYSCALE
        self.rf_filter.order = self.filterOrderSpinBox.value()
        return

    def control_acquisition(self):
        """Control data acquisition from oscilloscope.

        This method checks the state of the acquisition button. If the 
        button is checked (active), it initiates a new trace acquisition loop. 
        If the button is unchecked (inactive), it requests the instrument to 
        halt any ongoing data collection.

        Side Effects
        ------------
        - Invokes either `self.acquire_trace` or `self.stop_acquisition` 
          depending on the boolean state of `self.acquireButton`.

        """
        if self.acquireButton.isChecked():
            self.acquire_trace()
        else:
            self.stop_acquisition()
        return

    def acquire_trace(self):
        """Acquire waveforms from the instrument in a continuous measurement loop.

        Locks the acquisition state and enters a polling loop that repeatedly 
        captures data traces from the hardware as long as the shutdown flag 
        (`stop_acquisition`) is false. Forces the hardware to reconfigure if 
        sampling parameters were modified prior to or during the loop.
        Each captured trace updates the internal waveform memory and triggers 
        a redraw of the plots.


        Side Effects
        ------------
        - Disables the `Close` and enables the `Save` during execution.
        - Toggles state flags on `self.runstate` (`oscilloscope_ready`, `sampling_changed`).
        - Continuous updates to `self.wfm` properties (`y`, `dt`, `t0`) with new hardware data.
        - Frequently updates the `self.statusBar` text and triggers GUI redraws via `self.plot_result`.
        - Restores button states and sets the status bar back to 'Ready' once the loop terminates.

        Notes
        -----
        .. warning::
           This method contains a blocking `while` loop. If this class runs on the 
           main GUI thread (e.g., PyQt/PySide main thread), running this loop will 
           freeze the user interface unless `QCoreApplication.processEvents()` is 
           called inside the loop, or this entire method is executed within a 
           separate `QThread`.
        """
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
                    self.dso.configure_acquisition(self.sampling)
                    self.runstate.sampling_changed = False
                self.wfm.y = self.dso.acquire_trace(self.sampling,
                                                    self.channel)
                self.wfm.dt = self.sampling.dt
                self.wfm.t0 = self.sampling.t0()
                self.plot_result()
        self.update_status_box(False)
        self.statusBar.showMessage('Ready')
        self.runstate.stop_acquisition = False
        return

    def stop_acquisition(self):
        """Stop the continuous waveform acquisition loop safely.

        Flags active acquisition loop to terminate on its next iteration 
        without severing the physical hardware connection to the
        instrument. Updates user interface status indicators and restores
        interactive states of the GUI buttons (e.g., re-enabling the close
        button and disabling the save button).

        Side Effects
        ------------
        - Sets the `self.runstate.stop_acquisition` boolean flag to True.
        - Resets the `self.runstate.oscilloscope_ready` boolean flag to True.
        - Updates the `self.statusBar` text to 'Stopping' if a loop was running.
        - Re-enables `self.closeButton` and disables `self.saveButton`.
        - Updates the visual status indicators via `self.update_status_box`.
        """
        if not (self.runstate.stop_acquisition):
            self.statusBar.showMessage('Stopping')
            self.update_status_box(False)
        self.runstate.stop_acquisition = True
        self.runstate.oscilloscope_ready = True
        self.closeButton.setEnabled(True)
        self.saveButton.setEnabled(False)
        return

    def plot_result(self, time_unit: str = 'us'):
        """Process and plot the measured waveform trace on the screen.

        Applies the RF noise filter to the current waveform, crops the signal 
        to the display time limits, and calculates its power spectrum 
        in decibels (dB). It then updates the data of existing 
        Matplotlib line objects for each active channel across three plots: 
        raw trace, zoomed trace, and spectrum. Inactive channels are cleared 
        from the display.

        Parameters
        ----------
        time_unit : str, default 'us'
            The physical unit to use for the time axis. Must be one of 
            {'s', 'ms', 'us'}.

        Raises
        ------
        ValueError
            If an unsupported `time_unit` string is provided.

        Side Effects
        ------------
        - Extracts and transforms data from `self.wfm` via filtering and zooming.
        - Modifies the data arrays of Matplotlib line objects inside `self.graph` 
          by calling `set_data()`.
        - Empties lines for channels that are disabled in `self.display.channel`.
        - Forces Matplotlib to process pending events via `self.fig.canvas.flush_events()`.
        - Triggers an interface redraw by invoking `self.update_display()`.

        Notes
        -----
        - The power spectrum calculation uses a normalized dB scale.
        - Updating plot data using `line.set_data()` is significantly faster 
          than clearing and redrawing the entire axes, which is critical for 
          maintaining a smooth frame rate during continuous acquisition.
        - *Developer note:* If the plots fail to visually refresh on screen, 
          uncommenting `self.fig.canvas.draw()` right before `flush_events()` 
          is usually required in Matplotlib interactive modes.
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

        Side Effects
        ------------
        - Modifies the text and style sheet properties of the passed `element`.
        """
        text_color, bg_color = color
        element.setText(message)
        element.setStyleSheet(
            f'color: {text_color}; background-color: {bg_color}')
        return message

    def update_status_box(self, acquiring: bool = False) -> str:
        """Write the system acquisition state to the status box with visual colors.

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
            message, color = 'Stopped', COLOR_WARNING
        else:
            message, color = 'Acquiring', COLOR_OK

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
            The connection message string applied ('Connected' or 'Not connected').
        """
        if not connected:
            message, color = 'Not connected', COLOR_WARNING
        else:
            message, color = 'Connected', COLOR_OK

        return self._update_ui_element(self.connectedEdit, message, color)

    def update_transmit_box(self, available: bool = False, on: bool = False) -> str:
        """Write the operational state of the arbitrary waveform generator (AWG).

        Parameters
        ----------
        available : bool, default False
            Flag indicating if the connected PicoScope model features an AWG.
        on : bool, default False
            Flag indicating if the pulse generator is actively transmitting.

        Returns
        -------
        str
            The transmit message string applied ('Not available', 'Transmitting', 
            or 'Off').
        """
        if not available:
            message, color = 'Not available', COLOR_WARNING
        elif on:
            message, color = 'Transmitting', COLOR_OK
        else:
            message, color = 'Off', COLOR_WARNING

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

        Finds the maximum voltage value and determines whether the data is 
        best represented in microvolts (uV),         millivolts (mV), or Volts (V). 

        Parameters
        ----------
        vmax : float
            The peak or maximum voltage value in Volts.

        Returns
        -------
        voltage_scale : float
            The scaling factor to convert Volts into the target unit. 
            Divide the raw Volt value by this factor (e.g., raw_v / voltage_scale).
        unit : str
            The short string representation of the voltage unit ('uV', 'mV', or 'V').
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

        Raises
        ------
        ValueError
            If an unsupported `time_unit` string is provided.

        Side Effects
        ------------
        - Modifies limits (`xlim`, `ylim`) across all Matplotlib axes dictionaries.
        - Updates text and background stylesheets for `self.chButton` and `self.chLabel`.
        - Updates the `self.display.t_lim` storage object (scaled back to base units).
        - Redraws the Matplotlib figure canvas via `self.fig.canvas.draw()`.
        """
        scale_map = {'s': 1.0, 'ms': 1e-3, 'us': 1e-6}
        if time_unit not in scale_map:
            raise ValueError(f"Invalid time_unit '{time_unit}'. Expected one of {
                             set(scale_map.keys())}")

        current_timescale = scale_map[time_unit]

        # Full trace
        t0_scaled = self.sampling.t0() / current_timescale
        tmax_scaled = self.sampling.t_max() / current_timescale
        self.axis['trace'][0].set_xlim(t0_scaled, tmax_scaled)

        # 2. Selected interval, 'zoom'
        zoom_range = [self.zoomStartSpinBox.value(),
                      self.zoomEndSpinBox.value()]
        t_lim = us.find_limits(zoom_range, min_diff=0.1)
        self.display.t_lim = t_lim * current_timescale

        self.graph['zoom_area'].set_x(t_lim[0])
        self.graph['zoom_area'].set_width(t_lim[1] - t_lim[0])
        # Pakker ut verdiene for Matplotlib
        self.axis['zoom'][0].set_xlim(*t_lim)

        # Vertical scale
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
            text_color = "white" if is_on else "#808080"
            btn.setStyleSheet(
                f"color: {text_color}; background-color: {bg_color};")
            label.setStyleSheet(
                f"color: {text_color}; background-color: {bg_color};")

            vzoom = us.read_scaled_value(vrange.currentText())
            ax_zoom.set_ylim(-vzoom, vzoom)

            v_max = ch.v_max()
            ax_trace.set_ylim(-v_max, v_max)

            ax_spectrum.set_ylim(db_lim)

        # Frequency axis
        f_range = [self.zoomFminSpinBox.value(), self.zoomFmaxSpinBox.value()]
        f_lim = us.find_limits(f_range, min_diff=0.1)

        self.axis['spectrum'][0].set_xlim(*f_lim)
        self.axis['awgspec'].set_ylim(db_lim)
        self.axis['awgspec'].set_xlim(*f_lim)

        self.fig.canvas.draw()
        return

    def connect_gui(self):
        """Connect GUI signals to their respective slot functions and initialize channels.

        Binds all interactive PyQt/PySide widgets (SpinBoxes, ComboBoxes, 
        and Buttons) to the appropriate hardware control and display update 
        functions. 
        Structures the multi-channel elements into iterable formats and applies 
        the initial stylesheet coloring.

        Side Effects
        ------------
        - Binds signals (`valueChanged`, `activated`, `clicked`) for over 20 GUI widgets.
        - Creates instance lists: `self.chButton`, `self.chLabel`, `self.displayrangeComboBox`, 
          `self.rangeComboBox`, `self.couplingComboBox`, `self.offsetSpinBox`, and `self.bwlComboBox`.
        - Modifies stylesheets for channel buttons, labels, and dropdowns.
        """
        # Display scales
        for spin_box in (self.zoomStartSpinBox, self.zoomEndSpinBox,
                         self.zoomFminSpinBox, self.zoomFmaxSpinBox,
                         self.dbMinSpinBox, self.dbMaxSpinBox):
            spin_box.valueChanged.connect(lambda: self.update_display())

        # RF filter
        self.filterComboBox.activated.connect(lambda: self.update_rf_filter())
        for spin_box in (self.fminSpinBox, self.fmaxSpinBox, self.filterOrderSpinBox):
            spin_box.valueChanged.connect(lambda: self.update_rf_filter())

        # Trigger
        for combo_box in (self.triggerSourceComboBox, self.triggerModeComboBox):
            combo_box.activated.connect(lambda: self.update_trigger())

        for spin_box in (self.triggerPositionSpinBox, self.triggerLevelSpinBox,
                         self.triggerDelaySpinBox, self.triggerAutoDelaySpinBox):
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
                "color": COLOR_CH[0]
            },
            {
                "btn": self.chBButton,
                "lbl": self.chBLabel,
                "disp": self.displayrangeBComboBox,
                "acq": self.rangeBComboBox,
                "cpl": self.couplingBComboBox,
                "offset": self.offsetBSpinBox,
                "bwl": self.bwlBComboBox,
                "color": COLOR_CH[1]
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

        Sets up a multi-plot layout, configures dual y-axes (twinx) for 
        multi-channel RF data, applies background theme coloring to 
        distinguish between raw traces, zoomed intervals, and pulse generator 
        responses, and instantiates empty line objects for high-speed updates.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Handle to the created Matplotlib result figure.
        axis : dict of list of matplotlib.axes.Axes
            Dictionary where keys are plot names ('trace', 'zoom', etc.) 
            and values are lists of axes handles (index 0 for ChA, 1 for ChB).
        graph : dict
            Dictionary containing the fast-update line objects (`Line2D`) 
            and visual indicators (like the `axvspan` zoom area).
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
            axis[key].set_facecolor(COLOR_ZOOM_BACKGROUND)
        for key in ['awg', 'awgspec']:
            axis[key].set_facecolor(COLOR_AWG_BACKGROUND)

        # Create dual y-axes and apply channel colours
        for key in ['trace', 'zoom', 'spectrum']:
            axis[key] = [axis[key], axis[key].twinx()]

        for ch_idx, ch_name in enumerate(ps.CH_NAMES):
            color = COLOR_CH[ch_idx]

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
                          for ax, color in zip(axis[key], COLOR_CH)]

        graph['zoom_area'] = axis['trace'][0].axvspan(0, 0, color=COLOR_ZOOM)
        graph['awg'], = axis['awg'].plot([], [], color=COLOR_AWG)
        graph['awgspec'], = axis['awgspec'].plot([], [], color=COLOR_AWG)

        fig.show()
        return fig, axis, graph


if __name__ == '__main__':
    """Main entry point for the ReadUltrasound application.

    Initializes the QApplication instance, ensures it handles 
    pre-existing instances correctly (critical when running inside IDEs ), 
    applies the cross-platform 'Fusion' visual theme, and manages clean 
    destruction of the window upon exit.

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
