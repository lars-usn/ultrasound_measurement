"""Wrappers to c-style function calls in DLLs from Picoscope SDK.

Wraps c-style commands (ctypes.xx) to standard Python variables and sets
scaling constants, ranges etc. specific for the instrument.
The classes and functions in this file shall provide an easy interface to
Picoscope 5000a from any standard Pyton environment

Based on example program from Picotech
  PS5000A BLOCK MODE EXAMPLE, # Copyright (C) 2018-2022 Pico Technology Ltd.

Reference
  PicoScope 5000 Series (A API) - Programmer's Guide. Pico Tecknology Ltd, 2018

Lars Hoff, USN, Sep 2022
    Modified June 2024 to better follow PEP-8 and numpy docstring style guides.
    Rearranged and cleaned, July 2026
"""
import time
import ctypes
import numpy as np
import bisect

from picosdk.ps5000a import ps5000a as picoscope
from picosdk.functions import adc2mV, assert_pico_ok
from ultrasound_utilities import Pulse

DAC_SAMPLERATE = 500e6   # [Samples/s] Fixed, see Programmer's guide
DAC_MAX_AMPLITUDE = 2.0   # [v] Max. amplitude from signal generator
CH_NAMES = ["A", "B"]


class Channel:
    """Oscilloscope vertical (voltage) channel settings and status.

    Attributes
    ----------
    no : int
        Channel number (0='A', 1='B', etc.).
    enabled : bool
        True if the channel is enabled, False otherwise.
    v_range : float
        Requested full-scale voltage range, single-sided.
    adc_max : int
        Maximum ADC value used for scaling to voltage.
    offset : float
        Offset voltage.
    coupling : str
        Channel coupling, "DC" or "AC".
    bwl : bool
        Bandwidth limiter status (not available on PS2000 series).
    """

    def __init__(self, no: int) -> None:
        """Initialise the channel with default values.

        Parameters
        ----------
        no : int
            The numerical channel identifier (0, 1, 2, ...).
        """
        self.no: int = no
        self.enabled: bool = True
        self.v_range: float = 1.0
        self.adc_max: int = 32767  # PicoScope 5000a, 12 to 16-bit resolution
        self.offset: float = 0.0
        self.coupling: str = "DC"
        self.bwl: bool = False

    def name(self) -> str:
        """Return the PicoScope channel name (A, B, ...) based on its number.

        Returns
        -------
        str
            The letter name of the channel.
        """
        return channel_no_to_name(self.no)

    def v_max(self) -> float:
        """Find the allowed voltage range from the requested range."""
        v = self.v_range

        valid_ranges = [
            0.01, 0.02, 0.05,
            0.1, 0.2, 0.5,
            1.0, 2.0, 5.0,
            10.0, 20.0, 50.0
        ]

        if v <= valid_ranges[0]:
            return valid_ranges[0]
        if v >= valid_ranges[-1]:
            return valid_ranges[-1]

        idx = bisect.bisect_left(valid_ranges, v)
        return valid_ranges[idx]


class Horizontal:
    """Oscilloscope horizontal (time) scale configurations.

    Attributes
    ----------
    timebase : int
        Number defining the oscilloscope sample rate configuration.
    n_samples : int
        Number of samples to acquire per channel.
    dt : float
        Oscilloscope sampling interval in seconds.
    trigger_position : float
        Trigger position in percent (%) of the total trace length.
    """

    def __init__(self) -> None:
        """Initialise the horizontal settings with default values."""
        self.timebase: int = 3
        self.n_samples: int = 1000
        self.dt: float = 1.0  # Dynamic value, to be read from instrument
        self.trigger_position: float = 0.0

    def fs(self) -> float:
        """Return the PicoScope sampling rate in samples per second (S/s).

        Returns
        -------
        float
            Actual sample rate (1 / dt).
        """
        fs = 1.0 / self.dt
        return fs

    def n_pretrigger(self) -> int:
        """Return number of samples recorded before the trigger.

        Returns
        -------
        int
            Number of pre-trigger sample points.
        """
        return int(self.n_samples * self.trigger_position / 100.0)

    def n_posttrigger(self) -> int:
        """Return the number of samples recorded after the trigger.

        Returns
        -------
        int
            Number of post-trigger sample points.
        """
        return int(self.n_samples - self.n_pretrigger())

    def t0(self) -> float:
        """Return start time, representing the time of the first sample.

        Returns
        -------
        float
            The start time in seconds relative to the trigger event.
        """
        return -self.n_pretrigger() * self.dt

    def t_max(self) -> float:
        """Return end time, representing the time of the last sample.

        Note that this is not the total trace duration, but the time referred
        to the trigger event. The start time may be negative.

        Returns
        -------
        float
            The end time in seconds relative to the trigger event.
        """
        return (self.n_samples - self.n_pretrigger() - 1) * self.dt


class Trigger:
    """Oscilloscope trigger settings and status.

    Attributes
    ----------
    source : str
        Name of the trigger source (e.g., "A", "B", "EXT", "Internal").
    level : float
        Trigger level in Volts.
    direction : str
        Trigger edge direction, typically "Rising" or "Falling".
    delay : float
        Trigger delay in seconds.
    autodelay : float
        Wait time for the auto-trigger feature in seconds.
    adc_max : int
        Instrument ADC maximum value used for scaling.
    """

    def __init__(self) -> None:
        """Initialise the trigger settings with default values."""
        self.source: str = "A"
        self.level: float = 0.5  # [V]
        self.direction: str = "Rising"
        self.delay: float = 0.0  # [s]
        self.autodelay: float = 10e-3  # [s]
        self.adc_max: int = 0  # To be dynamically updated by the instrument

    def enabled(self) -> bool:
        """Deactivate the trigger if the source is set to internal.

        Returns
        -------
        bool
            True if an external or channel-based trigger is active,
            False if it is an internal/software trigger.
        """
        return self.source.lower()[0:3] != "int"


class Picoscope5000A:
    """C-type variables for calling C-style functions in DLLs from Pico SDK.

    Manages the handle to and information about the instrument (oscilloscope).
    The SDK for Picoscope5000A with the instrument uses C-type function calls,
    which requires the use of C-type variables. This class provides an
    interface from Python to these functions.

    Attributes
    ----------
    handle : ctypes.c_int16
        Handle to the instrument, acting as a unique identifier.
    connected : bool
        True if the instrument is connected, False otherwise.
    signal_generator : bool
        True if the instrument contains an arbitrary waveform generator (AWG).
    status : dict of str
        Status messages for the instrument.
    acquisition_ready : ctypes.c_int16
        Flag indicating whether the instrument acquisition has finished.
    max_samples : ctypes.c_int32
        Maximum number of samples to acquire.
    max_adc : ctypes.c_int16
        Maximum value for the instrument ADC.
    overflow : ctypes.c_int16
        Flag indicating if an overflow was detected in the input data.
    channel : str
        Channel name, typically "A", "B", etc.
    buffer : list of ctypes.c_void_p
        Buffer for storing acquired data points.
    awg_max_value : ctypes.c_int16
        Maximum allowed value for the arbitrary waveform generator.
    awg_min_value : ctypes.c_int16
        Minimum allowed value for the arbitrary waveform generator.
    awg_min_length : ctypes.c_int32
        Minimum number of points required for the arbitrary waveform generator.
    awg_max_length : ctypes.c_int32
        Maximum number of points allowed for the arbitrary waveform generator.
    """

    def __init__(self) -> None:
        """Initialise the C-type Picoscope5000A interface."""
        self.handle: ctypes.c_int16 = ctypes.c_int16(0)
        self.connected: bool = False
        self.signal_generator: bool = False
        self.status: dict[str, int | str] = {}
        self.acquisition_ready: ctypes.c_int16 = ctypes.c_int16(0)
        self.max_samples: ctypes.c_int32 = ctypes.c_int32(0)
        self.max_adc: ctypes.c_int16 = ctypes.c_int16(0)
        self.overflow: ctypes.c_int16 = ctypes.c_int16(0)
        self.channel: str = "A"
        self.buffer: list[ctypes.c_void_p] = []
        self.awg_max_value: ctypes.c_int16 = ctypes.c_int16(0)
        self.awg_min_value: ctypes.c_int16 = ctypes.c_int16(0)
        self.awg_min_length: ctypes.c_int32 = ctypes.c_int32(0)
        self.awg_max_length: ctypes.c_int32 = ctypes.c_int32(0)

    def open_adc(self):
        """Open a connection to the PicoScope oscilloscope.

        Configures device resolution to 15 bits.
        Handles alternative power supply states if required by hardware.
        Retrieves maximum ADC value.

        Raises
        ------
        PicoNotOkError
            If the instrument fails to initialize and the returned status code
            does not match any expected power-handling exceptions.
        """
        resolution = picoscope.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_15BIT"]

        self.status["openunit"] = picoscope.ps5000aOpenUnit(
            ctypes.byref(self.handle), None, resolution)

        try:
            assert_pico_ok(self.status["openunit"])
            self.status["changePowerSource"] = 0
        except Exception:
            power_state = self.status["openunit"]

            # Check for power-related status codes
            expected_errors = [
                picoscope.PICO_STATUS["PICO_POWER_SUPPLY_NOT_CONNECTED"],
                picoscope.PICO_STATUS["PICO_USB3_0_DEVICE_NON_USB3_0_PORT"],
            ]

            if power_state in expected_errors:
                self.status["changePowerSource"] = (
                    picoscope.ps5000aChangePowerSource(
                        self.handle,
                        power_state))
                assert_pico_ok(self.status["changePowerSource"])
            else:
                raise

        self.status["maximumValue"] = picoscope.ps5000aMaximumValue(
            self.handle,
            ctypes.byref(self.max_adc),
        )
        assert_pico_ok(self.status["maximumValue"])

        self.connected = ((self.status["changePowerSource"] == 0) and
                          (self.status["maximumValue"] == 0))

        return 0

    def stop_adc(self):
        """Stop oscilloscope acquisition.

        Stops device data collection, applicable when in streaming mode.

        Raises
        ------
        PicoNotOkError
            If the PicoScope SDK returns a status code that indicates the stop
            command failed.
        """
        self.status["stop"] = picoscope.ps5000aStop(self.handle)
        assert_pico_ok(self.status["stop"])

        return 0

    def close_adc(self):
        """Close connection to the PicoScope oscilloscope.

        Shuts down unit associated and updates connection status flags.

        Raises
        ------
        PicoNotOkError
            If the PicoScope SDK returns a status code indicating that the unit
            could not be closed successfully.
        """
        self.status["close"] = picoscope.ps5000aCloseUnit(self.handle)
        assert_pico_ok(dso.status["close"])
        self.connected = False
        return 0

    def set_vertical(self, channel: Channel):
        """Set the vertical scale and channel properties in the oscilloscope.

        Configures enabled channels, coupling mode, voltage range, and
        analogue offset voltage

        Parameters
        ----------
        channel : Channel
            Instance of Channel class defining the channel number, range,
            coupling, and offset.

        Raises
        ------
        PicoNotOkError
            If the PicoScope SDK returns a status code indicating that the channel
            configuration failed.
        """
        name = channel_no_to_name(channel.no)
        status_name = f"setCh{name}"

        coupling_key = f"PS5000A_{channel.coupling.upper()}"
        coupling_code = int(picoscope.PS5000A_COUPLING[coupling_key])

        adc_range = self.find_adc_range(channel.v_max())

        # Call the PicoScope SDK to apply the vertical configuration
        self.status[status_name] = picoscope.ps5000aSetChannel(
            self.handle,
            channel.no,
            channel.enabled,
            coupling_code,
            adc_range,
            channel.offset,
        )
        assert_pico_ok(self.status[status_name])

        return 0

    def set_bwl(self, channel: Channel):
        """Activate or deactivate the bandwidth limit in oscilloscope

        Configures the hardware bandwidth filter (20 MHz) to
        reduce high-frequency noise on the input signal.

        Parameters
        ----------
        channel : Channel
            Instance of Channel class defining channel number and
            bandwidth limiter (bwl) state.

        Raises
        ------
        PicoNotOkError
            If the PicoScope SDK returns a status code indicating that setting the
            bandwidth filter failed.
        """
        status_name = f"setBwl{channel.name()}"
        bwl_value = 1 if channel.bwl else 0
        bwl_param = ctypes.c_int32(bwl_value)

        self.status[status_name] = picoscope.ps5000aSetBandwidthFilter(
            self.handle,
            channel.no,
            bwl_param,
        )
        assert_pico_ok(self.status[status_name])

        return 0

    def set_trigger(self,
                    trigger: Trigger,
                    channel: list[Channel],
                    sampling: Horizontal,
                    ) -> Picoscope5000A:
        """Configure oscilloscope trigger.

        Parameters
        ----------
        trigger : Trigger
            Settings for oscilloscope trigger.
        channel : list of Channel
            Settings for oscilloscope vertical scale, indexed by channel number.
        sampling : Horizontal
            Settings for oscilloscope horizontal scale.
        """
        enabled: int = int(trigger.enabled())

        if trigger.source == "EXT":
            source = picoscope.PS5000A_CHANNEL["PS5000A_EXTERNAL"]
            relative_level = np.clip(trigger.level / 5.0, -1, 1)
            threshold = int(relative_level * trigger.adc_max)

        elif trigger.source in ("A", "B"):
            source = picoscope.PS5000A_CHANNEL[f"PS5000A_CHANNEL_{
                trigger.source}"]
            ch_no = channel_name_to_no(trigger.source)
            relative_level = np.clip(
                trigger.level / channel[ch_no].v_max(), -1, 1
            )
            threshold = int(relative_level * channel[ch_no].adc_max)

        else:
            self.status["trigger"] = -1
            return -1

        # Only two basic trigger modes implemented: Rising or falling edge
        if trigger.direction.lower().startswith("fall"):
            mode = 3  # Trigger mode "Falling"
        else:
            mode = 2  # Trigger mode "Rising"

        delay_pts = int(trigger.delay / sampling.dt)
        autotrigger_ms = ctypes.c_int16(int(trigger.autodelay * 1e3))
        autotrigger_us = ctypes.c_uint64(int(trigger.autodelay * 1e6))

        self.status["trigger"] = picoscope.ps5000aSetSimpleTrigger(
            self.handle,
            enabled,
            source,
            threshold,
            mode,
            delay_pts,
            autotrigger_ms,
        )
        assert_pico_ok(self.status["trigger"])

        self.status["autoTrigger"] = picoscope.ps5000aSetAutoTriggerMicroSeconds(
            self.handle, autotrigger_us
        )
        assert_pico_ok(self.status["autoTrigger"])

        return 0

    def get_trigger_time_offset(self):
        """Read offset of last trigger.

        Returns
        -------
        float
            Time from trigger to first sample point.
        """
        segment_index = 0
        trigger_time = ctypes.c_int64(0)
        time_units = ctypes.c_int32(0)

        self.status["triggerTimeOffset"] = picoscope.ps5000aGetTriggerTimeOffset64(
            self.handle,
            ctypes.byref(trigger_time),
            ctypes.byref(time_units),
            segment_index,
        )
        assert_pico_ok(self.status["triggerTimeOffset"])
        trigger_time_offset = float(trigger_time.value)

        return trigger_time_offset

    def get_sample_interval(self, sampling: Horizontal) -> float:
        """Read actual sample interval from oscilloscope.

        Parameters
        ----------
        sampling : Horizontal
            Settings for oscilloscope horizontal scale.

        Returns
        -------
        float
            Sampling interval for acquired trace.
        """
        sample_interval_ns = ctypes.c_float(0)
        max_n_samples = ctypes.c_int32(0)

        ok = picoscope.ps5000aGetTimebase2(
            self.handle,
            sampling.timebase,
            sampling.n_samples,
            ctypes.byref(sample_interval_ns),
            ctypes.byref(max_n_samples),
            0,
        )
        assert_pico_ok(ok)
        sample_interval = sample_interval_ns.value * 1e-9

        return sample_interval

# =============================================================================
# Taken from documentation, but not available in the Python library
# Not tested, probably not implemented in Python
# def find_sample_interval(dso, dt_requested):
#     """Find sample interval nearest to the requested value."""
#     enabled_channel_flags = 5   # 1+4 meand ch A and B, see documentation
#     adc_resolution = 3          # Corresponds to 15 bit, see documentation
#     use_ets = 0                 # Equivalent Time Sampling, not used
#     timebase = ctypes.c_uint32(0)
#     dt_actual = ctypes.c_double(0)
#     dso.status["findDt"] = picoscope.ps5000aNearestSampleIntervalStateless(
#         dso.handle,
#         enabled_channel_flags,
#         dt_requested,
#         adc_resolution,
#         use_ets,
#         ctypes.byref(timebase),
#         ctypes.byref(dt_actual))
#
#     return int(timebase), float(dt_actual)
# =============================================================================

    def configure_acquisition(self, sampling: Horizontal):
        """Configure acquisition of data from oscilloscope.

        Configures oscilloscope and sets up buffers for input data.

        Parameters
        ----------
        sampling : Horizontal
            Oscilloscope horizontal settings.

        """
        self.max_samples.value = sampling.n_samples
        segment_index = 0
        downsample_mode = 0
        self.buffer = []

        self.buffer.append((ctypes.c_int16 * sampling.n_samples)())
        self.buffer.append((ctypes.c_int16 * sampling.n_samples)())

        status_prefix = "setDataBuffers"
        for ch_no, ch_name in enumerate(CH_NAMES):
            status_name = f"{status_prefix}{ch_name}"

            self.status[status_name] = picoscope.ps5000aSetDataBuffer(
                self.handle,
                ch_no,
                ctypes.byref(self.buffer[ch_no]),
                sampling.n_samples,
                segment_index,
                downsample_mode,
            )
            assert_pico_ok(self.status[status_name])

        return 0

    def acquire_trace(self,
                      sampling: Horizontal,
                      ch: list[Channel],
                      timeout: float = 5.0,
                      ) -> np.ndarray:
        """Acquire voltage trace from oscilloscope.

        Parameters
        ----------
        sampling : Horizontal
            Settings for oscilloscope horizontal scale.
        ch : list of Channel
            Names of channels to acquire.
        timeout : float, optional
            Maximum time to wait for data collection in seconds. Default is 5.0.

        Returns
        -------
        np.ndarray
            Acquired traces, scaled in Volts (2D array of float).

        Raises
        ------
        TimeoutError
            If the oscilloscope does not become ready within `timeout_sec`.
        """
        start_index = 0
        downsample_ratio = 0
        downsample_mode = 0
        segment_index = 0

        self.status["runBlock"] = picoscope.ps5000aRunBlock(
            self.handle,
            sampling.n_pretrigger(),
            sampling.n_posttrigger(),
            sampling.timebase,
            None,
            segment_index,
            None,
            None,
        )
        assert_pico_ok(self.status["runBlock"])

        self.acquisition_ready.value = 0
        start_time = time.time()
        poll_interval = 0.005

        while self.acquisition_ready.value == 0:
            self.status["isReady"] = picoscope.ps5000aIsReady(
                self.handle,
                ctypes.byref(self.acquisition_ready)
            )

            if time.time() - start_time > timeout:
                self.status["isReady"] = -2
                raise TimeoutError(
                    f"Oscilloscope not ready within {timeout} s.")

            time.sleep(poll_interval)

        # Transfer data values
        self.status["getValues"] = picoscope.ps5000aGetValues(
            self.handle,
            start_index,
            ctypes.byref(self.max_samples),
            downsample_ratio,
            downsample_mode,
            segment_index,
            ctypes.byref(self.overflow),
        )
        assert_pico_ok(self.status["getValues"])

        # Convert ADC counts data to Volts
        n_channels = len(ch)
        v_mv = np.zeros([sampling.n_samples, n_channels])

        for ch_no, channel_obj in enumerate(ch):
            adc_range = self.find_adc_range(channel_obj.v_max())
            v_mv[:, ch_no] = adc2mV(self.buffer[ch_no],
                                    adc_range,
                                    self.max_adc,
                                    )

        v = 1e-3 * v_mv

        return v

    def check_awg(self):
        """Check whether the oscilloscope has a waveform generator.

        No dedicated function for this was found in the documentation.
        Uses instead a call to the simplest signal generator function and
        checks for error.   
        """
        self.status["sigGenArbMinMax"] = picoscope.ps5000aSigGenArbitraryMinMaxValues(
            self.handle,
            ctypes.byref(self.awg_min_value),
            ctypes.byref(self.awg_max_value),
            ctypes.byref(self.awg_min_length),
            ctypes.byref(self.awg_max_length),
        )

        try:
            assert_pico_ok(self.status["sigGenArbMinMax"])
            self.signal_generator = True
        except AssertionError:
            self.signal_generator = False

        return 0

    def set_signal(self, sampling: Horizontal, pulse: Pulse):
        """Send pulse to arbitrary waveform generator.

        Parameters
        ----------
        sampling : Horizontal
            Settings for oscilloscope horizontal scale.
        pulse : Pulse
            Definition of pulse for AWG.    
        """
        if pulse.on:
            amplitude = min(pulse.a, DAC_MAX_AMPLITUDE)
        else:
            amplitude = 0

        self.status["sigGenArbMinMax"] = picoscope.ps5000aSigGenArbitraryMinMaxValues(
            self.handle,
            ctypes.byref(self.awg_min_value),
            ctypes.byref(self.awg_max_value),
            ctypes.byref(self.awg_min_length),
            ctypes.byref(self.awg_max_length),
        )

        try:
            assert_pico_ok(self.status["sigGenArbMinMax"])
            self.signal_generator = True
        except AssertionError:
            self.signal_generator = False
            return 0

        if pulse.a == 0:
            y_scaled = np.zeros_like(pulse.y())
        else:
            y_scaled = pulse.y() / pulse.a * self.awg_max_value

        pulsedata = y_scaled.astype(ctypes.c_int16)
        buffer_length = ctypes.c_uint32(len(pulsedata))
        index_mode = ctypes.c_int32(0)
        delta_phase = ctypes.c_uint32(0)

        duration = pulse.duration()
        if duration == 0:
            raise ValueError("Pulse duration cannot be 0.")

        self.status["freqToPhase"] = picoscope.ps5000aSigGenFrequencyToPhase(
            self.handle,
            1 / duration,
            index_mode,
            buffer_length,
            ctypes.byref(delta_phase),
        )
        assert_pico_ok(self.status["freqToPhase"])

        # Values passed as c-type variables
        # See documentation in Programmer's Guide.
        offset_voltage_uv = ctypes.c_int32(0)
        pp_voltage_uv = ctypes.c_uint32(
            int(2 * amplitude * 1e6))  # Peak-to-peak, uV
        trigger_type = ctypes.c_int32(0)
        trigger_source = ctypes.c_int32(pulse.trigger_source)
        shots = ctypes.c_uint32(1)

        waveform_length = ctypes.c_int32(len(pulsedata))
        waveform_pointer = pulsedata.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int16))

        # Parameters not in use
        delta_phase_increment = ctypes.c_uint32(0)
        dwell_count = ctypes.c_uint32(0)
        sweep_type = ctypes.c_int32(0)
        operation = ctypes.c_int32(0)
        sweeps = ctypes.c_uint32(0)
        ext_in_threshold = ctypes.c_int16(0)

        self.status["setSigGenArbitrary"] = picoscope.ps5000aSetSigGenArbitrary(
            self.handle,
            offset_voltage_uv,
            pp_voltage_uv,
            delta_phase.value,
            delta_phase.value,
            delta_phase_increment,
            dwell_count,
            waveform_pointer,
            waveform_length,
            sweep_type,
            operation,
            index_mode,
            shots,
            sweeps,
            trigger_type,
            trigger_source,
            ext_in_threshold,
        )
        assert_pico_ok(self.status["setSigGenArbitrary"])

        return self

    def find_timebase(self, fs_requested: float) -> tuple[int, float]:
        """Find instrument timebase based on requested sample rate.

        Parameters
        ----------
        fs_requested : float
            Requested sample rate.

        Returns
        -------
        timebase : int
            Oscilloscope timebase closest to requested sample rate.
        fs_actual : float
            Actual sample rate for timebase.
        """
        if fs_requested == 0:
            raise ValueError("Sample rate cannot be  0.")

        fs_max = 125e6
        # See documentation for 'Timebases'
        timebase = int(fs_max / fs_requested) + 2
        timebase = max(timebase, 3)                # Min. allowed timebase is 3

        dt = (timebase - 2) / fs_max
        fs_actual = 1 / dt

        return timebase, fs_actual

    def find_adc_range(self, v_maximum: float) -> int:
        """Return the PicoScope range code for the actual voltage range.

        Returns
        -------
        int
            The numerical range index required by the driver.
        """
        if v_maximum < 1.0:
            millivolts = round(v_maximum * 1000)
            adc_range_name = f"PS5000A_{millivolts}MV"
        else:
            volts = round(v_maximum)
            adc_range_name = f"PS5000A_{volts}V"

        return int(picoscope.PS5000A_RANGE[adc_range_name])


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def channel_no_to_name(no: int) -> str:
    """Convert number to Picoscope channel name (A, B, ...)."""
    return chr(ord("A") + no)


def channel_name_to_no(name: str) -> int:
    """Convert Picoscope channel name (A, B, ...) to number."""
    return ord(name) - ord("A")
