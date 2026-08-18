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
from bisect import bisect_left
from typing import ClassVar, Literal
from dataclasses import dataclass
from enum import StrEnum

from picosdk.ps5000a import ps5000a as picoscope
from picosdk.functions import adc2mV, assert_pico_ok
from picosdk.errors import PicoSDKCtypesError

from ultrasound_utilities import Pulse

DAC_SAMPLERATE = 500e6   # [Samples/s] Fixed, see Programmer's guide
CHANNEL_NAMES = ("A", "B")


class Coupling(StrEnum):
    """Oscilloscope coupling."""
    DC = "DC"
    AC = "AC"


@dataclass
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
    coupling : Coupling
        Channel coupling, "DC" or "AC".
    bwl : bool
        Bandwidth limiter status (not available on PS2000 series).
    """

    VALID_RANGES: ClassVar[tuple[float, ...]] = (
        0.01, 0.02, 0.05,
        0.1, 0.2, 0.5,
        1.0, 2.0, 5.0,
        10.0, 20.0, 50.0,
    )

    no: int
    enabled: bool = True
    v_range: float = 1.0
    adc_max: int = 32767
    offset: float = 0.0
    coupling: Coupling = Coupling.DC
    bwl: bool = False

    @property
    def name(self) -> str:
        """PicoScope channel name ('A', 'B', ...)."""
        return channel_no_to_name(self.no)

    @property
    def v_max(self) -> float:
        """Smallest supported range >= requested range."""
        idx = bisect_left(self.VALID_RANGES, self.v_range)

        if idx > len(self.VALID_RANGES):
            return self.VALID_RANGES[-1]

        return self.VALID_RANGES[idx]

    @property
    def coupling_code(self) -> int:
        return int(
            picoscope.PS5000A_COUPLING[f"PS5000A_{self.coupling.upper()}"])


@dataclass
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
        Trigger position in percent of otal trace length.
    """
    timebase: int = 3
    n_samples: int = 1000
    dt: float = 1.0
    trigger_position: float = 0.0

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError("dt must be positive")

        if not 0.0 <= self.trigger_position <= 100.0:
            raise ValueError("trigger_position must be between 0 and 100%")

    @property
    def sample_rate(self) -> float:
        """Sample rate in samples per second."""
        return 1.0 / self.dt

    @property
    def n_pretrigger(self) -> int:
        """Number of samples before trigger."""
        return round(self.n_samples * self.trigger_position / 100.0)

    @property
    def n_posttrigger(self) -> int:
        """Number of samples after trigger."""
        return self.n_samples - self.n_pretrigger

    @property
    def start_time(self) -> float:
        """Time of first sample relative to trigger."""
        return -self.n_pretrigger * self.dt

    @property
    def end_time(self) -> float:
        """Time of last sample relative to trigger."""
        return (self.n_posttrigger - 1) * self.dt


class TriggerDirection(StrEnum):
    """Supported trigger edge directions."""
    RISING = "Rising"
    FALLING = "Falling"


class TriggerSource(StrEnum):
    """Supported trigger edge directions."""
    A = "Ch A"
    B = "Ch B"
    EXT = 'EXT'
    INTERNAL = 'Internal'


@dataclass
class Trigger:
    """Oscilloscope trigger settings and status.

    Attributes
    ----------
    source : TriggerSource
        Trigger source (e.g. "A", "B", "EXT", "Internal").
    level : float
        Trigger level in volts.
    direction : TriggerDirection
        Trigger edge direction.
    delay : float
        Trigger delay in seconds.
    autodelay : float
        Auto-trigger timeout in seconds.
    adc_max : int
        Instrument ADC maximum value used for scaling.
    """
    source: TriggerSource = TriggerSource.A
    level: float = 0.5
    direction: TriggerDirection = TriggerDirection.RISING
    delay: float = 0.0
    autodelay: float = 0.01
    adc_max: int = 0

    def __post_init__(self) -> None:
        """Validate settings."""
        if self.autodelay < 0:
            raise ValueError("autodelay must be non-negative")

        if self.adc_max < 0:
            raise ValueError("adc_max must be non-negative")

    @property
    def enabled(self) -> bool:
        """Whether a hardware trigger is active."""
        return self.source.casefold().startswith("int") is False


class Picoscope5000A:
    """Interface between Python and PicoScope 5000A SDK.

    Manages device connection state and C-compatible variables required by
    the PicoSDK functions. The SDK for Picoscope5000A uses C-type function
    calls, which requires the use of C-type variables. This class provides an
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
    MAX_SAMPLE_RATE = 125e6
    DAC_MAX_AMPLITUDE = 2.0
    RESOLUTION = picoscope.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_15BIT"]

    POWER_ERRORS = {
        picoscope.PICO_STATUS["PICO_POWER_SUPPLY_NOT_CONNECTED"],
        picoscope.PICO_STATUS["PICO_USB3_0_DEVICE_NON_USB3_0_PORT"]}

    TRIGGER_MODES = {TriggerDirection.RISING: 2,
                     TriggerDirection.FALLING: 3}

    def __init__(self) -> None:
        self.handle = ctypes.c_int16()

        self.connected = False
        self.signal_generator = False

        self.status: dict[str, int] = {}

        self.acquisition_ready = ctypes.c_int16()
        self.max_samples = ctypes.c_int32()
        self.max_adc = ctypes.c_int16()
        self.overflow = ctypes.c_int16()

        self.channel = "A"
        self.buffer: list[ctypes.c_void_p] = []

        self.awg_max_value = ctypes.c_int16()
        self.awg_min_value = ctypes.c_int16()
        self.awg_min_length = ctypes.c_int32()
        self.awg_max_length = ctypes.c_int32()

    def _set_and_check(self, key: str, status: int) -> int:
        self.status[key] = status
        assert_pico_ok(status)
        return status

    def open_adc(self) -> None:
        """Open and initialise the PicoScope."""

        resolution = (picoscope.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_15BIT"])
        status = picoscope.ps5000aOpenUnit(ctypes.byref(self.handle),
                                           None,
                                           resolution)
        self.status["open_unit"] = status

        try:
            assert_pico_ok(status)
            self.status["change_power_source"] = 0
        except PicoSDKCtypesError:
            self._handle_power_state(status)

        self._read_max_adc()
        self.connected = True

        return

    def _handle_power_state(self, status: int) -> None:
        """Handle expected power-related startup states."""
        if status not in self.POWER_ERRORS:
            raise

        self._set_and_check("changePowerSource",
                            picoscope.ps5000aChangePowerSource(
                                self.handle, status)
                            )
        return

    def _read_max_adc(self) -> None:
        """Read ADC scaling information from the instrument."""
        self._set_and_check("maximum_value",
                            picoscope.ps5000aMaximumValue(
                                self.handle,
                                ctypes.byref(self.max_adc))
                            )
        return

    def stop_adc(self) -> None:
        """Stop oscilloscope acquisition.

        Stops device data collection, applicable when in streaming mode.

        Raises
        ------
        PicoSDKCtypesError
            If the PicoScope SDK returns a status code that indicates the stop
            command failed.
        """
        self._set_and_check("stop", picoscope.ps5000aStop(self.handle))
        return

    def close_adc(self) -> None:
        """Close the PicoScope connection.

        Shuts down unit associated and updates connection status flags.

        Raises
        ------
        PicoSDKCtypesError
            If the PicoScope SDK returns a status code indicating that the unit
            could not be closed successfully.
        """
        self._set_and_check("close", picoscope.ps5000aCloseUnit(self.handle))
        self.connected = False

        return

    def set_vertical(self, channel: Channel) -> None:
        """Configure oscilloscope channel settings.

        Applies the channel enable state, coupling mode, voltage range,
        and analogue offset.

         Parameters
        ----------
        channel : Channel
            Channel configuration.
        """
        self._set_and_check(f"set_ch_{channel.name}",
                            picoscope.ps5000aSetChannel(
                                self.handle,
                                channel.no,
                                channel.enabled,
                                channel.coupling_code,
                                self.find_adc_range(channel.v_max),
                                channel.offset),
                            )
        return

    def set_bwl(self, channel: Channel) -> None:
        """Configure the channel bandwidth limiter.

        Configures the hardware bandwidth filter (20 MHz) to
        reduce high-frequency noise on the input signal.

        Parameters
        ----------
        channel : Channel
            Instance of Channel class defining channel number and
            bandwidth limiter (bwl) state.

        Raises
        ------
        PicoSDKCtypesError
            If the PicoScope SDK returns a status code indicating that setting
            the bandwidth filter failed.
        """
        bwl_param = ctypes.c_int32(int(channel.bwl))

        self._set_and_check(f"set_bwl_{channel.name}",
                            picoscope.ps5000aSetBandwidthFilter(
                                self.handle,
                                channel.no,
                                bwl_param)
                            )
        return

    def _trigger_source_and_threshold(self,
                                      trigger: Trigger,
                                      channels: list[Channel],
                                      ) -> tuple[int, int]:
        """Find trigger source and ADC threshold."""
        if trigger.source == "EXT":
            source = picoscope.PS5000A_CHANNEL["PS5000A_EXTERNAL"]
            relative_level = np.clip(trigger.level / 5.0, -1.0, 1.0)
            threshold = int(relative_level * trigger.adc_max)
            return source, threshold

        if trigger.source in {"A", "B"}:
            ch_no = channel_name_to_no(trigger.source)
            source = picoscope.PS5000A_CHANNEL[
                f"PS5000A_CHANNEL_{trigger.source}"]
            relative_level = np.clip(trigger.level / channels[ch_no].v_max,
                                     -1.0, 1.0)

            threshold = int(relative_level * channels[ch_no].adc_max)
            return source, threshold

    def set_trigger(self,
                    trigger: Trigger,
                    channels: list[Channel],
                    sampling: Horizontal) -> None:
        """Configure the oscilloscope trigger.

        Parameters
        ----------
        trigger : Trigger
            Trigger configuration.
        channels : list[Channel]
            Channel configurations indexed by channel number.
        sampling : Horizontal
            Horizontal acquisition settings.
        """
        enabled = int(trigger.enabled)
        source, threshold = self._trigger_source_and_threshold(trigger,
                                                               channels)
        mode = self.TRIGGER_MODES[trigger.direction]
        delay_points = int(trigger.delay / sampling.dt)
        autotrigger_ms = ctypes.c_int16(int(trigger.autodelay * 1e3))
        autotrigger_us = ctypes.c_uint64(int(trigger.autodelay * 1e6))

        self._set_and_check("trigger",
                            picoscope.ps5000aSetSimpleTrigger(
                                self.handle,
                                enabled,
                                source,
                                threshold,
                                mode,
                                delay_points,
                                autotrigger_ms)
                            )

        self._set_and_check("auto_trigger",
                            picoscope.ps5000aSetAutoTriggerMicroSeconds(
                                self.handle,
                                autotrigger_us)
                            )
        return

    def get_trigger_time_offset(self) -> float:
        """Return the trigger time offset.

        Returns
        -------
        float
            Time offset reported by the PicoScope SDK.
        """
        trigger_time = ctypes.c_int64()
        time_units = ctypes.c_int32()

        self._set_and_check("trigger_time_offset",
                            picoscope.ps5000aGetTriggerTimeOffset64(
                                self.handle,
                                ctypes.byref(trigger_time),
                                ctypes.byref(time_units),
                                0)
                            )

        return float(trigger_time.value)

    def get_sample_interval(self, sampling: Horizontal) -> float:
        """Query and update the actual sampling interval.

        Parameters
        ----------
        sampling : Horizontal
            Settings for oscilloscope horizontal scale.

        Returns
        -------
        float
            Sampling interval for acquired trace.
        """
        sample_interval_ns = ctypes.c_float()
        max_samples = ctypes.c_int32()

        self._set_and_check("timebase",
                            picoscope.ps5000aGetTimebase2(
                                self.handle,
                                sampling.timebase,
                                sampling.n_samples,
                                ctypes.byref(sample_interval_ns),
                                ctypes.byref(max_samples),
                                0)
                            )

        sampling.dt = sample_interval_ns.value * 1e-9
        self.max_samples = max_samples

        return sampling.dt

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

    def configure_acquisition(self, sampling: Horizontal) -> None:
        """Configure acquisition and data buffers.

        Parameters
        ----------
        sampling : Horizontal
            Horizontal acquisition settings.
        """
        self.max_samples.value = sampling.n_samples
        segment_index = 0
        downsample_mode = 0

        self.buffer = [(ctypes.c_int16 * sampling.n_samples)()
                       for _ in CHANNEL_NAMES]

        for ch_no, ch_name in enumerate(CHANNEL_NAMES):
            self._set_and_check(f"set_data_buffer_{ch_name}",
                                picoscope.ps5000aSetDataBuffer(
                                    self.handle,
                                    ch_no,
                                    ctypes.byref(self.buffer[ch_no]),
                                    sampling.n_samples,
                                    segment_index,
                                    downsample_mode)
                                )
        return

    def acquire_trace(self,
                      sampling: Horizontal,
                      channels: list[Channel],
                      timeout: float = 5.0,
                      ) -> np.ndarray:
        """Acquire voltage traces from the oscilloscope.

        Parameters
        ----------
        sampling : Horizontal
            Horizontal acquisition settings.
        channels : list[Channel]
            Configured channels.
        timeout : float, optional
            Maximum wait time in seconds.

        Returns
        -------
        np.ndarray
            Sample data in volts with shape
            (n_samples, n_channels).
        """

        self._start_block_acquisition(sampling)
        self._wait_for_acquisition(timeout)
        self._read_acquisition_data()

        voltages_mv = np.empty(
            (sampling.n_samples, len(channels)),
            dtype=float)

        for channel in channels:
            adc_range = self.find_adc_range(channel.v_max)
            voltages_mv[:, channel.no] = adc2mV(self.buffer[channel.no],
                                                adc_range,
                                                self.max_adc)

        return voltages_mv * 1e-3

    def _start_block_acquisition(self, sampling: Horizontal) -> None:
        """Start a block acquisition."""

        self._set_and_check("run_block",
                            picoscope.ps5000aRunBlock(
                                self.handle,
                                sampling.n_pretrigger,
                                sampling.n_posttrigger,
                                sampling.timebase,
                                None,
                                0,
                                None,
                                None)
                            )
        return

    def _wait_for_acquisition(self,
                              timeout: float,
                              poll_interval: float = 0.001) -> None:
        """Wait for acquisition to complete."""
        self.acquisition_ready.value = 0
        start = time.monotonic()

        while not self.acquisition_ready.value:
            self.status["is_ready"] = picoscope.ps5000aIsReady(
                self.handle,
                ctypes.byref(self.acquisition_ready),
            )

            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Oscilloscope not ready within {timeout:.1f} s."
                )

            time.sleep(poll_interval)
        return

    def _read_acquisition_data(self) -> None:
        """Transfer acquisition data from the scope to the buffer."""

        self._set_and_check("get_values",
                            picoscope.ps5000aGetValues(
                                self.handle,
                                0,
                                ctypes.byref(
                                    self.max_samples),
                                0,
                                0,
                                0,
                                ctypes.byref(self.overflow))
                            )

    def check_awg(self) -> bool:
        """Check whether the oscilloscope supports an AWG.

        No dedicated function for this was found in the documentation.
        Uses instead a call to the simplest signal generator function and
        checks for error.

        Returns
        -------
        bool
            True if an arbitrary waveform generator is available,
            otherwise False.
        """

        status = picoscope.ps5000aSigGenArbitraryMinMaxValues(
            self.handle,
            ctypes.byref(self.awg_min_value),
            ctypes.byref(self.awg_max_value),
            ctypes.byref(self.awg_min_length),
            ctypes.byref(self.awg_max_length),
        )
        self.status["sigGenArbMinMax"] = status

        try:
            assert_pico_ok(status)
            self.signal_generator = True
        except PicoSDKCtypesError:
            self.signal_generator = False

        return self.signal_generator

    def set_signal(self,
                   sampling: Horizontal,
                   pulse: Pulse) -> None:
        """Send a pulse to the arbitrary waveform generator.

        Parameters
        ----------
        sampling : Horizontal
            Horizontal acquisition settings.
        pulse : Pulse
            AWG pulse definition.
        """
        amplitude = min(pulse.a, self.DAC_MAX_AMPLITUDE) if pulse.on else 0.0

        if not self.check_awg():
            return

        if pulse.duration <= 0:
            raise ValueError("Pulse duration must be positive.")

        if pulse.a == 0:
            y_scaled = np.zeros_like(pulse.y)
        else:
            y_scaled = (pulse.y / pulse.a * self.awg_max_value.value)

        pulse_data = y_scaled.astype(np.int16)

        buffer_length = ctypes.c_uint32(len(pulse_data))
        waveform_length = ctypes.c_int32(len(pulse_data))
        index_mode = ctypes.c_int32(0)
        delta_phase = ctypes.c_uint32()

        self._set_and_check("freq_to_phase",
                            picoscope.ps5000aSigGenFrequencyToPhase(
                                self.handle,
                                1.0 / pulse.duration,
                                index_mode,
                                buffer_length,
                                ctypes.byref(delta_phase))
                            )

        waveform_pointer = pulse_data.ctypes.data_as(
            ctypes.POINTER(ctypes.c_int16)
        )

        offset_voltage_uv = ctypes.c_int32(0)
        pp_voltage_uv = ctypes.c_uint32(int(2 * amplitude * 1e6))
        shots = ctypes.c_uint32(1)
        sweeps = ctypes.c_uint32(0)
        trigger_type = ctypes.c_int32(0)
        trigger_source = ctypes.c_int32(pulse.trigger_source)
        sweep_type = ctypes.c_int32(0)
        operation = ctypes.c_int32(0)
        delta_phase_increment = ctypes.c_uint32(0)
        dwell_count = ctypes.c_uint32(0)
        ext_in_threshold = ctypes.c_int16(0)

        self._set_and_check("set_sig_gen_arbitrary",
                            picoscope.ps5000aSetSigGenArbitrary(
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
                            ),
                            )
        return

    def find_timebase(self, fs_requested: float) -> tuple[int, float]:
        """Find the closest supported timebase.

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
        if fs_requested <= 0:
            raise ValueError("Sample rate must be positive.")

        timebase = max(int(self.MAX_SAMPLE_RATE / fs_requested) + 2, 3)
        fs_actual = (self.MAX_SAMPLE_RATE / (timebase - 2))

        return timebase, fs_actual

    def find_adc_range(self, v_maximum: float) -> int:
        """Return the PicoScope range code for a voltage range.

        Parameters
        ----------
        v_maximum : float
            Full-scale voltage range.

        Returns
        -------
        int
            PicoScope SDK range code.
        """
        if v_maximum <= 0:
            raise ValueError("Voltage range must be positive.")
        if v_maximum < 1.0:
            range_name = f"PS5000A_{round(v_maximum*1000)}MV"
        else:
            range_name = f"PS5000A_{round(v_maximum)}V"

        try:
            return int(picoscope.PS5000A_RANGE[range_name])
        except KeyError as exc:
            raise ValueError(f"Unsupported voltage range: {v_maximum}"
                             ) from exc

        return


def channel_no_to_name(channel_no: int) -> str:
    """Convert a channel number to a channel name."""
    try:
        return CHANNEL_NAMES[channel_no]
    except IndexError as exc:
        raise ValueError(
            f"Invalid channel number: {channel_no}"
        ) from exc
    return


def channel_name_to_no(channel_name: str) -> int:
    """Convert a channel name to a channel number."""
    try:
        return CHANNEL_NAMES.index(channel_name.upper())
    except ValueError as exc:
        raise ValueError(
            f"Invalid channel name: {channel_name!r}"
        ) from exc
    return
