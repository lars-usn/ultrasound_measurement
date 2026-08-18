"""Utility functions for ultrasound measurement systems at USN IMS.

Based on former systems in LabVIEW and Matlab

Lars Hoff, USN, Sep 2022
Modified July 2026
    - Follow PEP-8 and numpy docstring style guides.
    - Tested on Ubuntu
    - General code cleanup with help from Gemini
"""

from dataclasses import dataclass, field
from math import pi, radians, log10, floor, frexp, isclose
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from datetime import date
from pathlib import Path
from enum import StrEnum


@dataclass
class Waveform:
    """Measurement results as 1D time traces."""

    y: np.ndarray = field(
        default_factory=lambda: np.zeros((100, 1))
    )

    dt: float = 1.0
    t0: float = 0.0
    dtr: float = 0.0

    def __post_init__(self) -> None:
        """Validate and standardize waveform data."""

        self.y = np.asarray(self.y)

        if self.y.ndim == 1:
            self.y = self.y.reshape((-1, 1))

        self.dt = float(self.dt)
        self.t0 = float(self.t0)
        self.dtr = float(self.dtr)

        if self.dt <= 0:
            raise ValueError("dt must be positive.")

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return self.y.shape[1]

    @property
    def n_samples(self) -> int:
        """Number of samples per channel."""
        return self.y.shape[0]

    @property
    def fs(self) -> float:
        """Sample rate [Hz]."""
        return 1.0 / self.dt

    @property
    def t(self) -> np.ndarray:
        """Time vector [s]."""
        return self.t0 + np.arange(self.n_samples) * self.dt

    def n_fft(self, upsample: int = 0) -> int:
        """Number of FFT points."""
        upsample = max(round(upsample), 0)

        m, e = frexp(self.n_samples)
        n = 2 ** (e + upsample)

        return max(n, 1024)

    @property
    def f(self) -> np.ndarray:
        """Frequency vector."""
        return (np.arange(0, self.n_fft() // 2) / self.n_fft() * self.fs)

    def powerspectrum(self, normalise=False, scale="linear", upsample=2):
        """Calculate power spectrum."""
        return powerspectrum(self.y, self.dt,
                             n_fft=self.n_fft(upsample=upsample),
                             scale=scale,
                             normalise=normalise)

    def filtered(self, wave_filter):
        """Return filtered copy of waveform."""
        filter_type = (str(wave_filter.filter_type).strip().lower())

        print('')
        print('filtered method')
        print(wave_filter)
        print('')

        if filter_type.startswith("no"):
            y_filtered = self.y.copy()

        elif filter_type.startswith("ac"):
            y_filtered = self.y - self.y.mean(axis=0)

        else:
            b, a = wave_filter.coefficients()
            y_filtered = signal.filtfilt(b, a, self.y, axis=0)

        return Waveform(y=y_filtered, dt=self.dt, t0=self.t0, dtr=self.dtr)

    def zoomed(self, tlim):
        """Return waveform limited to interval."""
        t_start, t_end = sorted(tlim)

        idx = np.flatnonzero((self.t >= t_start) & (self.t <= t_end))

        if idx.size == 0:
            raise ValueError(f"No samples found between "
                             f"{t_start:g} and {t_end:g} s.")

        return Waveform(y=self.y[idx, :], dt=self.dt,
                        t0=self.t[idx[0]], dtr=self.dtr)

    def plot(self, time_unit="us", ch=(0, 1), y_max=None):
        """Plot time traces using specified time unit.

        Parameters
        ----------
        time_unit : str, optional
            Unit to plot time in ('s', 'ms', 'us'). Defaults to "us".
        ch : array_like, optional
            Channels to plot. Defaults to (0, 1) if not specified.
        y_max : float, optional
            Maximum scale on the amplitude axis. Defaults to None.

        """
        ch = np.array(ch)
        ch = ch[ch < self.n_channels]
        plot_pulse(t=self.t, y=self.y[:, ch], time_unit=time_unit, y_max=y_max)
        return

    def plot_spectrum(self, time_unit="s", ch=(0, 1), y_max=None, f_max=None,
                      normalise=True, scale="dB", db_min=-40, ax=None):
        """Plot trace and power spectrum in one graph.

        Parameters
        ----------
        time_unit : str, optional
            Unit to plot time in ('s', 'ms', 'us'). Defaults to "s".
        ch : array_like, optional
            Channels to plot. Defaults to [0, 1] if not specified.
        y_max : float, optional
            Maximum scale on the amplitude axis. Defaults to None.
        f_max : float, optional
            Maximum scale on the frequency axis. Defaults to None.
        normalise : bool, optional
            Normalise power spectrum plot to 1 (0 dB). Defaults to True.
        scale : str, optional
            Scaling option, either "linear" or "dB". Defaults to "dB".
        db_min : float, optional
            Dynamic range on dB-plot. Defaults to -40.
        ax : array_like, optional
            List of axes objects to plot time trace and spectrum.
            Defaults to None.

        """
        plot_spectrum(self.t(), self.y[:, ch],
                      time_unit=time_unit,
                      y_max=y_max,
                      f_max=f_max,
                      n_fft=self.n_fft(),
                      normalise=normalise,
                      scale=scale,
                      db_min=db_min,
                      ax=ax)
        return

    def save(self, filename, overwrite=True):
        """Save 'Waveform' variable to binary file as 4-byte (sgl) floats.

        Compatible with the internal format used since the 1990s on a variety
        of platforms (LabWindows, C, LabVIEW, MATLAB). Uses 'C-order' of arrays
        and IEEE big-endian byte order. Complements load().

        Parameters
        ----------
        filename : str
            Full path of the file to save data in.
        overwrite : bool, optional
            If True, overwrites the file if it exists. If False, raises a
            FileExistsError. Defaults to True.

        Returns
        -------
        int
            Returns 0 upon successful execution.

        Raises
        ------
        FileExistsError
            If `overwrite` is False and the file already exists.
        """
        header = "<WFM_Python_>f4>"  # Header gives source and data format
        n_header = len(header)
        mode = 'wb' if overwrite else 'xb'

        y_contiguous = np.ascontiguousarray(self.y, dtype='>f4')
        with open(filename, mode) as fid:
            fid.write(np.array(n_header, dtype='>i4').tobytes())
            fid.write(bytes(header, 'utf-8'))
            fid.write(np.array(self.n_channels, dtype='>u4').tobytes())
            fid.write(np.array(self.t0, dtype='>f8').tobytes())
            fid.write(np.array(self.dt, dtype='>f8').tobytes())
            fid.write(np.array(self.dtr, dtype='>f8').tobytes())
            fid.write(y_contiguous.tobytes())

        return 0

    @classmethod
    def load(cls, filename):
        """Load waveform from file and return a new instance.

        Loads the contents of the binary file into the instance variables.
        This format is compatible with the internal format used since the
        1990s across various platforms (LabWindows, C, LabVIEW, MATLAB).
        It uses C-order arrays and IEEE big-endian byte order.

        Parameters
        ----------
        filename : str
            The full path of the file to load.

        Returns
        -------
        Waveform class
            New instance of a waveform

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        IOError
            If the file cannot be read.
        """

        with open(filename, "rb") as fid:
            # Header length and the header string
            n_header = int(np.fromfile(fid, dtype=">i4", count=1)[0])
            header = fid.read(n_header).decode("utf-8")

            # Channel configuration and time parameters
            n_ch = int(np.fromfile(fid, dtype=">u4", count=1)[0])
            t0 = float(np.fromfile(fid, dtype=">f8", count=1)[0])
            dt = float(np.fromfile(fid, dtype=">f8", count=1)[0])
            dtr = float(np.fromfile(fid, dtype=">f8", count=1)[0])

            # Signal traces, 2D array
            y = np.fromfile(fid, dtype=">f4")
            y = y.reshape((-1, n_ch))

        wfm = cls(y=y, dt=dt, t0=t0, dtr=dtr)
        wfm.header = header
        wfm.sourcefile = filename

        return wfm


@dataclass
class Pulse:
    """Create standardised theoretical ultrasound pulses.

    For simulations or transfer to a signal generator. Defines a standard
    pulse from given attributes.

    Attributes
    ----------
    shape : str
        Carrier wave shape: "sine", "square", "triangle", "sawtooth".
    envelope : str
        Pulse envelope: "rectangular", "hann", "hamming", "triangle", "tukey".
    n_cycles : float
        Pulse length as number of cycles.
    f0 : float
        Carrier wave frequency in Hz.
    a : float
        Amplitude.
    phase : float
        Phase of carrier wave in degrees, referenced to a cosine.
    dt : float
        Sample interval in seconds.
    alpha : float
        Tukey window cosine-fraction, alpha = 0.0 to 1.0.
    trigger_source : int
        Trigger source identifier (not fully implemented yet).
    available : bool
        Availability status of the pulser.
    on : bool
        Power/activation status ("ON"/"OFF").
    """

    shape: str = "sine"
    envelope: str = "rectangular"
    n_cycles: float = 2.0
    f0: float = 2.0e6
    a: float = 1.0
    phase: float = 0.0
    dt: float = 8e-9
    alpha: float = 0.5
    trigger_source: int = 1
    available: bool = False
    on: bool = False

    def __post_init__(self) -> None:

        if self.f0 <= 0:
            raise ValueError("f0 must be positive.")

        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        if self.n_cycles <= 0:
            raise ValueError("n_cycles must be positive.")

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1.")

    @property
    def t(self) -> np.ndarray:
        """Time vector [s].

        Returns
        -------
        np.ndarray
            1D array of float representing the time vector in seconds.
        """

        return np.arange(0.0, self.duration, self.dt)

    @property
    def y(self) -> np.ndarray:
        """Create the pulse time trace from the input specification.

        Returns
        -------
        np.ndarray
            1D array of float representing the generated pulse waveform.
        """
        # Select the appropriate window/envelope
        windows = {
            "rec": lambda n: signal.windows.boxcar(n),
            "han": lambda n: signal.windows.hann(n),
            "ham": lambda n: signal.windows.hamming(n),
            "tri": lambda n: signal.windows.triang(n),
            "tuk": lambda n: signal.windows.tukey(n, self.alpha),
        }

        win = windows.get(
            self.envelope[:3].lower(),
            windows["rec"]
        )(self.n_samples)

        phase_arg = 2 * pi * self.f0 * self.t + radians(self.phase)

        # Select the carrier wave shape
        match self.shape.lower()[0:3]:
            case "squ":
                s = 0.5 * signal.square(phase_arg, duty=0.5)
            case "tri":
                s = 0.5 * signal.sawtooth(phase_arg, width=0.5)
            case "saw":
                s = 0.5 * signal.sawtooth(phase_arg, width=1.0)
            case _:
                s = np.cos(phase_arg)

        y_signal = self.a * win * s
        if len(y_signal) > 0:
            y_signal[-1] = 0.0

        return y_signal

    @property
    def period(self) -> float:
        """Carrier-wave period [s]."""
        return 1.0 / self.f0

    @property
    def duration(self) -> float:
        """Pulse duration [s]."""
        return self.n_cycles * self.period

    @property
    def n_samples(self) -> int:
        """Find the number of samples in the pulse."""
        return len(self.t)

    @property
    def time_unit(self) -> str:
        """Set time unit for plotting based on centre frequency.

        Returns
        -------
        str
            Time unit string ("ns", "us", "ms", or "s").
        """
        if self.f0 > 1e9:
            return "ns"
        if self.f0 > 1e6:
            return "us"
        if self.f0 > 1e3:
            return "ms"
        return "s"

    @property
    def n_fft(self) -> int:
        """Set the number of points used to calculate the spectrum.

        The number is always a power of 2, using for zero-padding.

        Returns
        -------
        int
            Number of FFT points (minimum 2048).
        """
        # math.frexp splits a float into mantissa and exponent
        _, e = frexp(self.n_samples)
        n = 2 ** (e + 3)
        return max(n, 2048)

    def powerspectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the power spectrum of the pulse trace.

        Returns
        -------
        f : np.ndarray
            1D array of float representing the frequency vector.
        psd : np.ndarray
            1D array of float representing the power spectral density.
        """
        f, psd = powerspectrum(y=self.y,
                               dt=self.dt,
                               n_fft=self.n_fft,
                               scale="dB",
                               normalise=True)
        return f, psd

    def plot(self) -> int:
        """Plot the pulse in the time domain.

        Returns
        -------
        int
            Returns 0 upon successful execution.
        """
        plot_pulse(t=self.t, y=self.y, time_unit=self.time_unit)
        return 0

    def plot_spectrum(self) -> int:
        """Plot both the time trace and the power spectrum.

        Returns
        -------
        int
            Returns 0 upon successful execution.
        """
        plot_spectrum(t=self.t, y=self.y,
                      time_unit=self.time_unit,
                      f_max=scale_125(3*self.f0),
                      n_fft=self.n_fft,
                      scale="db",
                      normalise=True)
        return 0


class FilterType(StrEnum):
    """Supported waveform filter types."""
    BYPASS = "No"
    AC = "AC"
    RF = "RF filter"


@dataclass(slots=True)
class WaveformFilter:
    """Digital filter definition for waveform processing."""

    filter_type: FilterType = FilterType.BYPASS
    f_min: float = 100e3
    f_max: float = 10e6
    order: int = 2
    sample_rate: float = 100e6

    @property
    def nyquist(self) -> float:
        """Nyquist frequency in Hz."""
        return self.sample_rate / 2

    def validate(self) -> None:
        """Validate filter parameters."""

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        if self.order < 1:
            raise ValueError("order must be >= 1")

        if self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min")

    def normalized_cutoffs(self) -> np.ndarray:
        """Return cutoff frequencies normalized to Nyquist."""
        return np.array([self.f_min, self.f_max]) / self.nyquist

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Return Butterworth filter coefficients."""

        self.validate()
        fn = self.normalized_cutoffs()
        if fn[0] <= 0:
            return signal.butter(self.order, fn[1],
                                 btype="lowpass", output="ba")

        if fn[1] > 1.0:
            return signal.butter(self.order, fn[0],
                                 btype="highpass", output="ba")

        return signal.butter(self.order, fn, btype="bandpass", output="ba")


@dataclass
class ResultFile:
    """Configuration and metadata for a result file.

    Attributes
    ----------
    prefix : str
        Prefix for the file name.
    ext : str
        File extension (e.g. "trc").
    path : str
        Full path to the file.
    directory : str
        Directory where the file is stored.
    name : str
        Base name of the file.
    counter : int
        File counter or index.
    """

    prefix: str = "test"
    ext: str = "trc"
    path: str = ""
    directory: str = ""
    name: str = ""
    counter: int = 0


def scale_125(x: float) -> float:
    """Find the next number in a 1-2-5-10-20... sequence.

    Parameters
    ----------
    x : float
        Reference value, positive or negative.

    Returns
    -------
    float
        Next number in the 1-2-5 sequence greater than or equal to the
        magnitude of x.
    """
    if x == 0:
        return 1.0

    magnitude = abs(x)
    exponent = floor(log10(magnitude))
    mantissa = magnitude / 10**exponent

    for prefix in (1, 2, 5, 10):
        if mantissa < prefix or isclose(mantissa, prefix):
            return prefix * 10**exponent

    return 10**(exponent + 1)


def find_timescale(time_unit: str = "s") -> tuple[float, str]:
    """Return time multiplier and frequency axis scaling based on a time unit.

    Parameters
    ----------
    time_unit : str, default "s"
        Time unit used in plots: "s", "ms", "us", "ns".

    Returns
    -------
    multiplier : float
        Multiplier for time to get the requested unit.
    freq_unit : str
        Corresponding frequency unit.
    """

    scales = {"s": (1.0, "Hz"),
              "ms": (1e3, "kHz"),
              "us": (1e6, "MHz"),
              "ns": (1e9, "GHz")}

    return scales.get(time_unit, (1.0, "Hz"))


def find_limits(limits: np.ndarray, min_diff: float = 1.0) -> np.ndarray:
    """Find the minimum and maximum values as a NumPy array.

    Ensures that the difference between the maximum and minimum values
    is at least the specified minimum difference.

    Parameters
    ----------
    limits : np.ndarray
        1D array or array-like of floats representing the requested limits.
    min_diff : float, default 1.0
        The minimum required difference between the min and max values.

    Returns
    -------
    np.ndarray
        1D array containing [min_value, max_value].
    """
    min_value = np.min(limits)
    max_value = max(np.max(limits), min_value + min_diff)

    return np.array([min_value, max_value], dtype=float)


def read_scaled_value(quantity: str) -> float:
    """Interpret a text string as a scaled floating-point value.

    Parses strings containing metric prefixes (e.g., micro, milli, kilo, Mega,
    Giga) followed by their unit, separating the number and the unit.

    Parameters
    ----------
    quantity : str
        The value as a string (e.g., "3.4 MHz", "100 us", "50").

    Returns
    -------
    float
        The value scaled according to its metric prefix.
    """
    prefixes = {"u": 1e-6,
                "µ": 1e-6,
                "μ": 1e-6,
                "m": 1e-3,
                "k": 1e3,
                "M": 1e6,
                "G": 1e9}

    parts = quantity.strip().split()
    if not parts:
        raise ValueError("Empty input")

    value = float(parts[0])
    if len(parts) == 1:
        return value

    multiplier = prefixes.get(parts[1][0], 1.0)
    return value * multiplier


def find_filename(prefix: str = "test",
                  ext: str = "trc",
                  resultdir: str = "../results") -> ResultFile:
    """Find a new unique file name based on the current date and a counter.

    Finds the next free file name in the format `prefix_yyyy_mm_dd_nnnn.ext`
    where `yyyy_mm_dd` is the date and `nnnn` is a counter. The file is mapped
    to the directory `resultdir`. The last counter value is tracked and saved
    in a local counter file named `prefix.cnt`.

    Parameters
    ----------
    prefix : str, default "test"
        Code that characterises the measurement type.
    ext : str, default "trc"
        File extension.
    resultdir : str, default "../results"
        Directory where results should be stored.

    Returns
    -------
    ResultFile
        Instance of the ResultFile class populated with the new file details.
    """

    prefix = prefix.lower()
    ext = ext.removeprefix(".").lower()

    base_dir = Path(resultdir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    counter_file = base_dir / f"{prefix}.cnt"

    try:
        counter = int(counter_file.read_text().strip())
    except (FileNotFoundError, ValueError):
        counter = 0

    date_code = date.today().strftime("%Y_%m_%d")

    # Find the lowest free file number, starting at counter value
    while True:
        counter += 1
        filename = f"{prefix}_{date_code}_{counter:04d}.{ext}"
        path = base_dir / filename

        if not path.exists():
            break

    counter_file.write_text(str(counter))

    return ResultFile(prefix=prefix,
                      counter=counter,
                      ext=ext,
                      directory=str(base_dir),
                      name=filename,
                      path=str(path))


def plot_pulse(ax: plt.Axes | None = None,
               t: np.ndarray | None = None,
               y: np.ndarray | None = None,
               time_unit: str = "s",
               y_max: float | None = None) -> None:
    """Plot a pulse as a standardized time trace."""

    if t is None or y is None:
        raise ValueError("Both t and y must be provided.")

    if len(t) != len(y):
        raise ValueError("t and y must have the same length.")

    if ax is None:
        ax = plt.gca()

    multiplier, _ = find_timescale(time_unit)

    ax.plot(t * multiplier, y)
    ax.set(xlabel=f"Time [{time_unit}]", ylabel="Amplitude")
    ax.grid(True)

    if y_max is not None:
        ax.set_ylim(-y_max, y_max)

    return


def powerspectrum(y: np.ndarray, dt: float,
                  n_fft: int | None = None,
                  scale: str = "linear",
                  normalise: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the power spectrum of a pulse waveform.

    Computes the periodogram for a finite-length signal without applying
    additional windowing. Supports both 1D (single channel) and 2D
    (multi-channel) arrays where data points are in rows (dimension 0) and
    channels are in columns (dimension 1).

    Parameters
    ----------
    y : np.ndarray
        Time trace data. Can be 1D or 2D.
    dt : float
        Sample interval in seconds.
    n_fft : int, optional
        Number of points to use in the FFT. If None, the signal length is used.
    scale : str, default "linear"
        Scaling format for the spectrum: "linear" (power) or "dB".
    normalise : bool, default False
        If True, normalises the spectrum of each channel to its maximum value.

    Returns
    -------
    f : np.ndarray
        1D array of float representing the frequency vector.
    psd : np.ndarray
        1D or 2D array of float representing the power spectral density.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")

    f, psd = signal.periodogram(y, fs=1.0 / dt, nfft=n_fft,
                                detrend=False, scaling="density", axis=0)

    if normalise:
        max_vals = np.max(psd, axis=0, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        psd = psd / max_vals

    if scale.lower() == "db":
        psd = 10.0 * np.log10(np.maximum(psd, 1e-20))

    return f, psd


def plot_spectrum(t: np.ndarray, y: np.ndarray,
                  n_fft: int | None = None,
                  time_unit: str = "s",
                  y_max: float | None = None,
                  f_max: float | None = None,
                  db_min: float = -40.0,
                  scale: str = "dB",
                  normalise: bool = True,
                  ax: list[plt.Axes] | None = None):
    """Plot time trace and power spectrum in a standardised format.

    Requires evenly sampled data points.

    Parameters
    ----------
    t : np.ndarray
        1D array of float, time vector.
    y : np.ndarray
        1D or 2D array of float, time trace values.
    time_unit : str, default "s"
        Unit for the time axis, also determines frequency scale.
    y_max : float, optional
        Set symmetric y-axis limits for the time plot.
    f_max : float, optional
        Maximum frequency to plot in Hz
    n_fft : int, optional
        Number of points in FFT.
    scale : str, default "dB"
        Scaling format for the spectrum: "linear" (Power) or "dB".
    normalise : bool, default True
        If True, normalises the spectrum to 1.0 (or 0 dB)
    db_min : float, default -40.0
        Minimum relative value on the dB scale (dynamic range to show).
    ax : list of matplotlib.axes.Axes, optional
        List or array containing two axes objects: [ax_time, ax_freq].
        If None, a new figure with two subplots will be created.

        """
    if t.ndim != 1:
        raise ValueError("t must be a 1D array")

    if len(t) < 2:
        raise ValueError("At least two time samples are required")

    if y.shape[0] != len(t):
        raise ValueError("t and y must have matching lengths")

    dt = np.diff(t)
    if not np.allclose(dt, dt[0]):
        raise ValueError("Time vector must be evenly sampled")

    if ax is None:
        fig, ax = plt.subplots(2, 1, figsize=(10, 10), constrained_layout=True)

    # Plot time-domain pulse
    plot_pulse(ax[0], t, y, time_unit, y_max)

    # Calculate power spectrum (assumes even sampling)
    dt = float(t[1] - t[0])
    f, psd = powerspectrum(y, dt, n_fft=n_fft,
                           scale=scale, normalise=normalise)

    # Get scaling parameters for frequency axis
    multiplier, freq_unit = find_timescale(time_unit)
    f_limit = float(f.max()) if f_max is None else f_max

    if scale.lower() == "db":
        db_lim = np.array([db_min, 0.0])
        if not np.any(np.isnan(psd)):
            db_lim = float(psd.max()) + db_lim

        ax[1].set_ylim(db_lim)

        if normalise:
            spectrum_label = "Power [dB re. max]"
        else:
            spectrum_label = "Power [dB]"
    else:
        spectrum_label = "Power"

    # Plot frequency-domain spectrum
    ax[1].plot(f / multiplier, psd)
    ax[1].set(xlabel=f"Frequency [{freq_unit}]",
              xlim=(0, f_limit / multiplier),
              ylabel=spectrum_label)
    ax[1].grid(True)

    return
