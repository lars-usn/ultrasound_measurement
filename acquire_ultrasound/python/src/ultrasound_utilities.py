"""Utility functions for ultrasound measurement systems at USN IMS.

Based on former systems in LabVIEW and Matlab

Lars Hoff, USN, Sep 2022
Modified July 2026
    - Follow PEP-8 and numpy docstring style guides.
    - Tested on Ubuntu
    - Genarel code cleanup
"""

from math import pi, radians, log10, floor, frexp
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from datetime import date
from pathlib import Path


# -----------------------------------------------------------------
# Wavefrom class
# -----------------------------------------------------------------
class Waveform:
    """Measurement results as 1D time-traces.

    Used to store traces sampled in time. Compatible with previous versions
    used in LabVIEW and MATLAB. Adapted from LabVIEW's waveform-type,
    similar to Python's mccdaq-library.


    Attributes
    ----------
    t0 : float
        Start time
    dt : float
        Sample interval
    dtr :float
        Interval between sample blocks. Rarely used
    y : 2D array of float
        Results. Each column is a channel, samples as rows

    Methods
    -------
    n_channels()
        Number of data channels.
    n_samples()
        Number of samples per channel.
    t()
        1D array of float time vector.
    fs()
        Sample rate.
    n_fft()
        Number of points used to calculate spectrum.
    f()
        1D array of float frequency vector.
    powerspectrum()
        1D array of float powerspectrum of traces.
    filtered()
        Bandpass filtered traces, all else equal.
    zoomed()
        Zoomed to specified interval, all else identical.
    plot()
        Plots result in figure.
    plot_spectrum()
        Plots traces and spectrum.
    save()
        Saves waveform to binary file.
    load()
        Loads waveform from binary file.
    """

    def __init__(self, y=None, dt=1.0, t0=0.0):
        """Initialize the waveform with data and sample parameters.

        Parameters
        ----------
        y : ndarray, optional
            Voltage traces as a numpy array. Rows are samples, columns are
            channels. If 1D, it will be reshaped to a 2D column vector.
            Defaults to a zero-filled array of shape (100, 1).
        dt : float, optional
            Sample interval in seconds. Defaults to 1.0.
        t0 : float, optional
            Time of first sample in seconds. Defaults to 0.0.

        """

        if y is None:
            self.y = np.zeros((100, 1))
        else:
            self.y = np.asarray(y)

        if self.y.ndim == 1:
            self.y = self.y.reshape((len(self.y), 1))

        self.dt = float(dt)
        self.t0 = float(t0)
        self.dtr = 0.0

    def n_channels(self):
        """Find number of data channels in trace."""
        return self.y.shape[1]

    def n_samples(self):
        """Find number of points in trace."""
        return self.y.shape[0]

    def t(self):
        """Calculate time vector from start time and sample interval [s]."""

        return np.linspace(self.t0,
                           self.t0 + self.dt * self.n_samples(),
                           self.n_samples())

    def fs(self):
        """Sample rate [Hz]."""
        return 1/self.dt

    def n_fft(self, upsample=0):
        """Set number of points used to calculate spectrum.

        Always a power of 2, zeros padded if needed.

        Parameters
        ----------
        upsample : int, optional
            Number of extra powers of 2 to add. Defaults to 0.

        Returns
        -------
        int
            Number of points for FFT, minimum 2048.
        """

        upsample = max(round(upsample), 0)
        m, e = frexp(self.n_samples())
        n = 2 ** (e + upsample)
        return max(n, 2048)

    def f(self):
        """Calculate frequency vector [Hz]."""
        return np.arange(0, self.n_fft() / 2) / self.n_fft() * self.fs()

    def powerspectrum(self, normalise=False, scale="linear", upsample=2):
        """Calculate power spectrum of time trace.

        Parameters
        ----------
        normalise : bool, optional
            Normalise to 1 (0 dB) as maximum. Defaults to False.
        scale : str, optional
            Scaling option, either "linear" or "dB". Defaults to "linear".
        upsample : int, optional
            Interpolate spectrum by padding to next power of 2. Defaults to 2.

        Returns
        -------
        f : ndarray
            1D frequency vector.
        psd : ndarray
            2D power spectral density array.
        """

        f, psd = powerspectrum(self.y, self.dt,
                               n_fft=self.n_fft(upsample=upsample),
                               scale=scale,
                               normalise=normalise)
        return f, psd

    def filtered(self, wave_filter):
        """Apply bandpass filter to trace.

        Parameters
        ----------
        wave_filter : WaveformFilter
            Filter specification object.

        Returns
        -------
        Waveform
            Copy of original waveform with filtered data.
        """

        filter_type = str(wave_filter.type).strip().lower()

        if filter_type.startswith("no"):
            y_filtered = self.y.copy()
        elif filter_type.startswith("ac"):
            y_filtered = self.y - self.y.mean(axis=0)
        else:
            b, a = wave_filter.coefficients()
            y_filtered = signal.filtfilt(b, a, self.y, axis=0)

        wfm = Waveform(y=y_filtered, dt=self.dt, t0=self.t0)
        return wfm

    def zoomed(self, tlim):
        """Extract copy of trace from interval specified by tlim.

        Parameters
        ----------
        tlim : array_like
            List or array containing start and end of interval to select.

        Returns
        -------
        Waveform
            Copy of original waveform zoomed to the specified interval.
        """
        time_vector = self.t()
        t_start, t_end = min(tlim), max(tlim)

        nlim = np.flatnonzero((time_vector >= t_start) &
                              (time_vector <= t_end))

        if nlim.size == 0:
            raise ValueError(
                f"No samples found within the time limits {
                    t_start} to {t_end}."
            )

        new_t0 = time_vector[nlim[0]]
        y_zoomed = self.y[nlim, :]

        wfm = Waveform(y=y_zoomed, dt=self.dt, t0=new_t0)
        return wfm

    def plot(self, time_unit="us", ch=None, y_max=None):
        """Plot time traces using specified time unit.

        Parameters
        ----------
        time_unit : str, optional
            Unit to plot time in ('s', 'ms', 'us'). Defaults to "us".
        ch : array_like, optional
            Channels to plot. Defaults to [0, 1] if not specified.
        y_max : float, optional
            Maximum scale on the amplitude axis. Defaults to None.

        Returns
        -------
        int
            Returns 0 upon successful execution.
        """

        if ch is None:
            ch = [0, 1]

        # CHECK
        plot_pulse(self.t(), self.y[:, ch], time_unit, y_max)
        # plot_pulse(self.t(), self.y[ch], time_unit, y_max)
        return 0

    def plot_spectrum(self, time_unit="s", ch=None, y_max=None, f_max=None,
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

        Returns
        -------
        int
            Returns 0 upon successful execution.
        """
        if ch is None:
            ch = [0, 1]
        plot_spectrum(self.t(), self.y[:, ch],
                      time_unit=time_unit,
                      y_max=y_max, f_max=f_max, n_fft=self.n_fft(),
                      normalise=normalise, scale=scale, db_min=db_min, ax=ax
                      )
        return 0

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
            fid.write(np.array(self.n_channels(), dtype='>u4').tobytes())
            fid.write(np.array(self.t0, dtype='>f8').tobytes())
            fid.write(np.array(self.dt, dtype='>f8').tobytes())
            fid.write(np.array(self.dtr, dtype='>f8').tobytes())
            fid.write(y_contiguous.tobytes())

        return 0

    def load(self, filename):
        """Load 'waveform' files from a binary file as 4-byte (sgl) floats.

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
        int
            Returns 0 upon successful completion.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        IOError
            If the file cannot be read.
        """
        with open(filename, "rb") as fid:
            # Read header length and the header string
            n_header = int(np.fromfile(fid, dtype=">i4", count=1)[0])
            header_bytes = fid.read(n_header)
            self.header = header_bytes.decode("utf-8")

            # Read channel configuration and time parameters
            n_ch = int(np.fromfile(fid, dtype=">u4", count=1)[0])
            self.t0 = float(np.fromfile(fid, dtype=">f8", count=1)[0])
            self.dt = float(np.fromfile(fid, dtype=">f8", count=1)[0])
            self.dtr = float(np.fromfile(fid, dtype=">f8", count=1)[0])

            # Read signal traces (2D array)
            y = np.fromfile(fid, dtype=">f4", count=-1)
            self.y = np.reshape(y, (-1, n_ch))

        self.sourcefile = filename
        return 0


# -----------------------------------------------------------------
# Generated signals: Pulse class
# -----------------------------------------------------------------
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

    def t(self) -> np.ndarray:
        """Create the time vector for the pulse.

        Returns
        -------
        np.ndarray
            1D array of float representing the time vector in seconds.
        """
        return np.arange(0, self.duration(), self.dt)

    def y(self) -> np.ndarray:
        """Create the pulse time trace from the input specification.

        Returns
        -------
        np.ndarray
            1D array of float representing the generated pulse waveform.
        """
        # Select the appropriate window/envelope
        envelope_type = self.envelope[0:3].lower()
        n_pts = self.n_samples()

        match envelope_type:
            case "rec":
                win = signal.windows.boxcar(n_pts)
            case "han":
                win = signal.windows.hann(n_pts)
            case "ham":
                win = signal.windows.hamming(n_pts)
            case "tri":
                win = signal.windows.triang(n_pts)
            case "tuk":
                win = signal.windows.tukey(n_pts, self.alpha)
            case _:
                win = signal.windows.boxcar(n_pts)

        phase_arg = 2 * pi * self.f0 * self.t() + radians(self.phase)

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

    def period(self) -> float:
        """Calculate the period of the carrier wave.

        Returns
        -------
        float
            Carrier wave period in seconds.
        """
        return 1.0 / self.f0

    def duration(self) -> float:
        """Calculate the total duration of the pulse.

        Returns
        -------
        float
            Pulse duration in seconds.
        """
        return self.period() * self.n_cycles

    def n_samples(self) -> int:
        """Find the number of samples in the pulse.

        Returns
        -------
        int
            Number of samples.
        """
        return len(self.t())

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

    def n_fft(self) -> int:
        """Set the number of points used to calculate the spectrum.

        The number is always a power of 2, using for zero-padding.

        Returns
        -------
        int
            Number of FFT points (minimum 2048).
        """
        # math.frexp splits a float into mantissa and exponent
        m, e = frexp(self.n_samples())
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
        f, psd = powerspectrum(y=self.y(),
                               dt=self.dt,
                               n_fft=self.n_fft(),
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
        plot_pulse(t=self.t(), y=self.y(), time_unit=self.time_unit())
        return 0

    def plot_spectrum(self) -> int:
        """Plot both the time trace and the power spectrum.

        Returns
        -------
        int
            Returns 0 upon successful execution.
        """
        plot_spectrum(t=self.t(),
                      y=self.y(),
                      time_unit=self.time_unit(),
                      f_max=scale_125(3*self.f0),
                      n_fft=self.n_fft(),
                      scale="db",
                      normalise=True)
        return 0


# -----------------------------------------------------------------
# Utility classes
# -----------------------------------------------------------------
class WaveformFilter:
    """Definition of a digital filter for the "Waveform" class.

    Attributes
    ----------
    type : str
        Type of filter: "No" (None), "AC", "BPF" (Bandpass).
    f_min : float
        Lower cutoff frequency in Hz.
    f_max : float
        Upper cutoff frequency in Hz.
    order : int
        Filter order.
    fs : float
        Sample rate in Hz.
    """

    # Class attributes with default values and type hints
    type: str = "No"
    f_min: float = 100e3
    f_max: float = 10e6
    order: int = 2
    fs: float = 100e6

    def fn(self) -> np.ndarray:
        """Return the cutoff frequencies normalised to the Nyquist frequency.

        Returns
        -------
        np.ndarray
            1D array of float containing the normalised lower and upper
            cutoff frequencies.
        """
        f_nyquist = self.fs / 2
        return np.array([self.f_min, self.f_max]) / f_nyquist

    def coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate filter coefficients (b, a) from the filter description.

        Determines the filter type (lowpass, highpass, or bandpass) from
        the cutoff frequencies and calculates the coefficients.

        Returns
        -------
        b : np.ndarray
            The numerator coefficient array of the filter.
        a : np.ndarray
            The denominator coefficient array of the filter.
        """
        fn_vals = self.fn()
        if fn_vals[0] <= 0:
            b, a = signal.butter(self.order, fn_vals[1],
                                 btype="lowpass", output="ba")
        elif fn_vals[1] > 0.5:
            b, a = signal.butter(self.order, fn_vals[0],
                                 btype="highpass", output="ba")
        else:
            b, a = signal.butter(self.order, fn_vals,
                                 btype="bandpass", output="ba")

        return b, a


class ResultFile:
    """Path, name, and counter configuration for a result file.

    Attributes
    ----------
    prefix : str
        Prefix for the file name.
    ext : str
        File extension (e.g., "trc").
    path : str
        Full path to the file.
    directory : str
        Directory where the file is stored.
    name : str
        Base name of the file.
    counter : int
        File counter or index.
    """

    # Class attributes with default values and type hints
    prefix: str = "test"
    ext: str = "trc"
    path: str = ""
    directory: str = ""
    name: str = ""
    counter: int = 0


# -----------------------------------------------------------------
# Utility classes
# -----------------------------------------------------------------
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

    prefixes = np.array([1, 2, 5, 10])
    magnitude = abs(x)

    exponent = int(floor(log10(magnitude)))
    mantissa = magnitude / (10**exponent)

    valid_indices = np.where(prefixes >= mantissa - 0.001)
    min_prefix = np.min(prefixes[valid_indices])

    return float(min_prefix * (10**exponent))


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
    match time_unit:
        case "ns":
            multiplier = 1e9
            freq_unit = "GHz"
        case "us":
            multiplier = 1e6
            freq_unit = "MHz"
        case "ms":
            multiplier = 1e3
            freq_unit = "kHz"
        case _:
            multiplier = 1.0
            freq_unit = "Hz"

    return multiplier, freq_unit


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
    min_value = float(np.min(limits))
    max_value = float(np.max(limits))

    # Ensure minimum difference
    max_value = max(max_value, min_value + min_diff)

    return np.array([min_value, max_value])


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

    Examples
    --------
    >>> read_scaled_value("3.4 MHz")
    3400000.0
    >>> read_scaled_value("100")
    100.0
    """
    # Split by any whitespace and remove extra padding
    parts = quantity.strip().split()

    if not parts:
        return 0.0

    number = float(parts[0])

    if len(parts) == 1:
        return number

    prefix = parts[1][0]
    match prefix:
        case "u":
            multiplier = 1e-6
        case "m":
            multiplier = 1e-3
        case "k":
            multiplier = 1e3
        case "M":
            multiplier = 1e6
        case "G":
            multiplier = 1e9
        case _:
            multiplier = 1.0

    return number * multiplier


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
    resultfile = ResultFile()
    prefix = prefix.lower()
    ext = ext.lower().split(".")[-1]

    base_dir = Path(resultdir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    counter_file = base_dir / f"{prefix}.cnt"

    # Read existing counter or start at 0
    if counter_file.is_file():
        try:
            counter = int(counter_file.read_text().strip())
        except ValueError:
            counter = 0
    else:
        counter = 0

    date_code = date.today().strftime("%Y_%m_%d")

    # Find the lowest free file number
    while True:
        counter += 1
        filename = f"{prefix}_{date_code}_{counter:04d}.{ext}"
        result_path = base_dir / filename

        if not result_path.is_file():
            break

    # Save the updated counter back to the file
    counter_file.write_text(str(counter))

    # Populate resultFile
    resultfile.prefix = prefix
    resultfile.counter = counter
    resultfile.ext = ext
    resultfile.directory = str(base_dir)
    resultfile.name = filename
    resultfile.path = str(result_path)

    return resultfile


def plot_pulse(ax: plt.Axes | None = None,
               t: np.ndarray = None,
               y: np.ndarray = None,
               time_unit: str = "s",
               y_max: float | None = None) -> int:
    """Plot a pulse as a time-trace in a standardised graph."""
    if ax is None:
        ax = plt.gca()

    multiplier, freq_unit = find_timescale(time_unit)
    ax.plot(t * multiplier, y)
    ax.set(xlabel=f"Time [{time_unit}]", ylabel="Amplitude")
    ax.grid(True)

    if y_max is not None:
        ax.set_ylim(y_max * np.array([-1.0, 1.0]))

    return 0


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
    # SciPy's periodogram calculates along axis=-1
    y_transposed = y.transpose()
    f, psd_transposed = signal.periodogram(y_transposed, fs=1.0 / dt,
                                           nfft=n_fft,
                                           detrend=False)

    psd = psd_transposed.transpose()

    if normalise:
        max_vals = psd.max(axis=0, keepdims=True)
        max_vals[max_vals == 0] = 1.0
        psd = psd / max_vals

    if scale.lower() == "db":
        psd = 10.0 * np.log10(np.maximum(psd, 1e-20))

    return f, psd


def plot_spectrum(t: np.ndarray, x: np.ndarray,
                  n_fft: int | None = None,
                  time_unit: str = "s",
                  y_max: float | None = None,
                  f_max: float | None = None,
                  db_min: float = -40.0,
                  scale: str = "dB",
                  normalise: bool = True,
                  ax: list[plt.Axes] | None = None) -> int:
    """Plot time trace and power spectrum in a standardised format.

    Requires evenly sampled data points.

    Parameters
    ----------
    t : np.ndarray
        1D array of float, time vector.
    x : np.ndarray
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

    Returns
    -------
    int
        Returns 0 upon successful execution.
    """
    # Create figure and subplots if axes are not provided
    if ax is None:
        fig = plt.figure(figsize=[10, 10])
        ax = [fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2)]

    # Plot the time-domain pulse
    plot_pulse(ax[0], t, x, time_unit, y_max)

    # Calculate the power spectrum (assumes even sampling)
    dt = float(t[1] - t[0])
    f, psd = powerspectrum(x, dt, n_fft=n_fft,
                           scale=scale, normalise=normalise)

    # Get scaling parameters for the frequency axis
    multiplier, freq_unit = find_timescale(time_unit)

    if f_max is None:
        f_max = float(f.max())

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
              xlim=(0, f_max / multiplier),
              ylabel=spectrum_label)
    ax[1].grid(True)

    return 0
