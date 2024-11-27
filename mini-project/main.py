from typing import Callable
from scipy import signal
from math import pi, sin, cos
from numpy import abs, angle, float64, complex128, linspace, real
from numpy.fft import fft, ifft
from os import mkdir
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Our know angular frequencies
cutoff_angular_frequency: float = pi/4
passband_angular_frequency: float = 3*pi/16
stopband_angular_frequency: float = 3*pi/8

# Our clean magnitudes for our pass- and stopband.
# This is used to determine if we found a good ?
passband_magnitude: float = 10**(-1/20)
stopband_magnitude: float = 10**(-1/2)

# The range of the size of our window function.
# We exclude smaller values of less than 10 because,
# then the DC in the SINC function becomes to "weighted"
MAX_L: int = 50
MIN_L: int = 10

# sample resolution
N = 5120

# To fix bug
L = 0

angular_frequencies = [ # Discrete [radians/sample]
    pi/80,
    3*pi/4,
]

sample_frequencies = [ # [Hz]
    10,
    100,
    1000,
]

# Our input signal
def input_signal(n: int):
    out: int = 0

    for freq in angular_frequencies:
        out += cos(freq*n)

    return out


def _find_magnitude_at_angular_frequency(
    target_angular_frequency: float,
    w: list[float64],
    h: list[complex128]
) -> float:
    """Since our angular frequency response is discrete, we are not
    able to find the precise magnitude for the continues frequency
    response. Instead we find the closet angular frequency, and returns
    its associated magnitude.
    """
    previous_angular_frequency: float64 = float64(0)
    previous_magnitude: complex128 = complex128(0)

    # Loop though each (angular_frequency, magnitude) pair
    for current_angular_frequency, current_magnitude in zip(w, h):

        # Check if we surpassed the target angular frequency
        if current_angular_frequency > target_angular_frequency:

            # If we have, then we must now determine which angular frequency
            # is closest to the targeted angular frequency, current or previous
            if (
                current_angular_frequency - target_angular_frequency <
                previous_angular_frequency - target_angular_frequency
            ):
                closet_magnitude: complex128 = current_magnitude
            else:
                closet_magnitude: complex128 = previous_magnitude
            return abs(closet_magnitude)

        # If we have not surpassed the target angular frequency, then
        # continue to search staring from the current
        previous_angular_frequency = current_angular_frequency
        previous_magnitude = current_magnitude

freq_xaxis_ticks = [0., pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4, 2*pi]
freq_xaxis_labels = ['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4', '2π']


def _plot_fun(w: list[float], h: list[complex128],
              window_name: str, path: str, label: str) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(w, abs(h), angle(h), color="red")
    ax.set_title(f"{label}: (FUN) {window_name} Window size {L}")
    ax.set_xlabel("Discrete Angular Frequency [Radians/Sample]")
    ax.set_ylabel("Magnitude")
    ax.set_zlabel("Phase [Radians]")
    ax.set_xticks(freq_xaxis_ticks[::2], freq_xaxis_labels[::2])

    fig.savefig(f"{path}/{L}.png")
    plt.close(fig)

def _plot_phase(w: list[float], h: list[complex128],
                window_name: str, path: str, label: str) -> None:
    fig, ax = plt.subplots()

    ax.plot(w, angle(h), color="blue")
    ax.set_title(f"{label}: (Phase) {window_name} Window size {L}")
    ax.set_xlabel("Discrete Angular Frequency [Radians/Sample]")
    ax.set_ylabel("Phase [Radians]")
    ax.set_xticks(freq_xaxis_ticks, freq_xaxis_labels)

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    fig.tight_layout()
    fig.savefig(f"{path}/{L}.png")
    plt.close(fig)

def _plot_freq(w: list[float], h: list[complex128],
               window_name: str, path: str, label: str, lines: bool = False) -> None:
    """This function is not important. It is only used for plotting
    and saving the plot as a png file."""
    fig, ax = plt.subplots()

    # General plot configuring
    ax.plot(w, abs(h), color="blue")
    title = ""
    if window_name != "": title = f"{window_name} Window size {L}"
    ax.set_title(f"{label}: (Frequency) {title}")
    ax.set_xlabel("Discrete Angular Frequency [Radians/Sample]")
    ax.set_ylabel("Magnitude")
    ax.set_xticks(freq_xaxis_ticks, freq_xaxis_labels)

    if lines:
        # lines and stuff
        # x
        ax.axvline(cutoff_angular_frequency, linestyle="-.", color="grey", label="Ideal Frequency Cutoff")
        ax.axvline(passband_angular_frequency, linestyle="dashed", color="green", label="Frequency Passband")
        ax.axvline(stopband_angular_frequency, linestyle="dashed", color="red", label="Frequency Stopband")
        
        # y
        ax.axhline(passband_magnitude, linestyle="dotted", color="green", label="Magnitude Passband")
        ax.axhline(stopband_magnitude, linestyle="dotted", color="red", label="Magnitude Stopband")

        # neg x
        ax.axvline(2*pi - cutoff_angular_frequency, linestyle="-.", color="grey", label="Ideal Frequency Cutoff")
        ax.axvline(2*pi - passband_angular_frequency, linestyle="dashed", color="green", label="Frequency Passband")
        ax.axvline(2*pi - stopband_angular_frequency, linestyle="dashed", color="red", label="Frequency Stopband")

        # Add text to each line 
        # x axis
        ax.text(passband_angular_frequency, -0.15, f"{passband_angular_frequency:.2f}π", color='green', fontsize=7, ha='center', va='center')
        ax.text(stopband_angular_frequency, -0.15, f"{stopband_angular_frequency:.2f}π", color='red', fontsize=7, ha='center', va='center')

        # Negative version
        ax.text(2*pi - passband_angular_frequency, -0.15, f"{2*pi - passband_angular_frequency:.2f}π", color='green', fontsize=7, ha='center', va='center')
        ax.text(2*pi - stopband_angular_frequency, -0.15, f"{2*pi - stopband_angular_frequency:.2f}π", color='red', fontsize=7, ha='center', va='center')

        # y axis
        ax.text(-0.5, passband_magnitude, f"{passband_magnitude:.2f} = -1 dB", color='green', fontsize=7, ha='right', va='center')
        ax.text(-0.5, stopband_magnitude, f"{stopband_magnitude:.2f} = -10 dB", color='red', fontsize=7, ha='right', va='center')

        passband_rect_pos = patches.Rectangle(
            (0, passband_magnitude),
            passband_angular_frequency,
            ax.get_ylim()[1] - passband_magnitude,
            edgecolor='none', facecolor='lightgreen', alpha=0.5
        )
        passband_rect_neg = patches.Rectangle(
            (2*pi - passband_angular_frequency, passband_magnitude),
            passband_angular_frequency,
            ax.get_ylim()[1] - passband_magnitude,
            edgecolor='none', facecolor='lightgreen', alpha=0.5
        )
        ax.add_patch(passband_rect_pos)
        ax.add_patch(passband_rect_neg)

        stopband_rect = patches.Rectangle(
            (stopband_angular_frequency, 0),
            2*pi - 2*stopband_angular_frequency,
            stopband_magnitude,
            edgecolor='none', facecolor='lightcoral', alpha=0.5
        )
        ax.add_patch(stopband_rect)

    # Add some padding so that we can see the text
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    fig.tight_layout()
    fig.savefig(f"{path}/{L}.png")
    plt.close(fig)

def _plot_time_index(vals: list[complex128], window_name: str,
               path: str, res: int, label: str) -> None:
    # Convert from frequency domain to time domain
    # We do that by using inverse DFT (by we use the optimized IDFT, IFFT)

    if(res > N):
        raise Exception("Too high resolution")

    x = linspace(0., res, res)
    fig, ax = plt.subplots()

    ax.plot(x, real(vals[:res]), color="blue")
    title = ""
    if window_name != "": title = f"{window_name} Window size {L}"
    ax.set_title(f"{label}: (Discrete Time) {title}")
    ax.set_xlabel("Time Index [n]")
    ax.set_ylabel("Magnitude")

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    fig.tight_layout()
    fig.savefig(f"{path}/{L}.png")
    plt.close(fig)

def _plot_time_real(vals: list[complex128], window_name: str,
                path_prefix: str, end_time: int, fs: int, label: str) -> None:
    
    res = end_time * fs

    if(res > N):
        raise Exception("Too high resolution")
    
    x = linspace(0., end_time, res)
    fig, ax = plt.subplots()

    ax.plot(x, real(vals[:res]), color="blue")
    title = ""
    if window_name != "": title = f"{window_name} Window size {L}"
    ax.set_title(f"{label}: (Time, fs: {fs}) {title}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Magnitude")

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    fig.tight_layout()
    fig.savefig(f"{path_prefix}/{fs}/{L}.png")
    plt.close(fig)

def rectangular_window(L: int, n: int) -> int:
    """We use a window function to make the SINC function
    finite and causal. Other window functions could also have been
    used, such as the Hamming window. (See in the book)"""
    if 0 <= n <= L - 1:
        return 1
    else:
        return 0


def _hann_function(L, n, a) -> float:
    return a-(1-a)*cos(2*pi*n/L)


def hann_window(L: int, n: int) -> float:
    # https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
    a = 0.5
    return _hann_function(L, n, a)


def hamming_window(L: int, n: int) -> float:
    # https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows
    a = 0.53836
    return _hann_function(L, n, a)


def blackman_window(L: int, n: int) -> float:
    # https://en.wikipedia.org/wiki/Window_function#Blackman_window
    a = 0.16
    a0 = (1-a)/2
    a1 = 0.5
    a2 = a/2
    return a0-a1*cos(2*pi*n/L)+a2*cos(4*pi*n/L)


def impulse_response(window: Callable, L: int, n: int):
    # Validate the index range
    if 0 > n or n > L - 1:
        return 0

    # Compute the normalized time index 'x' based on
    # the impulse response's center
    x = (n-(L-1)/2)

    # Handle the special case where 'x' is zero to
    # avoid division by zero
    # The value of sinc(0) is defined as 1
    if x == 0:
        return 1

    # Calculate the impulse response using the window function
    # and the sinc function
    return window(L, n) * (sin(cutoff_angular_frequency*x))/(pi*x)

# Exclude the DC from SINC function
# This is done by not using odd L's


windows = [
    (blackman_window, "Blackman"),
    (hamming_window, "Hamming"),
    (hann_window, "Hann"),
    (rectangular_window, "Rectangular")
]

figures = "./figures"
mkdir(figures)
input_path = figures + "/input"
mkdir(input_path)

input_freq_path = input_path + "/freq"
input_time_path = input_path + "/time"
input_time_index_path = input_time_path + "/index"
input_time_real_path = input_time_path + "/real"
mkdir(input_freq_path)
mkdir(input_time_path)
mkdir(input_time_index_path)
mkdir(input_time_real_path)
for fs in sample_frequencies:
    mkdir(input_time_real_path + f"/{fs}")

# Convert out input sample to freq domain
time_input_signal = [input_signal(n) for n in range(N)]
freq_input_signal = fft(time_input_signal) / (N)


_plot_freq(linspace(0., 2*pi, N), freq_input_signal, "", input_freq_path,
           "Input")
_plot_time_index(time_input_signal, "", input_time_index_path, N//10,
           "Input")
for fs in sample_frequencies:
    _plot_time_real(time_input_signal, "", input_time_real_path, 5, fs,
               "Input")


for window, window_name in windows:
    window_path = f"{figures}/{window_name}"
    mkdir(window_path)

    window_fun_path = window_path + "/fun"
    window_phase_path = window_path + "/phase"
    window_phase_transferfunction_path = window_phase_path + "/transferfunction"
    window_freq_path = window_path + "/freq"
    window_freq_transferfunction_path = window_freq_path + "/transferfunction"
    window_freq_output_path = window_freq_path + "/output"
    window_time_path = window_path + "/time"
    window_time_index_path = window_time_path + "/index"
    window_time_index_output_path = window_time_index_path + "/output"
    window_time_real_path = window_time_path + "/real"
    window_time_real_output_path = window_time_real_path + "/output"
    mkdir(window_fun_path)
    mkdir(window_phase_path)
    mkdir(window_phase_transferfunction_path)
    mkdir(window_freq_path)
    mkdir(window_freq_transferfunction_path)
    mkdir(window_freq_output_path)
    mkdir(window_time_path)
    mkdir(window_time_index_path)
    mkdir(window_time_index_output_path)
    mkdir(window_time_real_path)
    mkdir(window_time_real_output_path)
    for fs in sample_frequencies:
        mkdir(window_time_real_output_path + f"/{fs}")

    for L in range(MIN_L, MAX_L+1, 2):
        filter_coefficients = []
        for i in range(L):
            filter_coefficients.append(impulse_response(window, L=L, n=i))

        # w: total number points (aka resolution)
        # h: Complex frequency response
        w, h = signal.freqz(b=filter_coefficients, a=1, worN=N, whole=True)
        high = _find_magnitude_at_angular_frequency(passband_angular_frequency,
                                                    w, h)
        low = _find_magnitude_at_angular_frequency(stopband_angular_frequency,
                                                   w, h)
        
        if high > passband_magnitude and low < stopband_magnitude:
            print(f"[{window_name=}] We found a good value for L: {L}")

            # Påtryk vores signal via vores overføringsfunktion
            freq_output_signal = [a*b for a, b in zip(freq_input_signal, h)]

            # Find timedomain of output
            time_output_signal = ifft(freq_output_signal) * (N)

            _plot_fun(w, h, window_name, window_fun_path, "Filter")
            _plot_phase(w, h, window_name, window_phase_transferfunction_path, "Filter")
            _plot_freq(w, h, window_name, window_freq_transferfunction_path,
                       "Filter", lines=True)
            _plot_freq(w, freq_output_signal, window_name, window_freq_output_path,
                       "Output")
            _plot_time_index(time_output_signal, window_name, window_time_index_output_path, N//10,
                       "Output")
            for fs in sample_frequencies:
                _plot_time_real(time_output_signal, window_name, window_time_real_output_path, 5, fs,
                           "Output")