from typing import Callable
from scipy import signal
from math import pi, sin, cos
from numpy import abs, float64, complex128
from numpy.fft import fft
from os import mkdir
import matplotlib.pyplot as plt
# import matplotlib.patches as patches

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
MAX_L: int = 100
MIN_L: int = 10

# Our sample period
sample_period: float = 1/8000

# sample resolution
N = 51200


# Our input signal
def input_signal(n: int):
    return cos((pi/8)*n)+cos((3*pi/4)*n)


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


def _plot_freq(w: list[float], h: list[complex128],
               window_name: str, path: str, label: str) -> None:
    """This function is not important. It is only used for plotting
    and saving the plot as a png file."""
    fig, ax = plt.subplots()

    # General plot configuring
    ax.plot(w, abs(h), color="blue", label=label)
    ax.set_title(f"LP-FIR filter using {window_name} Window with size L: {L}")
    ax.set_xlabel("Angular Frequency")
    ax.set_ylabel("Magnitude")

    # Add some padding so that we can see the text
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{path}/{L}.png")
    plt.close(fig)

# def _plot_time(h: list[complex128], window_name: str):
#     """Please make this work later"""
#     # Convert from frequency domain to time domain
#     # We do that by using inverse DFT (by we use the optimized IDFT, IFFT)
#     y = ifft(h)
#     x = [i*sample_period for i in range(len(y))]
#     fig, ax = plt.subplots()
#
#     ax.plot(x, abs(y))
#
#     #ax.legend()
#     fig.tight_layout()
#     fig.savefig(f"./figures/{window_name}/{L}.png")
#     plt.close(fig)


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

# Convert out input sample to freq domain
time_input_signal = [input_signal(n) for n in range(N)]
freq_input_signal = fft(time_input_signal) / (N)

for window, window_name in windows:
    fig_save_location = f"{figures}/{window_name}"
    mkdir(fig_save_location)
    freq_path = fig_save_location + "/freq"
    freq_input_path = freq_path+"/input"
    freq_transferfunction_path = freq_path+"/transferfunction"
    freq_output_path = freq_path+"/output"
    mkdir(freq_path)
    mkdir(freq_input_path)
    mkdir(freq_transferfunction_path)
    mkdir(freq_output_path)

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

        # Påtryk vores signal via vores overføringsfunktion
        freq_output_signal = [a*b for a, b in zip(freq_input_signal, h)]

        if high > passband_magnitude and low < stopband_magnitude:
            print(f"[{window_name=}] We found a good value for L: {L}")
            _plot_freq(w, freq_input_signal,
                       window_name, freq_input_path, "Input frequencies")
            _plot_freq(w, h, window_name, freq_transferfunction_path,
                       "Angular Frequency Response")
            _plot_freq(w, freq_output_signal,
                       window_name, freq_output_path, "Filter output frequencies")
