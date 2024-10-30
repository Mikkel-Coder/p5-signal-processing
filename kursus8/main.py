from typing import Callable
from scipy import signal
from math import pi, sin
from numpy import abs, float64, complex128 
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
MAX_L: int = 100
MIN_L: int = 10


def find_magnitude_at_angular_frequency(
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
    for current_angular_frequency, current_magnitude in zip(w,h):

        # Check if we surpassed the target angular frequency
        if current_angular_frequency > target_angular_frequency:

            # If we have, then we must now determine which angular frequency 
            # is closest to the targeted angular frequency, current or previous
            if current_angular_frequency - target_angular_frequency < previous_angular_frequency - target_angular_frequency:
                closet_magnitude: complex128 = current_magnitude
            else:
                closet_magnitude: complex128 = previous_magnitude
            return abs(closet_magnitude)

        # If we have not surpassed the target angular frequency, then
        # continue to search staring from the current
        previous_angular_frequency = current_angular_frequency
        previous_magnitude = current_magnitude


def plotting(w: list[float], h: list[complex128]) -> None:
    """This function is not important. It is only used for plotting
    and saving the plot as a png file."""
    fig, ax = plt.subplots()

    # General plot configuring
    ax.plot(w, abs(h), color="blue", label="Angular Frequency Response")
    ax.set_title(f"LP-FIR filter using Rectangular Window with size L: {L}")
    ax.set_xlabel("Angular Frequency")
    ax.set_ylabel("Magnitude")
    x_axis_ticks = [0., pi/8, pi/4, 3*pi/8, pi/2, 5*pi/8, 3*pi/4, 7*pi/8, pi]
    x_axis_labels = ['0', 'π/8', 'π/4', '3π/8', 'π/2', '5π/8', '3π/4', '7π/8', 'π']
    ax.set_xticks(x_axis_ticks, x_axis_labels)

    # Plot the lines
    ax.axvline(cutoff_angular_frequency, linestyle="-.", color="grey", label="Ideal Frequency Cutoff")
    ax.axvline(passband_angular_frequency, linestyle="dashed", color="green", label="Frequency Passband")
    ax.axhline(passband_magnitude, linestyle="dotted", color="green", label="Magnitude Passband")
    ax.axvline(stopband_angular_frequency, linestyle="dashed", color="red", label="Frequency Stopband")
    ax.axhline(stopband_magnitude, linestyle="dotted", color="red", label="Magnitude Stopband")

    # Add text to each line 
    # x axis
    ax.text(passband_angular_frequency, -0.2, f"{passband_angular_frequency:.2f}π = 750 Hz", color='green', fontsize=7, ha='center', va='center')
    ax.text(stopband_angular_frequency, -0.2, f"{stopband_angular_frequency:.2f}π = 1500 Hz", color='red', fontsize=7, ha='center', va='center')

    # y axis
    ax.text(-0.2, passband_magnitude, f"{passband_magnitude:.2f} = -1 dB", color='green', fontsize=7, ha='right', va='center')
    ax.text(-0.2, stopband_magnitude, f"{stopband_magnitude:.2f} = -10 dB", color='red', fontsize=7, ha='right', va='center')

    # Add some padding so that we can see the text
    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20


    # Plot accepted pass band angular frequency and pass band magnitude area
    passband_rect = patches.Rectangle(
        (0, passband_magnitude),
        passband_angular_frequency,
        ax.get_ylim()[1] - passband_magnitude,
        edgecolor='none', facecolor='lightgreen', alpha=0.5
    )
    ax.add_patch(passband_rect)

    # Plot accepted stop band angular frequency and stop band magnitude area
    stopband_rect = patches.Rectangle(
        (stopband_angular_frequency, 0), 
        ax.get_xlim()[1] - stopband_angular_frequency,
        stopband_magnitude,
        edgecolor='none', facecolor='lightcoral', alpha=0.5
    )
    ax.add_patch(stopband_rect)

    ax.legend()
    fig.tight_layout()
    fig.savefig(f"./figures/{L}.png")
    plt.close(fig)

def rectangular_window(L: int, n: int) -> int:
    """We use a window function to make the SINC function
    finite and causal. Other window functions could also have been
    used, such as the Hamming window. (See in the book)"""
    if 0 <= n <= L -1:
        return 1
    else:
        return 0

def impulse_response(window: Callable, L: int, n: int):
    # Validate the index range
    if 0 > n or n > L - 1:
        return 0 

    # Compute the normalized time index 'x' based on the impulse response's center
    x = (n-(L-1)/2)

    # Handle the special case where 'x' is zero to avoid division by zero
    # The value of sinc(0) is defined as 1
    if x == 0: 
        return 1

    # Calculate the impulse response using the window function and the sinc function
    return window(L, n) * (sin(cutoff_angular_frequency*x))/(pi*x)

# Exclude the DC from SINC function
# This is done by not using odd L's
for L in range(MIN_L, MAX_L+1, 2):
    filter_coefficients = []
    for i in range(L):
        filter_coefficients.append(impulse_response(rectangular_window,L=L, n=i))

    # w: total number points (resolution=512) 
    # h: Complex frequency response
    w,h = signal.freqz(b=filter_coefficients, a=1, worN=512)
    high = find_magnitude_at_angular_frequency(passband_angular_frequency, w, h)
    low = find_magnitude_at_angular_frequency(stopband_angular_frequency, w, h)

    if high > passband_magnitude and low < stopband_magnitude:
        print(f"We found a good value for L: {L}")
        plotting(w,h)
