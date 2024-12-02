from numpy import abs, angle, float64, complex128, linspace, real
from math import pi, sin, cos
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from parameters import (
    cutoff_angular_frequency,
    passband_angular_frequency,
    stopband_angular_frequency,
    passband_magnitude,
    stopband_magnitude,
)

freq_xaxis_ticks = [0., pi/4, pi/2, 3*pi/4, pi, 5*pi/4, 3*pi/2, 7*pi/4, 2*pi]
freq_xaxis_labels = ['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4', '2π']

phase_yaxis_ticks = [-pi, -3*pi/4, -pi/4, -pi/2, 0., pi/4, pi/2, 3*pi/4, pi]
phase_yaxis_labels = ['-π', '-3π/4', '-π/4', '-π/2', '0', 'π/4', 'π/2', '3π/4', 'π']

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

def plot_fun(
        w: list[float], 
        h: list[complex128],
        window_name: str,
        path: str,
        label: str,
        L
    ) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.plot(w, abs(h), angle(h), color="red")
    ax.set_title(f"{label}: (FUN) {window_name} Window size {L}")
    ax.set_xlabel("Discrete Angular Frequency [Radians/Sample]")
    ax.set_ylabel("Magnitude")
    ax.set_zlabel("Phase [Radians]")
    ax.set_xticks(freq_xaxis_ticks[::2], freq_xaxis_labels[::2])

    fig.savefig(f"{path}/{L}.svg")
    plt.close(fig)

def plot_phase(
        w: list[float],
        h: list[complex128],
        path: str,
        title: str,
        L
    ) -> None:
    fig, ax = plt.subplots()

    ax.plot(w, angle(h), color="blue")
    ax.set_title(title)
    ax.set_xlabel("Discrete Angular Frequency [Radians/Sample]")
    ax.set_ylabel("Phase [Radians]")
    ax.set_xticks(freq_xaxis_ticks, freq_xaxis_labels)
    ax.set_yticks(phase_yaxis_ticks, phase_yaxis_labels)

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    fig.tight_layout()
    fig.savefig(f"{path}/{L}.svg")
    plt.close(fig)

def plot_freq(
        w: list[float], 
        h: list[complex128],
        path: str,
        title: str,
        L: int = 0,
        lines: bool = False,
    )-> None:
    """This function is not important. It is only used for plotting
    and saving the plot as a png file."""
    fig, ax = plt.subplots()

    # General plot configuring
    ax.plot(w, abs(h), color="blue")
    ax.set_title(title)
    ax.set_xlabel("Diskret Vinkelfrekvens [Radianer per Sample]")
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
    fig.savefig(f"{path}/{L}.svg")
    plt.close(fig)


def plot_time_index(
        vals: list[complex128],
        path: str,
        res: int, 
        title: str,
        L: int = 0
    ) -> None:
    # Convert from frequency domain to time domain
    # We do that by using inverse DFT (by we use the optimized IDFT, IFFT)

    x = linspace(0., res, res)
    fig, ax = plt.subplots()

    ax.plot(x, real(vals[:res]), color="blue")
    ax.set_title(title)
    ax.set_xlabel("Tid [SampleIndex]")
    ax.set_ylabel("Magnitude")

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    fig.tight_layout()
    fig.savefig(f"{path}/{L}.svg")
    plt.close(fig)


def plot_time_secs(
        vals: list[complex128], 
        path_prefix: str, 
        sf: int,
        end_time: int, 
        title: str,
        L: int = 0
        ) -> None:

    # Calculate the resolution needed (num of samples)
    # (s * sample Hz = samples)
    res = end_time * sf

    # Make our x axsis based on our samples
    x = linspace(0., end_time, res)
    fig, ax = plt.subplots()

    # Only plot the values in the resolution "window"
    ax.plot(x, real(vals[:res]), color="blue")
    ax.set_title(title)
    ax.set_xlabel("Tid [Sekunder]")
    ax.set_ylabel("Magnitude")

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    fig.tight_layout()
    fig.savefig(f"{path_prefix}/{sf}/{L}.svg")
    plt.close(fig)


