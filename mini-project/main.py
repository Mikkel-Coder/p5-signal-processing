from os import mkdir
from math import pi, sin, cos
from numpy import linspace
from numpy.fft import fft, ifft
from scipy.signal import freqz
from utils import (
    find_magnitude_at_angular_frequency,
    plot_fun,
    plot_phase,
    plot_freq,
    plot_time_index,
    plot_time_secs,
)
from parameters import (
    angular_frequencies,
    cutoff_angular_frequency,
    sample_frequencies,
    passband_angular_frequency,
    stopband_angular_frequency,
    passband_magnitude,
    stopband_magnitude,
    MAX_L,
    MIN_L,
    N,
    end_time,
)


def input_signal(n: int):
    """Our input signal"""
    out: int = 0

    for freq in angular_frequencies:
        out += cos(freq * n)

    return out


def rectangular_window(L: int, n: int) -> int:
    """We use a window function to make the SINC function
    finite and causal. Other window functions could also have been
    used, such as the Hamming window. (See in the book)"""
    if 0 <= n <= L - 1:
        return 1
    else:
        return 0


def _hann_function(L, n, a) -> float:
    return a - (1 - a) * cos(2 * pi * n / L)


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
    a0 = (1 - a) / 2
    a1 = 0.5
    a2 = a / 2
    return a0 - a1 * cos(2 * pi * n / L) + a2 * cos(4 * pi * n / L)


def impulse_response(window, L: int, n: int):
    # Validate the index range
    if 0 > n or n > L - 1:
        return 0

    # Compute the normalized time index 'x' based on
    # the impulse response's center
    x = n - (L - 1) / 2

    # Handle the special case where 'x' is zero to
    # avoid division by zero
    # The value of sinc(0) is defined as 1
    if x == 0:
        return 1

    # Calculate the impulse response using the window function
    # and the sinc function
    return window(L, n) * (sin(cutoff_angular_frequency * x)) / (pi * x)


# All of our window function we want to try and plot with
windows = [
    (blackman_window, "Blackman"),
    (hamming_window, "Hamming"),
    (hann_window, "Hann"),
    (rectangular_window, "Rectangular"),
]

# Not very important. We create folders to save our figures to
figures = "./figures"
mkdir(figures)
input_path = figures + "/input"
mkdir(input_path)
input_freq_path = input_path + "/freq"
input_time_path = input_path + "/time"
input_time_index_path = input_time_path + "/index"
input_time_secs_path = input_time_path + "/secs"
mkdir(input_freq_path)
mkdir(input_time_path)
mkdir(input_time_index_path)
mkdir(input_time_secs_path)
for sf in sample_frequencies:
    mkdir(input_time_secs_path + f"/{sf}")

# We prefer to work in the frequency domain, so we must first
# convert out input signal in the time domain to freq domain
# This is done by using the DTF (but we use FFT because it's fast)
time_input_signal = [input_signal(n) for n in range(N)]
freq_input_signal = fft(time_input_signal) / (N)

# Plot how our input signal looks like in frequency and time domain
plot_freq(
    w=linspace(0.0, 2 * pi, N, endpoint=False),
    h=freq_input_signal,
    path=input_freq_path,
    title="Indgangs signalet x[n] i Diskret Frekvensdomænet"
)
plot_time_index(
    vals=time_input_signal,
    path=input_time_index_path,
    res=N // 100,
    title="Indgangs signalet x[n] i Diskret Tidsdomænet \n"
          "[Zoomet ind med *100]",
)
for sf in sample_frequencies:
    plot_time_secs(
        vals=time_input_signal,
        path_prefix=input_time_secs_path,
        sf=sf,
        end_time=end_time,
        title="Indgangs signalet x[n] i Tidsdomænet \n"
              f"[Sample frekvens {sf}]"
    )

# Now we want to examen each window in freq, phase and time
for window, window_name in windows:

    # Not important. We make the filestructure to save our figures to
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
    window_time_secs_path = window_time_path + "/secs"
    window_time_secs_output_path = window_time_secs_path + "/output"
    mkdir(window_fun_path)
    mkdir(window_phase_path)
    mkdir(window_phase_transferfunction_path)
    mkdir(window_freq_path)
    mkdir(window_freq_transferfunction_path)
    mkdir(window_freq_output_path)
    mkdir(window_time_path)
    mkdir(window_time_index_path)
    mkdir(window_time_index_output_path)
    mkdir(window_time_secs_path)
    mkdir(window_time_secs_output_path)
    for sf in sample_frequencies:
        mkdir(window_time_secs_output_path + f"/{sf}")

    # For each different window, we want to examen with "trail and error" want the good lengths (L) are for the windows
    # Note that our SINC (so the rectangular window) does not work for odd numbers. sorry
    for L in range(MIN_L, MAX_L + 1, 2):

        # As we want to examen what happens when we apply our signal
        # to the LP-FIR filter, we must first calculate the impulse response. i.e we want to "reveal" the LP-FIR filter so that we know exactly what happens. x[n]*h[n] = y[n] (see 6,10)
        filter_coefficients = []
        for i in range(L):
            filter_coefficients.append(impulse_response(window, L=L, n=i))

        # But as we prefer to work in the frequency domain
        # (as it is A LOT easier) we must calculate the frequency response
        # The frequency response returns a complex function, where the amplitude- and phase response can easily be calculated
        # (See 8,10) H(e^jώ)
        # w: total number points (aka resolution)
        # h: The complex frequency response
        w, h = freqz(
            b=filter_coefficients, 
            a=1,
            worN=N, # The returned length of samples [0,2pi)
            whole=True # By default the range is [0,pi), but we need [0,2pi)
        )

        # Find the accepted range that our pass- and stop bands can be within
        high = find_magnitude_at_angular_frequency(passband_angular_frequency, w, h)
        low = find_magnitude_at_angular_frequency(stopband_angular_frequency, w, h)

        # Check if the current window function with length L, is
        # acceptable for our requirements
        if high > passband_magnitude and low < stopband_magnitude:
            print(f"[{window_name=}] We found a good value for L: {L}")
            # If we found a good length, then we want to plot all of
            # its properties, so that we can look at it later

            # Apply our input signal to the LP-FIR filter using the
            # using the currently selected window function
            # Note that because we are in the frequency domain, 
            # we can multiply instead of convoluting in the time domain 
            freq_output_signal = [x_n * b_n for x_n, b_n in zip(freq_input_signal, h)]

            # As we are also asked, in the mini-project, we must
            # also plot the system in the time domain.
            # So we transfer "back" to the time domain by doing the 
            # inverse of the DTFT (aka IDFT)
            # (We use IFFT because it's fast)
            time_output_signal = ifft(freq_output_signal) * (N)

            # Now we want to plot:

            # Ignore plot_fun as it is not very useful
            plot_fun(w, h, window_name, window_fun_path, "Filter", L)

            # Plot the phase response
            plot_phase(
                w=w, 
                h=h, 
                path=window_phase_transferfunction_path, 
                title=  "LP-FIR filter Fase Respons \n"
                       f"med Vinduet: {window_name} og længde: {L}", 
                L=L
            )

            # Plot the LP-FIR filter
            plot_freq(
                w=w,
                h=h,
                path=window_freq_transferfunction_path,
                title=  "LP-FIR filter Magnitude Respons \n"
                       f"med Vinduet: {window_name} og længde: {L}",
                L=L,
                lines=True
            )

            # Plot the frequency response
            plot_freq(
                w=w, 
                h=freq_output_signal, 
                path=window_freq_output_path,
                L=L,
                title=  "Udgangs signalet y[n] i Diskret Tidsdomænet \n" 
                       f"med Vinduet: {window_name} og længde: {L}"
            )

            # Plot in the time domain (not that useful)
            plot_time_index(
                time_output_signal,
                window_time_index_output_path,
                N // 10,
                title=  "Udgangs signalet y[n] i Diskret Tidsdomænet \n" 
                       f"med Vinduet: {window_name} og længde: {L}. \n"
                        "[Zoomet ind med *10]",
                L=L,
            )

            for sf in sample_frequencies:
               plot_time_secs(
                   time_output_signal,
                   path_prefix=window_time_secs_output_path,
                   sf=sf,
                   end_time=end_time,
                   title="Output signalet y[n] i Tidsdomænet \n"
                         f"[Sample frekvens {sf}]",
                   L=L,
               )
