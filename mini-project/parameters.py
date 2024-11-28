from math import pi

# Our know angular frequencies
cutoff_angular_frequency: float = pi/4
passband_angular_frequency: float = 3*pi/16
stopband_angular_frequency: float = 3*pi/8

# Our clean magnitudes for our pass- and stopband.
# This is used to determine if we found a good L(ength)
# for a LP-FIR filter
passband_magnitude: float = 10**(-1/20)
stopband_magnitude: float = 10**(-1/2)

# The range of the size of our window function.
# We exclude smaller values of less than 10 because,
# then the DC in the SINC function becomes to "weighted"
MAX_L: int = 50
MIN_L: int = 10

# Sample resolution
# So the number of samples in the range [0,2pi)
N = 8000

# Discrete angular frequencies that 
# our input signal contains measured in 
#[radians/sample]
angular_frequencies = [ 
    # Our low frequency
    pi/8 * 1/10, # We lower the ang freq so that it is easier to see in the plots 

    # Our high frequency
    3*pi/4,
]

# Our discrete sample period
# [Hz]
sample_frequencies = [
   8000,
   # 100, # If time we should perhaps try this  
   # 1000,
   # 10000,
]