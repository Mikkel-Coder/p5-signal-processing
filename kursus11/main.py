import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter


N = 1_000 # number of samples
omega_hat = np.linspace(-np.pi, np.pi, N)

# Our frequency response in the frequency domain 
H = 0.2 - 0.3*np.exp(-1j*omega_hat)+0.1*np.exp(-2.j*omega_hat)

# calculate the magnitude and phase
magnitude = np.abs(H)
phase = np.angle(H)

# Set ticks for the plots
x_axis_ticks_positive = [0., np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2, 5*np.pi/8, 3*np.pi/4, 7*np.pi/8, np.pi]
x_axis_ticks_negative = [-np.pi/8, -np.pi/4, -3*np.pi/8, -np.pi/2, -5*np.pi/8, -3*np.pi/4, -7*np.pi/8, -np.pi]
x_axis_labels_positive = ['0', 'π/8', 'π/4', '3π/8', 'π/2', '5π/8', '3π/4', '7π/8', 'π']
x_axis_labels_negative = ['-π/8', '-π/4', '-3π/8', '-π/2', '-5π/8', '-3π/4', '-7π/8', '-π']

# Plot of the magnitude
ax1 = plt.subplot(2,1,1)
plt.title("Magnitude Response")
plt.plot(omega_hat, magnitude)
ax1.set_xticks(x_axis_ticks_negative+x_axis_ticks_positive, x_axis_labels_negative+x_axis_labels_positive)
plt.ylabel("|H(e^-jω)|")

# Plot of the pase
ax2 = plt.subplot(2,1,2)
plt.title("Phase Response")
plt.plot(omega_hat, phase)
ax2.set_xticks(x_axis_ticks_negative+x_axis_ticks_positive, x_axis_labels_negative+x_axis_labels_positive)
plt.ylabel("Phase [radians]")

# Show the plot
plt.tight_layout()
plt.show()
