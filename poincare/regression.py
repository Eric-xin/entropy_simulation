import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# Define the Hamiltonian of the system (example: 2-level system)
H = np.array([[1, 0.1], [0.1, 1]])

# Define the initial state (example: superposition state)
psi_0 = np.array([1, 0]) / np.sqrt(2)

# Time evolution parameters
t_max = 100
dt = 0.1
times = np.arange(0, t_max, dt)

# Function to calculate the time-evolved state
def time_evolved_state(H, psi_0, t):
    U_t = expm(-1j * H * t)
    return np.dot(U_t, psi_0)

# Calculate the overlap with the initial state over time
overlaps = []
for t in times:
    psi_t = time_evolved_state(H, psi_0, t)
    overlap = np.abs(np.dot(np.conj(psi_0), psi_t))**2
    overlaps.append(overlap)

# Plot the recurrence
plt.plot(times, overlaps)
plt.xlabel('Time')
plt.ylabel('Overlap with initial state')
plt.title('Poincaré Recurrence in Quantum System')
plt.show()