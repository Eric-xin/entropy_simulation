import numpy as np
import matplotlib.pyplot as plt

# Parameters for the 2D box
Lx, Ly = 10, 10  # Dimensions of the box

# Number of particles
n = 20

# Random initial velocities in x and y directions for n particles
vx = np.random.uniform(-2.0, 2.0, n)
vy = np.random.uniform(-2.0, 2.0, n)

# Random initial positions within the box for n particles
x0 = np.random.uniform(0, Lx, n)
y0 = np.random.uniform(0, Ly, n)

# Time evolution parameters
t_max = 1000
dt = 0.1
times = np.arange(0, t_max, dt)

# Function to calculate the position at time t for n particles
def position(t, x0, y0, vx, vy, Lx, Ly):
    x_t = (x0 + vx * t) % Lx
    y_t = (y0 + vy * t) % Ly
    return x_t, y_t

# Calculate the average distance from the initial position over time for all particles
average_distances = np.zeros(len(times))
for i, t in enumerate(times):
    x_t, y_t = position(t, x0, y0, vx, vy, Lx, Ly)
    distances = np.sqrt((x_t - x0)**2 + (y_t - y0)**2)
    average_distances[i] = np.mean(distances)

# Plot the average recurrence
plt.plot(times, average_distances, label='Average Distance')

plt.xlabel('Time')
plt.ylabel('Average Distance from Initial Position')
plt.title('Average Poincaré Recurrence in 2D Particle System with Randomization')
plt.legend()
plt.show()