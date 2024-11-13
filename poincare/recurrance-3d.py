import numpy as np
import matplotlib.pyplot as plt

# Parameters for the 3D box
Lx, Ly, Lz = 10, 10, 10  # Dimensions of the box

# Random initial velocities in x, y, and z directions
vx, vy, vz = np.random.uniform(-2, 2, 3)

# Random initial position within the box
x0, y0, z0 = np.random.uniform(0, 10, 3)

# Time evolution parameters
t_max = 1000
dt = 0.1
times = np.arange(0, t_max, dt)

# Function to calculate the position at time t
def position(t, x0, y0, z0, vx, vy, vz, Lx, Ly, Lz):
    x_t = (x0 + vx * t) % Lx
    y_t = (y0 + vy * t) % Ly
    z_t = (z0 + vz * t) % Lz
    return x_t, y_t, z_t

# Calculate the distance from the initial position over time
distances = []
for t in times:
    x_t, y_t, z_t = position(t, x0, y0, z0, vx, vy, vz, Lx, Ly, Lz)
    distance = np.sqrt((x_t - x0)**2 + (y_t - y0)**2 + (z_t - z0)**2)
    distances.append(distance)

# Plot the recurrence
plt.plot(times, distances)
plt.xlabel('Time')
plt.ylabel('Distance from initial position')
plt.title('Poincaré Recurrence in 3D Particle System with Randomization')
plt.show()