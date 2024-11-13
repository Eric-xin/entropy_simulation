import numpy as np
import matplotlib.pyplot as plt

# Parameters for the 2D box
Lx, Ly = 10, 10  # Dimensions of the box

# Random initial velocities in x and y directions
vx = np.random.uniform(-2.0, 2.0)
vy = np.random.uniform(-2.0, 2.0)

# Random initial position within the box
x0 = np.random.uniform(0, Lx)
y0 = np.random.uniform(0, Ly)

# Time evolution parameters
t_max = 500
dt = 0.01
times = np.arange(0, t_max, dt)

# Function to calculate the position at time t
def position(t, x0, y0, vx, vy, Lx, Ly):
    x_t = (x0 + vx * t) % Lx
    y_t = (y0 + vy * t) % Ly
    return x_t, y_t

# Calculate the distance from the initial position over time
distances = []
for t in times:
    x_t, y_t = position(t, x0, y0, vx, vy, Lx, Ly)
    distance = np.sqrt((x_t - x0)**2 + (y_t - y0)**2)
    distances.append(distance)

# Plot the recurrence
plt.plot(times, distances)
plt.xlabel('Time')
plt.ylabel('Distance from initial position')
plt.title('Poincaré Recurrence in 2D Particle System with Randomization')
plt.show()