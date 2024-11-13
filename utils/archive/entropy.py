# Simulate the entropy of a system of N particles in a box
# using the Boltzmann formula S = k ln W

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
N = 100  # Number of particles
box_size = 10  # Size of the box
h = 1  # Planck constant

# Initialize particle positions and velocities
positions = np.random.rand(N, 2) * box_size
velocities = np.random.randn(N, 2)

# Function to calculate entropy
def calculate_entropy(positions):
    hist, _ = np.histogramdd(positions, bins=(box_size, box_size))
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob))
    return entropy

# Function to update particle positions
def update_positions(positions, velocities, dt=0.1):
    positions += velocities * dt
    positions = np.mod(positions, box_size)  # Ensure particles stay within the box
    return positions

# Initialize entropy list
entropy_list = []

# Simulation parameters
num_steps = 1000
dt = 0.1

# Set up the figure and axis for animation
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1])

# Function to update the animation
def animate(i):
    global positions
    positions = update_positions(positions, velocities, dt)
    scat.set_offsets(positions)
    entropy = calculate_entropy(positions)
    entropy_list.append(entropy)
    return scat,

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=50, blit=True)

# Save the animation
ani.save('particle_simulation.mp4', writer='ffmpeg')

# Plot the entropy curve
plt.figure()
plt.plot(entropy_list)
plt.xlabel('Time step')
plt.ylabel('Entropy')
plt.title('Entropy vs Time')
plt.savefig('entropy_curve.png')
plt.show()