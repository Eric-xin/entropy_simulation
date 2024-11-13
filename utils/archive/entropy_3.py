# Simulate the entropy of a system of N particles in a box
# using the Boltzmann formula S = k ln W

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
N = 100  # Number of particles
box_size = 10  # Size of the box
h = 1  # Planck constant
dissipation_constant = 0.01  # 1% energy dissipation
temperature = 1  # Temperature of the environment in Kelvin
k_B = 1.38e-23  # Boltzmann constant

# Initialize particle positions and velocities
positions = np.random.rand(N, 2) * box_size
velocities = np.random.normal(size=(N, 2))

# Function to calculate entropy
def calculate_entropy(positions):
    hist, _ = np.histogramdd(positions, bins=(box_size, box_size), range=[[0, box_size], [0, box_size]])
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    entropy = -np.sum(prob * np.log(prob))
    return entropy

# Function to update particle positions
def update_positions(positions, velocities, dt=0.1):
    positions += velocities * dt
    positions = np.mod(positions, box_size)  # Ensure particles stay within the box
    return positions

# Function to handle collisions and energy dissipation
def handle_collisions(positions, velocities):
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 0.1:  # Assuming particles have a small radius
                # Elastic collision
                v1, v2 = velocities[i], velocities[j]
                velocities[i] = v1 - 2 * (np.dot(v1 - v2, positions[i] - positions[j]) / (np.linalg.norm(positions[i] - positions[j])**2 + 1e-10)) * (positions[i] - positions[j])
                velocities[j] = v2 - 2 * (np.dot(v2 - v1, positions[j] - positions[i]) / (np.linalg.norm(positions[j] - positions[i])**2 + 1e-10)) * (positions[j] - positions[i])
                
                # Energy dissipation
                velocities[i] *= (1 - dissipation_constant)
                velocities[j] *= (1 - dissipation_constant)
                
                # Adjust velocities based on environmental temperature
                thermal_velocity = np.sqrt(k_B * temperature / h)
                velocities[i] += np.random.normal(size=2) * thermal_velocity * dissipation_constant
                velocities[j] += np.random.normal(size=2) * thermal_velocity * dissipation_constant

# Initialize entropy list
entropy_list = []

# Simulation parameters
num_steps = 250
dt = 1000

# Set up the figure and axis for animation
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1])

# Function to update the animation
def animate(_):
    global positions, velocities
    positions = update_positions(positions, velocities, dt)
    handle_collisions(positions, velocities)
    scat.set_offsets(positions)
    entropy = calculate_entropy(positions)
    entropy_list.append(entropy)
    return scat,

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=50, blit=False)

# Save the animation
ani.save('particle_simulation.mp4', writer='ffmpeg')

# Plot the entropy curve
plt.figure()
plt.plot(entropy_list)
plt.xlabel('Time Step')
plt.ylabel('Entropy')
plt.title('Entropy vs Time')
plt.savefig('entropy_curve.png')
plt.show()