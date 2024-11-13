import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Constants
N = 100  # Number of particles
box_size = 10  # Size of the box
h = 1  # Planck constant
dissipation_constant = 0.01  # 1% energy dissipation
initial_temperature = 10  # Initial temperature of the environment in Kelvin
k_B = 1.38e-23  # Boltzmann constant
particle_radius = 0.1  # Radius of the particles

# Initialize particle positions and velocities
positions = np.random.rand(N, 2) * box_size
velocities = np.random.normal(size=(N, 2))

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

# Function to handle collisions and energy dissipation
def handle_collisions(positions, velocities):
    collision_count = 0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < 2 * particle_radius:  # Check for collision
                collision_count += 1
                # Calculate collision angle
                collision_vector = positions[i] - positions[j]
                collision_angle = np.arctan2(collision_vector[1], collision_vector[0])
                
                # Rotate velocities to collision frame
                v1_rot = rotate(velocities[i], -collision_angle)
                v2_rot = rotate(velocities[j], -collision_angle)
                
                # Elastic collision in 1D
                v1_rot[0], v2_rot[0] = v2_rot[0], v1_rot[0]
                
                # Rotate velocities back to original frame
                velocities[i] = rotate(v1_rot, collision_angle)
                velocities[j] = rotate(v2_rot, collision_angle)
                
                # Energy dissipation
                velocities[i] *= (1 - dissipation_constant)
                velocities[j] *= (1 - dissipation_constant)
                
                # Adjust velocities based on environmental temperature
                thermal_velocity = np.sqrt(k_B * initial_temperature / h)
                velocities[i] += np.random.randn(2) * thermal_velocity * dissipation_constant
                velocities[j] += np.random.randn(2) * thermal_velocity * dissipation_constant
    return collision_count

# Function to rotate a vector by a given angle
def rotate(vector, angle):
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(rotation_matrix, vector)

# Function to calculate temperature
def calculate_temperature(velocities):
    kinetic_energy = 0.5 * np.sum(velocities**2)
    temperature = (2 / (3 * N * k_B)) * kinetic_energy
    return temperature

# Initialize lists
entropy_list = []
temperature_list = []
collision_list = []
morphed_entropy_list = []

# Simulation parameters
num_steps = 1000
dt = 0.1

# Set up the figure and axis for animation
fig, ax = plt.subplots()
scat = ax.scatter(positions[:, 0], positions[:, 1])

# Function to update the animation
def animate(i):
    global positions, velocities
    positions = update_positions(positions, velocities, dt)
    collision_count = handle_collisions(positions, velocities)
    scat.set_offsets(positions)
    entropy = calculate_entropy(positions)
    temperature = calculate_temperature(velocities)
    morphed_entropy = entropy * (1 + dissipation_constant)
    
    entropy_list.append(entropy)
    temperature_list.append(temperature)
    collision_list.append(collision_count)
    morphed_entropy_list.append(morphed_entropy)
    
    return scat,

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=num_steps, interval=50, blit=True)

# Save the animation
ani.save('particle_simulation.mp4', writer='ffmpeg')

# Plot the entropy curve
plt.figure()
plt.plot(entropy_list, label='Entropy')
plt.plot(morphed_entropy_list, label='Morphed Entropy')
plt.xlabel('Time Step')
plt.ylabel('Entropy')
plt.title('Entropy vs Time')
plt.legend()
plt.savefig('entropy_curve.png')

# Plot the temperature curve
plt.figure()
plt.plot(temperature_list)
plt.xlabel('Time Step')
plt.ylabel('Temperature (K)')
plt.title('Temperature vs Time')
plt.savefig('temperature_curve.png')

# Plot the collision curve
plt.figure()
plt.plot(collision_list)
plt.xlabel('Time Step')
plt.ylabel('Number of Collisions')
plt.title('Collisions vs Time')
plt.savefig('collision_curve.png')

plt.show()