import numpy as np
import matplotlib.pyplot as plt

# Constants
num_particles = 100       # Number of particles
box_size = 10.0           # Size of the box (10x10 units)
time_steps = 1000         # Number of time steps
dt = 0.01                 # Time step duration
k_B = 1.0                 # Boltzmann constant (arbitrary units)
mass = 1.0                # Mass of each particle
inelasticity_factor = 0.9 # Factor for inelastic collisions (1.0 = elastic)

# Initialize positions and velocities
np.random.seed(0)
positions = np.random.rand(num_particles, 2) * box_size
angles = np.random.rand(num_particles) * 2 * np.pi
speeds = np.random.normal(loc=1.0, scale=0.5, size=num_particles)
velocities = np.column_stack((speeds * np.cos(angles), speeds * np.sin(angles)))

# Helper functions
def apply_periodic_boundary(positions, box_size):
    """Ensure particles stay within box by applying periodic boundaries."""
    return positions % box_size

def compute_entropy(velocities, positions, num_bins=50):
    """Estimate entropy based on both speed and spatial distribution."""
    speeds = np.linalg.norm(velocities, axis=1)
    # Velocity (speed) histogram for entropy
    speed_hist, speed_bins = np.histogram(speeds, bins=num_bins, density=True)
    speed_hist = speed_hist[speed_hist > 0]  # Ignore zero entries
    speed_entropy = -k_B * np.sum(speed_hist * np.log(speed_hist) * np.diff(speed_bins)[:len(speed_hist)])

    # Position histogram for spatial entropy
    pos_hist, xedges, yedges = np.histogram2d(positions[:, 0], positions[:, 1], bins=num_bins, density=True)
    pos_hist = pos_hist[pos_hist > 0]  # Ignore zero entries
    pos_entropy = -k_B * np.sum(pos_hist * np.log(pos_hist) * (xedges[1] - xedges[0]) * (yedges[1] - yedges[0]))

    # Total entropy as sum of speed and position entropies
    return speed_entropy + pos_entropy

# Simulation loop
entropy_values = []
for t in range(time_steps):
    # Update positions
    positions += velocities * dt
    positions = apply_periodic_boundary(positions, box_size)
    
    # Track entropy
    entropy = compute_entropy(velocities, positions)
    entropy_values.append(entropy)

    # Simple collision detection and response (inelastic collisions)
    for i in range(num_particles):
        for j in range(i + 1, num_particles):
            r_ij = positions[j] - positions[i]
            distance = np.linalg.norm(r_ij)
            if distance < 0.1:  # Assume particles collide if closer than 0.1 units
                # Reflect velocities with an inelastic factor
                v_i, v_j = velocities[i], velocities[j]
                n = r_ij / distance
                v_i_parallel = np.dot(v_i, n) * n
                v_j_parallel = np.dot(v_j, n) * n
                velocities[i] = v_i - inelasticity_factor * v_i_parallel + inelasticity_factor * v_j_parallel
                velocities[j] = v_j - inelasticity_factor * v_j_parallel + inelasticity_factor * v_i_parallel

# Plotting entropy over time
plt.figure(figsize=(10, 6))
plt.plot(entropy_values)
plt.xlabel('Time Step')
plt.ylabel('Entropy (arbitrary units)')
plt.title('Entropy of a 2D Gas Particle System with Inelastic Collisions')
plt.show()