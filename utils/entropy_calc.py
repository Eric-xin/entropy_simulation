import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt

class ParticleSimulation:
    def __init__(self, N=100, box_size=10, h=1, dissipation_constant=0.01, initial_temperature=300, k_B=1.38e-23, particle_radius=0.2):
        self.N = N
        self.box_size = box_size
        self.h = h
        self.dissipation_constant = dissipation_constant
        self.initial_temperature = initial_temperature
        self.k_B = k_B
        self.particle_radius = particle_radius

        # Initialize particle positions and velocities
        self.positions = np.random.rand(N, 2) * box_size
        # self.velocities = np.ones((N, 2))
        self.velocities = np.random.randn(N, 2)

        # Initialize lists for tracking simulation data
        self.entropy_list = []
        self.temperature_list = []
        self.collision_list = []
        self.morphed_entropy_list = []

    def calculate_entropy(self, temperature):
        return self.N * self.k_B * np.log(temperature)

    def update_positions(self, dt=0.1):
        self.positions += self.velocities * dt
        self.positions = np.mod(self.positions, self.box_size)  # Ensure particles stay within the box

    def handle_collisions(self):
        collision_count = 0
        total_energy_diffused = 0
        for i in range(self.N):
            for j in range(i + 1, self.N):
                dist = np.linalg.norm(self.positions[i] - self.positions[j])
                if dist < 2 * self.particle_radius:  # Check for collision
                    collision_count += 1
                    # Calculate collision angle
                    collision_vector = self.positions[i] - self.positions[j]
                    collision_angle = np.arctan2(collision_vector[1], collision_vector[0])
                    
                    # Rotate velocities to collision frame
                    v1_rot = self.rotate(self.velocities[i], -collision_angle)
                    v2_rot = self.rotate(self.velocities[j], -collision_angle)
                    
                    # Elastic collision in 1D
                    v1_rot[0], v2_rot[0] = v2_rot[0], v1_rot[0]
                    
                    # Rotate velocities back to original frame
                    self.velocities[i] = self.rotate(v1_rot, collision_angle)
                    self.velocities[j] = self.rotate(v2_rot, collision_angle)
                    
                    # Energy dissipation
                    energy_diffused = 0.5 * self.dissipation_constant * (np.sum(v1_rot**2) + np.sum(v2_rot**2))
                    total_energy_diffused += energy_diffused
                    self.velocities[i] *= (1 - self.dissipation_constant)
                    self.velocities[j] *= (1 - self.dissipation_constant)
                    
                    # Adjust velocities based on environmental temperature
                    thermal_velocity = np.sqrt(self.k_B * self.initial_temperature / self.h)
                    self.velocities[i] += np.random.randn(2) * thermal_velocity * self.dissipation_constant
                    self.velocities[j] += np.random.randn(2) * thermal_velocity * self.dissipation_constant

            # Handle collisions with the walls
            for dim in range(2):
                if self.positions[i, dim] < self.particle_radius or self.positions[i, dim] > self.box_size - self.particle_radius:
                    self.velocities[i, dim] *= -1  # Reverse velocity component

        return collision_count, total_energy_diffused

    def rotate(self, vector, angle):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return np.dot(rotation_matrix, vector)

    def update_temperature(self, temperature, dissipated_energy):
        temperature += dissipated_energy / (self.N * self.k_B)
        return temperature

    def animate(self, i):
        self.update_positions()
        collision_count, total_energy_diffused = self.handle_collisions()
        self.scat.set_offsets(self.positions)
        self.scat.set_sizes([self.particle_radius * 100] * self.N)  # Set the dot size
        
        # Calculate temperature based on energy diffused
        temperature = self.update_temperature(self.initial_temperature, total_energy_diffused)
        entropy = self.calculate_entropy(temperature)
        morphed_entropy = entropy * (1 + self.dissipation_constant)
        
        self.entropy_list.append(entropy)
        self.temperature_list.append(temperature)
        self.collision_list.append(collision_count)
        self.morphed_entropy_list.append(morphed_entropy)
        
        return self.scat,

    def run_simulation(self, num_steps=100, dt=0.1):
        self.dt = dt
        fig, ax = plt.subplots()
        self.scat = ax.scatter(self.positions[:, 0], self.positions[:, 1])
        
        ani = animation.FuncAnimation(fig, self.animate, frames=num_steps, interval=50, blit=True)
        ani.save('particle_simulation.mp4', writer='ffmpeg')

        # Plot the entropy curve
        plt.figure()
        plt.plot(self.entropy_list, label='Entropy')
        plt.plot(self.morphed_entropy_list, label='Morphed Entropy')
        plt.xlabel('Time Step')
        plt.ylabel('Entropy')
        plt.title('Entropy vs Time')
        plt.legend()
        plt.savefig('entropy_curve.png')

        # Plot the temperature curve
        plt.figure()
        plt.plot(self.temperature_list)
        plt.xlabel('Time Step')
        plt.ylabel('Temperature (K)')
        plt.title('Temperature vs Time')
        plt.savefig('temperature_curve.png')

        # Plot the collision curve
        plt.figure()
        plt.plot(self.collision_list)
        plt.xlabel('Time Step')
        plt.ylabel('Number of Collisions')
        plt.title('Collisions vs Time')
        plt.savefig('collision_curve.png')

        # plt.show()

# Usage example
if __name__ == "__main__":
    sim = ParticleSimulation()
    sim.run_simulation()