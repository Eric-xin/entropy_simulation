import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse

class Particle:
    def __init__(self, position, velocity, mass=1.0, radius=0.5):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.mass = mass
        self.radius = radius
        self.update_kinetic_energy_and_temperature()

    def update_kinetic_energy_and_temperature(self):
        self.kinetic_energy = 0.5 * self.mass * np.linalg.norm(self.velocity)**2
        self.temperature = self.kinetic_energy / (1.5 * 1.38e-23)  # Boltzmann constant

class Simulation:
    def __init__(self, n, box_size, initial_speed, particle_radius):
        self.n = n
        self.box_size = box_size
        self.initial_speed = initial_speed
        self.particle_radius = particle_radius
        self.particles = self.initialize_particles()
        self.avg_speed_data = []
        self.avg_temp_data = []
        self.entropy_data = []
        self.collision_freq_data = []

    def initialize_particles(self):
        particles = []
        for _ in range(self.n):
            position = np.random.rand(2) * self.box_size
            angle = np.random.rand() * 2 * np.pi
            velocity = self.initial_speed * np.array([np.cos(angle), np.sin(angle)])
            particles.append(Particle(position, velocity, radius=self.particle_radius))
        return particles

    def update_positions(self, dt):
        for particle in self.particles:
            particle.position += particle.velocity * dt
            # Handle wall collisions
            for i in range(2):
                if particle.position[i] - particle.radius <= 0 or particle.position[i] + particle.radius >= self.box_size:
                    particle.velocity[i] *= -1
                    particle.update_kinetic_energy_and_temperature()

    def handle_collisions(self):
        collision_count = 0
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles[i+1:], i+1):
                if np.linalg.norm(p1.position - p2.position) < p1.radius + p2.radius:
                    # Inelastic collision: exchange velocities
                    v1, v2 = p1.velocity, p2.velocity
                    p1.velocity, p2.velocity = v2, v1

                    # Introduce random fluctuations to simulate thermodynamic movement
                    fluctuation = np.random.normal(0, 0.1, size=2)
                    p1.velocity += fluctuation
                    p2.velocity -= fluctuation

                    # Update kinetic energy and temperature
                    p1.update_kinetic_energy_and_temperature()
                    p2.update_kinetic_energy_and_temperature()

                    collision_count += 1
        return collision_count

    def calculate_properties(self):
        avg_speed = np.mean([np.linalg.norm(p.velocity) for p in self.particles])
        avg_temperature = np.mean([p.temperature for p in self.particles])
        entropy = np.sum([p.temperature * np.log(p.temperature) for p in self.particles])
        return avg_speed, avg_temperature, entropy

    def animate(self, i, scat):
        self.update_positions(0.01)
        collision_count = self.handle_collisions()
        scat.set_offsets([p.position for p in self.particles])

        avg_speed, avg_temp, entropy = self.calculate_properties()
        self.avg_speed_data.append(avg_speed)
        self.avg_temp_data.append(avg_temp)
        self.entropy_data.append(entropy)
        self.collision_freq_data.append(collision_count)

        return scat,

    def run(self, steps):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.box_size)
        ax.set_ylim(0, self.box_size)
        max_length = self.box_size
        scat = ax.scatter([p.position[0] for p in self.particles], [p.position[1] for p in self.particles], s=(self.particle_radius / max_length * 1000)**2)

        cur_dir = os.getcwd()
        cur_dir = os.path.join(cur_dir, 'outputs')

        ani = animation.FuncAnimation(fig, self.animate, fargs=(scat,), frames=steps, interval=50, blit=True)
        ani.save(cur_dir + '/particles_simulation.mp4', writer=animation.FFMpegWriter(fps=20, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-b:v', '5000k']), dpi=100)

        # Plot average speed, temperature, and entropy curves
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(self.avg_speed_data)
        plt.title('Average Speed')

        plt.subplot(3, 1, 2)
        plt.plot(self.avg_temp_data)
        plt.title('Average Temperature')

        plt.subplot(3, 1, 3)
        plt.plot(self.entropy_data)
        plt.title('Entropy')

        plt.tight_layout()
        plt.savefig(cur_dir + '/average_speed_temperature_entropy.png')

        # Plot collision frequency curve
        plt.figure()
        plt.plot(self.collision_freq_data)
        plt.title('Collision Frequency')
        plt.savefig(cur_dir + '/collision_frequency.png')

def main():
    parser = argparse.ArgumentParser(description='Run particle simulation.')
    parser.add_argument('--n', type=int, default=100, help='Number of particles')
    parser.add_argument('--box_size', type=float, default=10.0, help='Size of the box')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps in the simulation')
    parser.add_argument('--initial_speed', type=float, default=1.0, help='Initial speed of particles')
    parser.add_argument('--particle_radius', type=float, default=0.5, help='Radius of particles')

    args = parser.parse_args()

    sim = Simulation(n=args.n, box_size=args.box_size, initial_speed=args.initial_speed, particle_radius=args.particle_radius)
    sim.run(steps=args.steps)

if __name__ == '__main__':
    main()