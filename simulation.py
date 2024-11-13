from utils import ParticleSimulation

# Create a simulation instance
sim = ParticleSimulation(N=100, box_size=10, initial_temperature=300)

# Run the simulation
sim.run_simulation(num_steps=1000, dt=0.1)