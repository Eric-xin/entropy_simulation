from calc import run_simulation

if __name__ == '__main__':
    # parameters: n, box_size, steps, initial_speed, particle_radius
    run_simulation(100, 100, 5000, 100, 2)
    # the dpi of animation is set to 100 by default. it can be changed in the run_simulation function