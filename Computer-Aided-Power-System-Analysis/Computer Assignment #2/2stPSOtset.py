import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Objective function
def J1(x):
    return (x[0]**2 + x[1]**2 - x[0]*x[1] - 10)*1e-3



# PSO Algorithm
def PSO(J, x0, n_particles, n_iter, w, c1, c2):
    # Initialization
    x_best = x0
    f_best = J(x_best)
    velocities = np.zeros((n_particles, 2))
    particles = np.random.uniform(-10, 10, (n_particles, 2))
    f_particles = np.apply_along_axis(J, 1, particles)
    best_positions = particles
    f_best_positions = f_particles

    # Animation function
    def animate(i):
        nonlocal velocities, particles, best_positions, f_best_positions, x_best, f_best

        r1, r2 = np.random.rand(n_particles, 2), np.random.rand(n_particles, 2)
        velocities = w * velocities + c1 * r1 * (best_positions - particles) + c2 * r2 * (x_best - particles)
        particles += velocities
        f_particles = np.apply_along_axis(J, 1, particles)

        # Update personal best
        mask = f_particles < f_best_positions
        best_positions[mask] = particles[mask]
        f_best_positions[mask] = f_particles[mask]

        # Update global best
        if np.min(f_particles) < f_best:
            x_best = particles[np.argmin(f_particles)]
            f_best = np.min(f_particles)

        # Create w values
        w_values = np.linspace(w, 0.5, n_iter)
        
        # Plotting animation
        plt.clf()
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.scatter(particles[:, 0], particles[:, 1], c='b')
        plt.scatter(x_best[0], x_best[1], c='r')
        plt.title(f'Iteration {i+1}, Best value: {f_best:.4f}, w={w_values[i]:.2f}')

    

    # Create animation
    fig = plt.figure()
    ani = animation.FuncAnimation(fig, animate, frames=n_iter, interval=100)

    # Show plot
    plt.show()

    return x_best, f_best

# Parameters
n_particles = 50
n_iter = 30
w = 0.75
c1 = 1.5
c2 = 1.5
x0 = np.array([0, 0])

# Run PSO
x_best, f_best = PSO(J1, x0, n_particles, n_iter, w, c1, c2)

# Print result
print(f'Best solution: {x_best}, Best value: {f_best:.4f}')
