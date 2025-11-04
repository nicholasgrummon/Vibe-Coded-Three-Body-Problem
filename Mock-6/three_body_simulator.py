import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os # Import the os module for file path operations

# === SIMULATION PARAMETERS (User-Configurable) ===
# Masses of the three bodies
M1 = 1.0
M2 = 10.0
M3 = 1.0

# Gravitational Constant (G)
G = 0.5

# Simulation Time Step (determines resolution and speed)
DT = 0.005

# Numerical Softening Factor (prevents division by zero/catastrophic forces on close approach)
# This stabilizes the chaotic dynamics without significantly altering the physics at large distances.
SOFTENING_EPSILON = 1

# Set the fixed range for both X and Y axes (Domain/Range: [-AXIS_LIMIT, AXIS_LIMIT])
AXIS_LIMIT = 7.5

# Number of steps per frame update (controls animation speed vs. calculation load)
STEPS_PER_FRAME = 10

# Maximum number of time steps to keep in the path history (for optimization)
HISTORY_LIMIT = 5000

# GIF generation parameters
OUTPUT_DIRECTORY = "outputs"
OUTPUT_FILENAME = "sim.gif"
FRAMES_TO_GENERATE = 200  # Total number of frames for the GIF (e.g., 2000 frames)
GIF_FPS = 50               # Frames per second for the saved GIF

# === CLASS DEFINITIONS ===

class Body:
    """Represents a single mass in the N-body system."""
    def __init__(self, name, mass, position, velocity=None):
        self.name = name
        self.mass = mass
        # Position vector (x, y)
        self.r = np.array(position, dtype=float)
        # Velocity vector (vx, vy). Assumes initial velocity is zero if not provided.
        self.v = np.array(velocity, dtype=float) if velocity is not None else np.array([0.0, 0.0])
        # To store the historical path
        self.history = [self.r.copy()]

    def update_state(self, total_force, dt):
        """Updates position and velocity based on net force and time step (Semi-implicit Euler)."""
        # 1. Calculate acceleration (a = F / m)
        acceleration = total_force / self.mass

        # 2. Update velocity (v_new = v_old + a * dt)
        self.v += acceleration * dt

        # 3. Update position (r_new = r_old + v_new * dt)
        self.r += self.v * dt

        # 4. Record new position
        self.history.append(self.r.copy())
        # Keep history list manageable
        if len(self.history) > HISTORY_LIMIT:
            self.history.pop(0)

class ThreeBodySimulator:
    """Manages the simulation state and dynamics of the three bodies."""
    def __init__(self, body1, body2, body3, G_const):
        self.bodies = [body1, body2, body3]
        self.G = G_const
        self.time = 0.0

    def calculate_force_on(self, body_i, body_j):
        """Calculates the gravitational force vector exerted by body_j on body_i."""
        # Vector from body_i to body_j
        r_vector = body_j.r - body_i.r

        # Distance between the bodies
        r = np.linalg.norm(r_vector)
        
        # Softening factor squared
        epsilon_squared = SOFTENING_EPSILON**2
        
        # Denominator for the force magnitude: F = G * m1 * m2 / (r^2 + epsilon^2)
        r_squared_softened = r**2 + epsilon_squared

        # Gravitational force magnitude
        magnitude = (self.G * body_i.mass * body_j.mass) / r_squared_softened

        # If r is zero (though softening should prevent singularities, this is a fail-safe)
        if r == 0:
            return np.array([0.0, 0.0])

        # Force direction: a unit vector in the direction of r_vector
        direction = r_vector / r

        # Force vector: F = magnitude * direction
        force_vector = magnitude * direction

        return force_vector

    def step(self, dt):
        """Performs one simulation step."""
        forces = []
        for i in range(3):
            net_force = np.array([0.0, 0.0])
            for j in range(3):
                if i != j:
                    # Sum the force exerted by body j on body i
                    net_force += self.calculate_force_on(self.bodies[i], self.bodies[j])
            forces.append(net_force)

        # Update position and velocity of each body after calculating all forces
        for i in range(3):
            self.bodies[i].update_state(forces[i], dt)

        self.time += dt

# === INITIALIZATION AND RUNNING ===

# Define initial positions (Equilateral Triangle for a dynamic start)
pos1 = [1.0, 0.0]
pos2 = [-0.5, np.sqrt(3)/2]
pos3 = [-0.5, -np.sqrt(3)/2]

# Initial velocity is zero for all bodies, as requested
vel = [0.0, 0.0]

body_A = Body("A", M1, pos1, vel)
body_B = Body("B", M2, pos2, vel)
body_C = Body("C", M3, pos3, vel)

simulator = ThreeBodySimulator(body_A, body_B, body_C, G)

# --- MATPLOTLIB SETUP ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_title(f"3-Body Problem Simulation (G={G}, dt={DT}, $\\epsilon$={SOFTENING_EPSILON})")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_aspect('equal', adjustable='box')

# Initial plot limits (now fixed based on AXIS_LIMIT)
ax.set_xlim(-AXIS_LIMIT, AXIS_LIMIT)
ax.set_ylim(-AXIS_LIMIT, AXIS_LIMIT)

# Explicitly disable autoscaling to enforce fixed limits
ax.autoscale(False)

# Setup for the historical paths (lines)
path_A, = ax.plot([], [], 'r-', linewidth=0.5, alpha=0.7, label=f"Body A (M={M1})")
path_B, = ax.plot([], [], 'g-', linewidth=0.5, alpha=0.7, label=f"Body B (M={M2})")
path_C, = ax.plot([], [], 'b-', linewidth=0.5, alpha=0.7, label=f"Body C (M={M3})")

paths = [path_A, path_B, path_C]
colors = ['r', 'g', 'b']
markers = ['o', 's', '^']

# NEW: Setup for individual body markers (to allow separate colors/markers)
marker_A, = ax.plot(body_A.r[0], body_A.r[1], marker=markers[0], color=colors[0], markersize=10, linestyle='')
marker_B, = ax.plot(body_B.r[0], body_B.r[1], marker=markers[1], color=colors[1], markersize=10, linestyle='')
marker_C, = ax.plot(body_C.r[0], body_C.r[1], marker=markers[2], color=colors[2], markersize=10, linestyle='')
markers_list = [marker_A, marker_B, marker_C]

ax.legend()
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)


def init_plot():
    """Initialization function for the animation."""
    # Reset all data
    for marker in markers_list:
        marker.set_data([], [])
    for path in paths:
        path.set_data([], [])
    time_text.set_text('')
    return markers_list + paths + [time_text]


def update_plot(frame):
    """Update function called at each animation frame."""
    
    # 1. Step the simulation multiple times for faster apparent animation
    for _ in range(STEPS_PER_FRAME):
        simulator.step(DT)

    # 2. Update the current positions (markers)
    # Update each marker separately
    for i, body in enumerate(simulator.bodies):
        # FIX: Ensure coordinates are passed as sequences (lists) to set_data
        markers_list[i].set_data([body.r[0]], [body.r[1]])

    # 3. Update the historical paths (lines)
    for i, body in enumerate(simulator.bodies):
        history_array = np.array(body.history)
        paths[i].set_data(history_array[:, 0], history_array[:, 1])

    # 4. Update the time display
    time_text.set_text(f'Time: {simulator.time:.2f} s')
    
    # 5. Dynamic Plot Limits (REMOVED: Axes are now fixed by AXIS_LIMIT)
    # The commented out section below previously handled dynamic scaling.
    # all_x = np.concatenate([np.array(body.history)[:, 0] for body in simulator.bodies])
    # all_y = np.concatenate([np.array(body.history)[:, 1] for body in simulator.bodies])
    
    # # Only update limits if there is data
    # if all_x.size > 0 and all_y.size > 0:
    #     x_min, x_max = all_x.min(), all_x.max()
    #     y_min, y_max = all_y.min(), all_y.max()

    #     x_range = x_max - x_min
    #     y_range = y_max - y_min
    #     max_range = max(x_range, y_range) * 1.2 # Add 20% buffer

    #     x_center = (x_min + x_max) / 2
    #     y_center = (y_min + y_max) / 2

    #     ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    #     ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    return markers_list + paths + [time_text]


if __name__ == '__main__':
    # Create the animation
    # The interval defines the speed of the animation.
    ani = FuncAnimation(
        fig,
        update_plot,
        frames=FRAMES_TO_GENERATE, # Run for the defined number of frames
        init_func=init_plot,
        blit=False, # Must be False for saving the animation properly
        interval=10 # Interval in milliseconds between frames (10ms = 100 FPS)
    )
    
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print("Running 3-Body Problem Simulation...")
    print(f"G: {G}, dt: {DT}, Steps/Frame: {STEPS_PER_FRAME}, Softening: {SOFTENING_EPSILON}")
    print(f"Generating and saving animation to {output_path} (Total Frames: {FRAMES_TO_GENERATE}, GIF FPS: {GIF_FPS}). This may take a moment...")

    # Save the animation as a GIF using the 'pillow' writer
    # Note: 'pillow' requires the Pillow library (`pip install Pillow`)
    ani.save(output_path, writer='pillow', fps=GIF_FPS)

    print(f"Animation saved successfully to {output_path}")
