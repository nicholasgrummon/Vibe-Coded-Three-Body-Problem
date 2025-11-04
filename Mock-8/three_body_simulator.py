import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os # Import the os module for file path operations

# === SIMULATION PARAMETERS (User-Configurable) ===
# Masses of the three bodies
M1 = 1.0
M2 = 1.0
M3 = 1.0

# Gravitational Constant (G)
G = 0.5

# Simulation Time Step (determines resolution and speed)
DT = 0.005

# Numerical Softening Factor (prevents division by zero/catastrophic forces on close approach)
# This stabilizes the chaotic dynamics without significantly altering the physics at large distances.
SOFTENING_EPSILON = 0.05 

# Set the fixed range for both X and Y axes (Domain/Range: [-AXIS_LIMIT, AXIS_LIMIT])
AXIS_LIMIT = 7.5

# Number of steps per frame update (controls animation speed vs. calculation load)
STEPS_PER_FRAME = 10

# Maximum number of time steps to keep in the path history (for optimization)
HISTORY_LIMIT = 5000

# GIF generation parameters
OUTPUT_DIRECTORY = "outputs"
OUTPUT_FILENAME = "sim.gif"
FRAMES_TO_GENERATE = 2000  # Maximum number of frames for the GIF (safety limit)
GIF_FPS = 50               # Frames per second for the saved GIF

# === HEURISTICS FOR "INTERESTING" STARTING CONDITIONS ===
# Minimum triangle area formed by the three points (to prevent near-collinear start)
MIN_AREA_THRESHOLD = 0.5 
# Minimum separation distance between any two bodies
MIN_SEPARATION = 1.0 
MAX_CONFIG_TRIES = 1000 # Safety limit for finding a good starting configuration

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

# === DYNAMIC STOPPING LOGIC ===

def are_all_bodies_outside(simulator, limit):
    """Checks if all bodies have moved outside the fixed axis limit."""
    outside_count = 0
    for body in simulator.bodies:
        # Check if the body's x OR y coordinate is outside the limit
        # We use > limit for one axis and > limit for the other. This ensures
        # we stop only when they have truly left the square viewing window.
        if abs(body.r[0]) > limit or abs(body.r[1]) > limit:
            outside_count += 1
    return outside_count == len(simulator.bodies)

def frame_generator(simulator, max_frames, axis_limit, steps_per_frame, dt):
    """A generator that advances the simulation and yields frame indices until a stop condition is met."""
    frame_count = 0
    while frame_count < max_frames:
        # 1. Check stopping condition (Stop if all bodies are outside the frame)
        if frame_count > 0 and are_all_bodies_outside(simulator, axis_limit):
            print(f"\nSimulation terminated: All bodies left the fixed frame (Time: {simulator.time:.2f} s, Frames: {frame_count}).")
            return # Stop the generator
        
        # 2. Perform simulation steps
        for _ in range(steps_per_frame):
            simulator.step(dt)
            
        yield frame_count # Yield the frame index for update_plot
        frame_count += 1
        
    if frame_count == max_frames:
        print(f"\nSimulation reached maximum frame limit ({max_frames}).")

def is_interesting_config(p1, p2, p3):
    """Checks if the three positions satisfy criteria for an 'interesting' simulation start."""
    
    # Check 1: Minimum Separation
    r12 = np.linalg.norm(p1 - p2)
    r13 = np.linalg.norm(p1 - p3)
    r23 = np.linalg.norm(p2 - p3)
    
    if r12 < MIN_SEPARATION or r13 < MIN_SEPARATION or r23 < MIN_SEPARATION:
        return False
    
    # Check 2: Non-Collinear (Area check: Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|)
    # Area is 0.5 * | (x2 - x1)(y3 - y1) - (x3 - x1)(y2 - y1) |
    area = 0.5 * np.abs( (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]) )
    
    if area < MIN_AREA_THRESHOLD:
        return False
        
    return True

# === INITIALIZATION AND RUNNING ===

# Define initial positions (Randomized within the plot limits for a chaotic start)
limit = AXIS_LIMIT
vel = [0.0, 0.0]

# Regeneration loop until conditions are met
config_ok = False
tries = 0

while not config_ok and tries < MAX_CONFIG_TRIES:
    # Randomize positions within the limit and ensure they are numpy arrays
    pos1 = np.random.uniform(-limit, limit, 2)
    pos2 = np.random.uniform(-limit, limit, 2)
    pos3 = np.random.uniform(-limit, limit, 2)
    
    # Check the configuration
    config_ok = is_interesting_config(pos1, pos2, pos3)
    tries += 1
    
if tries == MAX_CONFIG_TRIES:
    print(f"Warning: Could not find an 'interesting' starting configuration within {MAX_CONFIG_TRIES} tries. Using last attempt.")
else:
    print(f"Initial configuration found after {tries} attempt(s).")
    
# Convert to Body objects after finding the correct configuration
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
    """Update function called at each animation frame.
    The simulation state is advanced by the frame_generator before this is called.
    """
    
    # 1. Update the current positions (markers)
    for i, body in enumerate(simulator.bodies):
        markers_list[i].set_data([body.r[0]], [body.r[1]])

    # 2. Update the historical paths (lines)
    for i, body in enumerate(simulator.bodies):
        history_array = np.array(body.history)
        paths[i].set_data(history_array[:, 0], history_array[:, 1])

    # 3. Update the time display
    time_text.set_text(f'Time: {simulator.time:.2f} s')
    
    return markers_list + paths + [time_text]


if __name__ == '__main__':
    # Initialize the generator that controls simulation stepping and stopping
    frame_gen = frame_generator(
        simulator, 
        FRAMES_TO_GENERATE, 
        AXIS_LIMIT, 
        STEPS_PER_FRAME,
        DT
    )
    
    # Create the animation, using the generator for dynamic frame control
    ani = FuncAnimation(
        fig,
        update_plot,
        frames=frame_gen,
        init_func=init_plot,
        blit=False, 
        interval=10 # Interval in milliseconds between frames
    )
    
    output_path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILENAME)

    # Create the output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    print("Running 3-Body Problem Simulation with Dynamic Stopping...")
    print(f"Heuristics: Min Separation={MIN_SEPARATION}, Min Area={MIN_AREA_THRESHOLD}")
    print(f"Fixed Axis Limit: +/- {AXIS_LIMIT}")
    print(f"Generating and saving animation to {output_path} (Max Frames: {FRAMES_TO_GENERATE}, GIF FPS: {GIF_FPS}). This may take a moment...")

    # Save the animation as a GIF using the 'pillow' writer
    # The animation will stop when the generator terminates
    ani.save(output_path, writer='pillow', fps=GIF_FPS)

    print(f"Animation saved successfully to {output_path}")
