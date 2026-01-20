import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dem.particle import Particle
from dem.boundary import Boundary
from dem.solver import DEMSolver
from dem.contact_model import get_contact_model
from dem.time_integration import get_time_integration_method

tum_blue = "#0065BD"

def plot_velocity(ax, times, velocities, particle_colors):
    """
    Sets up the velocity subplot (Magnitude vs Time).
    """

    ax.set_title("Velocity vs Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Vel [m/s]")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    lines = []
    dots = []
    
    # Auto-scale logic
    if velocities:
        all_vels = np.concatenate([np.linalg.norm(v, axis=1) for v in velocities])
        all_times = np.concatenate(times)
        max_v = np.max(all_vels) if len(all_vels) > 0 else 1.0
        ax.set_xlim(np.min(all_times), np.max(all_times))
        ax.set_ylim(0, max_v * 1.1 + 0.1) 
    
    for i, (t, v) in enumerate(zip(times, velocities)):
        color = tum_blue
        line, = ax.plot([], [], color=color, linewidth=1.5, label=f'P{i+1}')
        dot, = ax.plot([], [], 'o', color=color)
        lines.append(line)
        dots.append(dot)
        
    ax.legend(loc='upper right', fontsize='small')
    return lines, dots

def plot_force(ax, times, forces, particle_colors):
    """
    Sets up the force subplot (Magnitude vs Time).
    """
    ax.set_title("Force vs Time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.grid(True, linestyle='--', alpha=0.5)
    
    lines = []
    dots = []
    
    # Auto-scale logic
    if forces:
        all_forces = np.concatenate([np.linalg.norm(f, axis=1) for f in forces])
        all_times = np.concatenate(times)
        max_f = np.max(all_forces) if len(all_forces) > 0 else 1.0
        if max_f == 0: max_f = 1.0
        ax.set_xlim(np.min(all_times), np.max(all_times))
        ax.set_ylim(0, max_f * 1.1 + 0.1)
    
    for i, (t, f) in enumerate(zip(times, forces)):
        color = tum_blue
        line, = ax.plot([], [], color=color, linewidth=1.5, label=f'P{i+1}')
        dot, = ax.plot([], [], 'o', color=color)
        lines.append(line)
        dots.append(dot)
        
    return lines, dots

def plot_field(boundaries, particles, filename):
    """
    Main visualization function: Generates a GIF with Field, Velocity, and Force subplots.
    """
    
    # --- Directory Handling ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "Plot_Results")
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    output_path = os.path.join(results_dir, filename)
    print(f"Preparing animation: {output_path}")

    # --- Data Extraction ---
    times = [np.array(p.history_time) for p in particles]
    positions = [np.array(p.history_position) for p in particles]
    velocities = [np.array(p.history_velocity) for p in particles]
    forces = [np.array(p.history_force) for p in particles]
    radii = [p.radius for p in particles]
    
    if not times:
        print("No particle data found.")
        return

    n_frames = len(times[0])
    # Reduce frames if too many (keep max ~100 frames for speed)
    stride = max(1, n_frames // 100) 
    frame_indices = np.arange(0, n_frames, stride)

    # --- Setup Figure ---
    fig = plt.figure(figsize=(10, 8))
    # Layout: Top half for Field, Bottom half split for Vel and Force
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    
    ax_field = fig.add_subplot(gs[0, :])
    ax_vel = fig.add_subplot(gs[1, 0])
    ax_force = fig.add_subplot(gs[1, 1])
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

    # --- 1. Field Plot ---
    ax_field.set_title("Simulation Field")
    ax_field.set_xlabel("X [m]")
    ax_field.set_ylabel("Y [m]")
    ax_field.set_aspect('equal')
    ax_field.grid(True, linestyle=':', alpha=0.6)
    
    # Draw Boundaries
    for b in boundaries:
        # FIXED: Explicitly slice to 2D (x, y) to avoid shape mismatch with 3D vectors
        p_c = np.array(b.point)[:2]
        n = np.array(b.normal)[:2]
        
        # Calculate tangent vector (-ny, nx)
        tangent = np.array([-n[1], n[0]])
        
        # Draw a line segment
        L = 20.0 
        p1 = p_c - tangent * L
        p2 = p_c + tangent * L
        
        ax_field.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k-', linewidth=3, alpha=0.7)
        # Draw normal arrow
        ax_field.arrow(p_c[0], p_c[1], n[0]*0.5, n[1]*0.5, head_width=0.1, color='k')

    # Auto-scale limits
    all_pos = np.concatenate(positions)
    x_min, x_max = np.min(all_pos[:,0]), np.max(all_pos[:,0])
    y_min, y_max = np.min(all_pos[:,1]), np.max(all_pos[:,1])
    margin = 0.5 + max(radii)
    ax_field.set_xlim(x_min - margin, x_max + margin)
    ax_field.set_ylim(y_min - margin, y_max + margin)

    # Init Particles
    field_circles = []
    field_trails = []
    
    for i, p in enumerate(particles):
        c = colors[i % len(colors)]
        circle = Circle((0, 0), p.radius, color=c, alpha=0.8, ec='black')
        ax_field.add_patch(circle)
        field_circles.append(circle)
        
        trail, = ax_field.plot([], [], '-', color=c, alpha=0.4, linewidth=1)
        field_trails.append(trail)

    time_text = ax_field.text(0.02, 0.95, '', transform=ax_field.transAxes, 
                              fontsize=12, bbox=dict(facecolor='white', alpha=0.8))

    # --- 2. Curve Plots ---
    vel_lines, vel_dots = plot_velocity(ax_vel, times, velocities, colors)
    force_lines, force_dots = plot_force(ax_force, times, forces, colors)

    # --- Animation Update ---
    def update(frame_idx):
        current_time = times[0][frame_idx]
        time_text.set_text(f"Time: {current_time:.2f} s")
        
        # Update Field
        for i, (pos, circle, trail) in enumerate(zip(positions, field_circles, field_trails)):
            current_pos = pos[frame_idx]
            circle.center = (current_pos[0], current_pos[1])
            trail.set_data(pos[:frame_idx, 0], pos[:frame_idx, 1])

        # Update Velocity
        for i, (time, vel, line, dot) in enumerate(zip(times, velocities, vel_lines, vel_dots)):
            vel_mag = np.linalg.norm(vel, axis=1)
            line.set_data(time[:frame_idx], vel_mag[:frame_idx])
            dot.set_data([time[frame_idx]], [vel_mag[frame_idx]])

        # Update Force
        for i, (time, force, line, dot) in enumerate(zip(times, forces, force_lines, force_dots)):
            force_mag = np.linalg.norm(force, axis=1)
            line.set_data(time[:frame_idx], force_mag[:frame_idx])
            dot.set_data([time[frame_idx]], [force_mag[frame_idx]])
            
        return field_circles + field_trails + vel_lines + vel_dots + force_lines + force_dots + [time_text]

    anim = animation.FuncAnimation(fig, update, frames=frame_indices, interval=50, blit=True)
    
    try:
        anim.save(output_path, writer='pillow', fps=20)
        print(f"Animation saved to {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
    
    plt.close(fig)

def main():
    print("Running real DEM simulation (Bouncing Scenario)...")

    # --- 1. Simulation Setup (Same as scenario_1_bouncing.py) ---
    # Contact model parameters
    coeff_of_restitution = 0.8
    mu_friction = 0.0

    # Particle parameters
    mass = 1.0
    E = 10000
    nu = 0.0
    radius = 0.5

    # External load
    gravity = np.array([0.0, -9.81, 0.0])

    # Simulation properties
    delta_t = 1e-3
    t_end = 2.0

    # Contact and integration models
    contact_params = {
        "coeff_of_restitution": coeff_of_restitution,
        "mu": mu_friction
    }

    # Initialize models
    # Note: Ensure get_contact_model and get_time_integration_method are imported
    contact_model = get_contact_model("HM+D", contact_params)
    time_integration = get_time_integration_method("velocity-verlet")

    # --- Create particle ---
    # Positioned at (5.0, 1.0, 0.0) to drop onto the floor
    p1 = Particle(position=[5.0, 1.0, 0.0], velocity=[0.0, 0.0, 0.0], omega=[0.0, 0.0, 0.0],
                  radius=radius, mass=mass, E=E, nu=nu, particle_id=1)

    # --- Create Boundary (Floor at y=0) ---
    b_point = np.array([0.0, 0.0, 0.0])
    b_normal = np.array([0.0, 1.0, 0.0])
    
    floor = Boundary(boundary_id_in=1, point_in=b_point, normal_in=b_normal,
                     E_in=1e10, nu_in=0.0, mu_in=mu_friction)

    # Solver setup
    solver = DEMSolver([p1], contact_model, time_integration, gravity, boundaries=[floor])

    # --- 2. Simulation Loop & Data Collection ---
    # Initialize lists to store history (compatible with what plot_field expects)
    times = []
    pos_hist = []
    vel_hist = []
    force_hist = []

    t = 0.0
    step_count = 0
    print_interval = 1000  # Print progress every 1000 steps

    while t < t_end:
        solver.solve_time_step(delta_t)

        # Log data
        times.append(t)
        pos_hist.append(p1.position.copy())
        vel_hist.append(p1.velocity.copy())
        
        # Capture force (assuming p1.force is available/calculated in solver)
        # If force is not directly exposed as a public attribute in your Particle class, 
        # you might need to use p1.acc * p1.mass or 0.0 if strictly kinematic.
        # Here we assume standard DEM particle has .force attribute.
        current_force = getattr(p1, 'force', np.zeros(3))
        force_hist.append(current_force.copy())

        t += delta_t
        step_count += 1
        
        if step_count % print_interval == 0:
             sys.stdout.write(f"\rSimulation Time: {t:.4f} / {t_end:.4f} s")
             sys.stdout.flush()

    print("\nSimulation finished. Preparing data for plot...")

    # --- 3. Format Data for plot_field ---
    # The plot_field function expects the particle object to have specific history attributes.
    # We dynamically attach them to the 'real' particle object here.
    
    p1.history_time = np.array(times)
    p1.history_position = np.array(pos_hist)
    p1.history_velocity = np.array(vel_hist)
    p1.history_force = np.array(force_hist)

    # Prepare lists for plotting
    boundaries = [floor]
    particles = [p1]

    # --- 4. Run Plot Function ---
    # Generates the simulation GIF using the real physics data
    plot_field(boundaries, particles, "simulation_bouncing.gif")

if __name__ == "__main__":
    main()