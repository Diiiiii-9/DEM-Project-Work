# validation/scenario_2_sliding.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import sys
import matplotlib.tri as tri


# Add parent directory to path to import 'dem' modules
# (Ensures imports work regardless of where you run the script from)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base_validation import BaseValidationScenario
from dem.particle import Particle
from dem.boundary import Boundary 


class SlidingFrictionScenario(BaseValidationScenario):
    """
    Scenario 2: Pure Sliding Friction Test
    """

    
    def setup_simulation(self):

        mu = getattr(self, "mu", 0.3)
        E_particle = getattr(self, "override_E", 1e7)
        v0 = getattr(self, "override_v0", 3.0)


        # 1. Define Parameters
        self.gravity = np.array([0, -9.81, 0])
        self.params = {
            "coeff_of_restitution": 0.5, 
            "mu": mu                 
        }
        
        # 2. Create Particles (Start at y=0.1)
        p1 = Particle(
            position=[0, 0.1, 0],
            velocity=[v0, 0, 0],
            omega=[0, 0, 0],
            radius=0.1,
            mass=1.0,
            E=E_particle,
            nu=0.3,
            particle_id=1
        )

        p1.inertia = 1e20 # Large inertia to prevent rotation
        self.particles = [p1]
        
        # 3. Create Boundary (Floor)
        floor = Boundary(
            boundary_id_in=101,
            point_in=[0, 0, 0],
            normal_in=[0, 1, 0],
            E_in=E_particle,
            nu_in=0.3,
            mu_in=mu
        )
  
        self.boundaries = [floor]

    def get_analytical_solution(self, time_array):
        mu = self.params["mu"]
        g = 9.81
        v0 = 3.0 
        t_stop = v0 / (mu * g)
        
        velocity_analytical = []
        position_analytical = []
        
        for t in time_array:
            if t < t_stop:
                v = v0 - mu * g * t
                x = v0 * t - 0.5 * mu * g * t**2
            else:
                v = 0.0
                x = v0 * t_stop - 0.5 * mu * g * t_stop**2 
            
            velocity_analytical.append(v)
            position_analytical.append(x)
            
        return np.array(velocity_analytical), np.array(position_analytical)

    def ensure_output_directory(self):
        """Creates the output directory if it doesn't exist."""
        # Assuming the structure is DEM-Project/Source Code/validation/scenario_2.py
        # We want DEM-Project/Source Code/results
        # Let's go up one level from the script to 'Source Code', then into 'results'
        
        # Method 1: Absolute path based on script location (Robust)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # points to "Source Code"
        output_dir = os.path.join(base_dir, "results")
        
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                print(f"Created output directory: {output_dir}")
            except OSError as e:
                print(f"Error creating directory {output_dir}: {e}")
                # Fallback to local folder
                return "."
        return output_dir

    def plot_results(self, i):
        """Plot simulation results and calculate errors for convergence analysis."""
        
        if not self.results["time"]: return
        time_sim = np.array(self.results["time"])
        vel_sim_x = np.array(self.results["vel"])[:, 0]
        pos_sim_x = np.array(self.results["pos"])[:, 0]
        vel_theory, pos_theory = self.get_analytical_solution(time_sim)
        
        # Calculate absolute errors
        error_vel = np.abs(vel_sim_x - vel_theory)
        error_pos = np.abs(pos_sim_x - pos_theory)

        # Calculate RMSE
        rmse_vel = calculate_rmse(vel_sim_x, vel_theory)
        rmse_pos = calculate_rmse(pos_sim_x, pos_theory)

        # Save errors for plotting later
        return error_vel, error_pos, rmse_vel, rmse_pos
    
    def plot_initial_oscillations(self, t_max=0.1):
        """
        Zoomed plot of early-time velocity oscillations.
        """
        time = np.array(self.results["time"])
        vel_x = np.array(self.results["vel"])[:, 0]

        mask = time <= t_max
        time_zoom = time[mask]
        vel_zoom = vel_x[mask]

        plt.figure(figsize=(8, 4))
        plt.plot(time_zoom, vel_zoom, 'b-', lw=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Velocity X [m/s]")
        plt.title("Early-Time Velocity Oscillations (HM+D Contact)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def apply_parameters(self, mu=None, E=None, v0=None):
        """
        Override parameters for sensitivity studies.
        """
        if mu is not None:
            self.mu = mu

        if E is not None:
            self.override_E = E

        if v0 is not None:
            self.override_v0 = v0



### RMSE Error from Predicted and True Solutions
def calculate_rmse(predicted, true):
    """Calculate Root Mean Square Error (RMSE) between predicted and true values."""
    return np.sqrt(np.mean((predicted - true) ** 2))
    



    # ================= COMBINED ANIMATION FUNCTION =================
    def plot_combined_animation(self, stride=50): 
        """
        Generates a GIF with Physical View (Top) and Data Curves (Bottom).
        """
        output_dir = self.ensure_output_directory()
        print(f"Generating combined animation... (Stride={stride})")

        # --- 1. Data Preparation ---
        # Full Data
        t_full = np.array(self.results["time"])
        p_full = np.array(self.results["pos"])
        v_full = np.array(self.results["vel"])
        
        # Strided Data (Frames)
        # We use these indices to drive the animation frames
        indices = np.arange(0, len(t_full), stride)
        
        # Theoretical Data for Background (Static lines)
        v_theory, p_theory = self.get_analytical_solution(t_full)

        radius = self.particles[0].radius
        max_x = np.max(p_full[:, 0])
        max_t = np.max(t_full)
        max_v = np.max(v_full[:, 0])

        # --- 2. Setup Figure Layout ---
        # Create a figure with a grid: Top row (Anim), Bottom row (Graphs)
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1]) # Top is animation, Bottom is graphs

        # A. Top Subplot: Physical Animation
        ax_anim = fig.add_subplot(gs[0, :]) # Span all columns
        ax_anim.set_aspect('equal')
        ax_anim.set_xlim(-0.2, max_x + 0.5)
        ax_anim.set_ylim(-0.1, radius * 4)
        ax_anim.set_xlabel("Position X [m]")
        ax_anim.set_title("Physical View: Sliding Particle")
        ax_anim.axhline(y=0, color='black', linewidth=2) # Ground
        
        # Particle Circle
        circle = Circle((0, 0), radius, fc='tab:blue', ec='black', lw=1.5)
        ax_anim.add_patch(circle)
        time_text = ax_anim.text(0.02, 0.85, '', transform=ax_anim.transAxes, fontsize=12)

        # B. Bottom Left Subplot: Velocity
        ax_vel = fig.add_subplot(gs[1, 0])
        ax_vel.set_xlim(0, max_t)
        ax_vel.set_ylim(0, max_v * 1.1)
        ax_vel.set_xlabel("Time [s]")
        ax_vel.set_ylabel("Velocity [m/s]")
        ax_vel.set_title("Velocity vs Time")
        ax_vel.grid(True, linestyle='--', alpha=0.5)
        
        # Static Background Line (Theory)
        ax_vel.plot(t_full, v_theory, 'r--', alpha=0.5, label='Theory', lw=1)
        # Dynamic Lines (Simulation)
        line_vel, = ax_vel.plot([], [], 'b-', lw=2, label='Sim')
        dot_vel, = ax_vel.plot([], [], 'bo') # The moving dot
        ax_vel.legend()

        # C. Bottom Right Subplot: Position
        ax_pos = fig.add_subplot(gs[1, 1])
        ax_pos.set_xlim(0, max_t)
        ax_pos.set_ylim(0, max_x * 1.1)
        ax_pos.set_xlabel("Time [s]")
        ax_pos.set_ylabel("Position X [m]")
        ax_pos.set_title("Position vs Time")
        ax_pos.grid(True, linestyle='--', alpha=0.5)

        # Static Background Line (Theory)
        ax_pos.plot(t_full, p_theory, 'k--', alpha=0.5, label='Theory', lw=1)
        # Dynamic Lines (Simulation)
        line_pos, = ax_pos.plot([], [], 'g-', lw=2, label='Sim')
        dot_pos, = ax_pos.plot([], [], 'go') # The moving dot
        ax_pos.legend()

        # --- 3. Animation Update Function ---
        def animate(frame_idx):
            # frame_idx is the index in our 'indices' array
            # real_idx is the index in the full data array
            real_idx = indices[frame_idx]
            
            # 1. Update Physical View
            current_pos = p_full[real_idx]
            current_time = t_full[real_idx]
            circle.set_center((current_pos[0], current_pos[1]))
            time_text.set_text(f'Time: {current_time:.2f} s')
            
            # 2. Update Graphs
            # We want to show the line from t=0 up to current time
            # Using slicing [0 : real_idx]
            
            # Prevent plotting empty arrays at start
            if real_idx > 0:
                t_history = t_full[:real_idx]
                v_history = v_full[:real_idx, 0]
                p_history = p_full[:real_idx, 0]
                
                # Update Velocity Plot
                line_vel.set_data(t_history, v_history)
                dot_vel.set_data([current_time], [v_full[real_idx, 0]]) # Must be sequence
                
                # Update Position Plot
                line_pos.set_data(t_history, p_history)
                dot_pos.set_data([current_time], [p_full[real_idx, 0]])
            
            return circle, time_text, line_vel, dot_vel, line_pos, dot_pos

        # --- 4. Create and Save ---
        anim = animation.FuncAnimation(
            fig, animate, frames=len(indices), interval=30, blit=True
        )

        save_path = os.path.join(output_dir, 'scenario_2_combined.gif')
        try:
            anim.save(save_path, writer='pillow', fps=30)
            print(f"Animation saved successfully to: {save_path}")
        except Exception as e:
            print(f"Error saving animation: {e}")
        
        plt.close(fig)

def plot_error(dt_values, errors_vel, errors_pos, rmse_vel, rmse_pos):
    """Plot the errors (absolute and RMSE) between predicted and real solutions."""
    plt.figure(figsize=(10, 6))
    
    # Plot Absolute Error for Velocity and Position
    plt.subplot(1, 2, 1)
    plt.plot(dt_values, errors_vel, 'bo-', label='Velocity Absolute Error')
    plt.plot(dt_values, errors_pos, 'go-', label='Position Absolute Error')
    plt.xscale('log')  # Log scale for dt
    plt.xlabel('dt (Time Step)')
    plt.ylabel('Error (Absolute)')
    plt.legend()
    plt.grid(True)
    plt.title('Absolute Error: Velocity and Position')
    
    # Plot RMSE for Velocity and Position
    plt.subplot(1, 2, 2)
    plt.plot(dt_values, rmse_vel, 'bo-', label='Velocity RMSE')
    plt.plot(dt_values, rmse_pos, 'go-', label='Position RMSE')
    plt.xscale('log')  # Log scale for dt
    plt.yscale('log')
    plt.xlabel('dt (Time Step)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    plt.title('RMSE: Velocity and Position')
    
    plt.tight_layout()
    plt.show()


### Sensitivity Analysis
## Variables Independent
def run_sensitivity(
    scenario_class,
    param_name,
    param_values,
    duration=1.5,
    dt=1e-4
):
    results = []

    for val in param_values:
        scenario = scenario_class(
            f"{param_name}={val}",
            duration=duration,
            dt=dt
        )

        # Apply parameter
        if param_name == "mu":
            scenario.apply_parameters(mu=val)
        elif param_name == "E":
            scenario.apply_parameters(E=val)
        elif param_name == "v0":
            scenario.apply_parameters(v0=val)

        scenario.run_simulation()

        time = np.array(scenario.results["time"])
        vel = np.array(scenario.results["vel"])      # shape (N,3)
        pos = np.array(scenario.results["pos"])      # shape (N,3)

        # Detect stopping time (optional metric)
        v_tol = 1e-3
        vel_x = vel[:, 0]
        stopped = np.where(vel_x < v_tol)[0]

        if len(stopped) == 0:
            t_stop = np.nan
            x_stop = np.nan
        else:
            idx = stopped[0]
            t_stop = time[idx]
            x_stop = pos[idx, 0]

        results.append({
            "param": val,

            # full histories
            "time": time,
            "vel": vel,
            "pos": pos,

            # optional reduced metrics
            "t_stop": t_stop,
            "x_stop": x_stop
        })

    return results

def mu_parameter_analysis():
    results_mu = run_sensitivity(
        SlidingFrictionScenario,
        param_name="mu",
        param_values=[0.0, 0.3, 0.4, 0.5, 0.6, 1.0, 3058.0]
    )


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Velocity subplot ---
    for r in results_mu:
        axes[0].plot(
            r["time"],
            r["vel"][:, 0],
            label=f"μ = {r['param']}"
        )

    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Velocity X [m/s]")
    axes[0].set_title("Velocity vs Time – Friction Sensitivity")
    axes[0].grid(True)
    axes[0].legend()

    # --- Position subplot ---
    for r in results_mu:
        axes[1].plot(
            r["time"],
            r["pos"][:, 0],
            label=f"μ = {r['param']}"
        )

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Position X [m]")
    axes[1].set_title("Position vs Time – Friction Sensitivity")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def v0_parameter_analysis():
    results_mu = run_sensitivity(
        SlidingFrictionScenario,
        param_name="v0",
        param_values=[0.0, 1.0, 3.0, 5.0]
    )


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Velocity subplot ---
    for r in results_mu:
        axes[0].plot(
            r["time"],
            r["vel"][:, 0],
            label=f"v0 = {r['param']}"
        )

    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Velocity X [m/s]")
    axes[0].set_title("Velocity vs Time – Initial Velocity Sensitivity")
    axes[0].grid(True)
    axes[0].legend()

    # --- Position subplot ---
    for r in results_mu:
        axes[1].plot(
            r["time"],
            r["pos"][:, 0],
            label=f"v0 = {r['param']}"
        )

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Position X [m]")
    axes[1].set_title("Position vs Time – Initial Velocity Sensitivity")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

def E_parameter_analysis():
    results_mu = run_sensitivity(
        SlidingFrictionScenario,
        param_name="E",
        param_values=[1e5, 1e6, 1e7, 1e8]
    )


    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Velocity subplot ---
    for r in results_mu:
        axes[0].plot(
            r["time"],
            r["vel"][:, 0],
            label=f"E = {r['param']}"
        )

    axes[0].set_xlabel("Time [s]")
    axes[0].set_xlim(0.0, 0.2) 
    axes[0].set_ylim(2.0, 3.0) 
    axes[0].set_ylabel("Velocity X [m/s]")
    axes[0].set_title("Velocity vs Time – Young's Modulus Sensitivity")
    axes[0].grid(True)
    axes[0].legend()

    # --- Position subplot ---
    for r in results_mu:
        axes[1].plot(
            r["time"],
            r["pos"][:, 0],
            label=f"E = {r['param']}"
        )

    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Position X [m]")
    axes[1].set_title("Position vs Time – Young's Modulus Sensitivity")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()

## Phase Plot of (mu, v0)
def run_mu_v0_phase_study(
    scenario_class,
    mu_values,
    v0_values,
    duration=1.5,
    dt=1e-4
):
    data = []

    for mu in mu_values:
        for v0 in v0_values:
            scenario = scenario_class(
                f"mu={mu}, v0={v0}",
                duration=duration,
                dt=dt
            )

            scenario.apply_parameters(mu=mu, v0=v0)
            scenario.run_simulation()

            time = np.array(scenario.results["time"])
            vel_x = np.array(scenario.results["vel"])[:, 0]

            # Area under velocity curve = distance traveled
            area = np.trapz(vel_x, time)

            data.append({
                "mu": mu,
                "v0": v0,
                "area": area
            })

    return data

def plot_mu_v0_phase_contour(data):
    mu = np.array([d["mu"] for d in data])
    v0 = np.array([d["v0"] for d in data])
    area = np.array([d["area"] for d in data])

    triang = tri.Triangulation(mu, v0)

    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(
        triang,
        area,
        levels=30,
        cmap="viridis"
    )
    plt.colorbar(contour, label="Distance traveled (∫ v dt) [m]")

    plt.xlabel("Friction coefficient μ")
    plt.ylabel("Initial velocity v₀ [m/s]")
    plt.title("Phase Plot: Distance Traveled vs (μ, v₀)")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()

def phase_mu_v0_analysis():

    mu_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    v0_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    data = run_mu_v0_phase_study(
        SlidingFrictionScenario,
        mu_values,
        v0_values
    )

    plot_mu_v0_phase_contour(data)


if __name__ == "__main__":

    # 1st PART
    # Study the influence of dt iin the RMSE error between (Predicted, True) pairs.
    FirstPart = False
    if FirstPart:
        dt_values = [0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001, 0.000001]

        errors_vel = []
        errors_pos = []
        rmse_vel = []
        rmse_pos = []

        for dt in dt_values:
            scenario = SlidingFrictionScenario("Sliding Test", duration=1.5, dt=dt)
            scenario.run_simulation()
            
            # Collect errors for each dt value
            error_vel, error_pos, rmse_v, rmse_p = scenario.plot_results(dt)
            
            # Store the results
            errors_vel.append(np.mean(error_vel))  # Mean absolute error for velocity
            errors_pos.append(np.mean(error_pos))  # Mean absolute error for position
            rmse_vel.append(rmse_v)  # RMSE for velocity
            rmse_pos.append(rmse_p)  # RMSE for position

        # Plot the errors
        plot_error(dt_values, errors_vel, errors_pos, rmse_vel, rmse_pos)

    # 2nd PART
    # Plot Initial Oscillations cause by Damper in HM+D Model
    SecondPart = False
    if SecondPart:
        scenario = SlidingFrictionScenario("Sliding Test", duration=1.5, dt=1e-4)
        scenario.run_simulation()
        scenario.plot_initial_oscillations(t_max = 0.06)

    # 3rd PART
    # Parameter Analysis
    ThirdPart = True
    if ThirdPart:
        # 3.1 Mu values
        mu_parameter_analysis()

        # 3.2 young Moduus
        E_parameter_analysis()

        # 3.3 initial velocity
        v0_parameter_analysis()

        # 3.4 Phase plot of (mu, v0)
        phase_mu_v0_analysis()

    



