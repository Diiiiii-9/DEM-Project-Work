# validation/scenario_2_sliding.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import sys

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
        # 1. Define Parameters
        self.gravity = np.array([0, -9.81, 0])
        self.params = {
            "coeff_of_restitution": 0.5, 
            "mu": 0.3                    
        }
        
        # 2. Create Particles (Start at y=0.1)
        p1 = Particle(
            position=[0, 0.1, 0],   
            velocity=[3.0, 0, 0], 
            omega=[0, 0, 0],
            radius=0.1,
            mass=1.0,
            E=1e7,
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
            E_in=1e7,
            nu_in=0.3,
            mu_in=0.3
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

    def plot_results(self):
        # Save static result first
        output_dir = self.ensure_output_directory()
        
        if not self.results["time"]: return
        time_sim = np.array(self.results["time"])
        vel_sim_x = np.array(self.results["vel"])[:, 0]
        pos_sim_x = np.array(self.results["pos"])[:, 0]
        vel_theory, pos_theory = self.get_analytical_solution(time_sim)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(time_sim, vel_sim_x, 'b-', label='DEM')
        plt.plot(time_sim, vel_theory, 'r--', label='Theory')
        plt.xlabel('Time [s]'); plt.ylabel('Velocity X [m/s]'); plt.legend(); plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(time_sim, pos_sim_x, 'g-', label='DEM')
        plt.plot(time_sim, pos_theory, 'k--', label='Theory')
        plt.xlabel('Time [s]'); plt.ylabel('Position X [m]'); plt.legend(); plt.grid(True)
        
        save_path = os.path.join(output_dir, 'scenario_2_static_plot.png')
        plt.savefig(save_path)
        print(f"Static plot saved to: {save_path}")
        plt.close()

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

if __name__ == "__main__":
    # 1. Run Simulation
    scenario = SlidingFrictionScenario("Sliding Test", duration=1.5, dt=1e-4)
    scenario.run_simulation()
    
    # 2. Plot Static Results (png)
    scenario.plot_results()
    
    # 3. Plot Combined Animation (gif)
    # Stride=100 means we plot every 100th point. 
    # 15000 total steps / 100 = 150 frames. At 30fps, gif is ~5 seconds.
    scenario.plot_combined_animation(stride=100)