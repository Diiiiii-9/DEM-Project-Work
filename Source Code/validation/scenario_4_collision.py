# validation/scenario_4_collision.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import os
import sys

# Path setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from base_validation import BaseValidationScenario
from dem.particle import Particle
from dem.solver import DEMSolver
from dem.contact_model import HertzMindlinDashpot
from dem.time_integration import VelocityVerletIntegrator

# No boundary needed for this test

class ObliqueCollisionScenario(BaseValidationScenario):
    """
    Scenario 4: Oblique Particle-Particle Collision
    """

    def setup_simulation(self):
        # 1. Zero Gravity
        self.gravity = np.array([0, 0, 0]) 
        
        self.params = {
            "coeff_of_restitution": 0.8,
            "mu": 0.5 
        }
        
        R = 0.1
        mass = 1.0
        b = R * 1.0 
        
        # Particle 1: Moving Right
        p1 = Particle(
            position=[-0.4, b, 0], 
            velocity=[2.0, 0, 0], 
            omega=[0, 0, 0],
            radius=R, mass=mass, E=1e7, nu=0.3, particle_id=1
        )
        
        # Particle 2: Stationary
        p2 = Particle(
            position=[0, 0, 0], 
            velocity=[0, 0, 0], 
            omega=[0, 0, 0],
            radius=R, mass=mass, E=1e7, nu=0.3, particle_id=2
        )
        
        self.particles = [p1, p2]

    # Override run_simulation to handle multi-particle data recording
    def run_simulation(self):
        """
        Overriding run_simulation to support multi-particle data recording.
        """
        print(f"--- Starting Scenario (Multi-Particle): {self.name} ---")
        
        # 1. Setup
        self.setup_simulation()
        boundaries = getattr(self, 'boundaries', [])
        contact_model = HertzMindlinDashpot(self.params)
        integrator = VelocityVerletIntegrator()
        
        # Create solver
        solver = DEMSolver(self.particles, contact_model, integrator, self.gravity, boundaries)
        
        # 2. Time Loop
        steps = int(self.duration / self.dt)
        print(f"Simulating {steps} steps...")
        
        for i in range(steps):
            current_time = i * self.dt
            
            # --- FIX: Record data for ALL particles (List of Arrays) ---
            current_pos_list = [p.position.copy() for p in self.particles]
            current_vel_list = [p.velocity.copy() for p in self.particles]
            current_omega_list = [p.omega.copy() for p in self.particles]
            
            self.results["time"].append(current_time)
            self.results["pos"].append(current_pos_list)
            self.results["vel"].append(current_vel_list)

            # Handle Omega initialization safely
            if "omega" not in self.results:
                self.results["omega"] = []
            self.results["omega"].append(current_omega_list)
            
            # Advance Step
            solver.solve_time_step(self.dt)
            
        print("Simulation finished.")

    def get_analytical_solution(self, time_array):
        m1 = self.particles[0].mass
        v1_init = np.array([2.0, 0, 0])
        p_total_init = m1 * v1_init 
        
        p_theory_x = np.full_like(time_array, p_total_init[0])
        p_theory_y = np.full_like(time_array, p_total_init[1])
        
        return p_theory_x, p_theory_y

    def ensure_output_directory(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base_dir, "results")
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                return "."
        return output_dir

    def plot_results(self):
        output_dir = self.ensure_output_directory()
        if not self.results["time"]: return
        
        time_sim = np.array(self.results["time"])
        
        # Now this works because run_simulation saved list of lists
        vel_data = np.array(self.results["vel"]) # (Steps, 2, 3)
        omega_data = np.array(self.results["omega"]) # (Steps, 2, 3)
        
        m1 = self.particles[0].mass
        m2 = self.particles[1].mass
        
        P_total_x = m1 * vel_data[:, 0, 0] + m2 * vel_data[:, 1, 0]
        P_total_y = m1 * vel_data[:, 0, 1] + m2 * vel_data[:, 1, 1]
        
        P_theory_x, P_theory_y = self.get_analytical_solution(time_sim)
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(time_sim, omega_data[:, 0, 2], 'b-', label='P1 Omega Z')
        plt.plot(time_sim, omega_data[:, 1, 2], 'g-', label='P2 Omega Z')
        plt.xlabel('Time [s]'); plt.ylabel('Omega [rad/s]'); plt.legend(); plt.grid(True)
        plt.title('Angular Velocity Generation')

        plt.subplot(1, 2, 2)
        plt.plot(time_sim, P_total_x, 'k-', lw=2, label='Sim Px')
        plt.plot(time_sim, P_theory_x, 'r--', lw=2, label='Theory Px')
        plt.plot(time_sim, P_total_y, 'b-', lw=1, label='Sim Py')
        plt.xlabel('Time [s]'); plt.ylabel('Momentum'); plt.legend(); plt.grid(True)
        plt.title('Momentum Conservation')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scenario_4_static.png'))
        print("Static plot saved.")

    def plot_combined_animation(self, stride=20):
        # ... (这里保持你之前的动画代码不变，或者复制之前我发给你的动画代码) ...
        # ... (为了节省篇幅，这里省略动画代码，请直接使用上面的动画代码块) ...
        output_dir = self.ensure_output_directory()
        print(f"Generating animation... (Stride={stride})")

        t_full = np.array(self.results["time"])
        pos_data = np.array(self.results["pos"]) 
        omega_data = np.array(self.results["omega"]) 
        
        indices = np.arange(0, len(t_full), stride)
        
        fig = plt.figure(figsize=(10, 8), constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

        ax_anim = fig.add_subplot(gs[0])
        ax_anim.set_aspect('equal')
        ax_anim.set_xlim(-0.6, 0.6)
        ax_anim.set_ylim(-0.4, 0.4)
        ax_anim.grid(True, linestyle='--', alpha=0.5)

        R = self.particles[0].radius
        c1 = Circle((0,0), R, fc='tab:blue', ec='black', label='P1')
        c2 = Circle((0,0), R, fc='tab:green', ec='black', label='P2')
        line1, = ax_anim.plot([], [], 'w-', lw=2) 
        line2, = ax_anim.plot([], [], 'w-', lw=2)
        ax_anim.add_patch(c1); ax_anim.add_patch(c2); ax_anim.legend()

        ax_graph = fig.add_subplot(gs[1])
        ax_graph.set_xlim(0, t_full[-1])
        max_w = np.max(np.abs(omega_data[:, :, 2])) + 1.0
        ax_graph.set_ylim(-max_w, max_w)
        ax_graph.set_ylabel("Omega Z"); ax_graph.grid(True)
        l_w1, = ax_graph.plot([], [], 'b-', label='P1')
        l_w2, = ax_graph.plot([], [], 'g-', label='P2')
        ax_graph.legend()

        def animate(frame_idx):
            idx = indices[frame_idx]
            p1_pos, p2_pos = pos_data[idx, 0], pos_data[idx, 1]
            c1.set_center((p1_pos[0], p1_pos[1]))
            c2.set_center((p2_pos[0], p2_pos[1]))
            
            theta1 = np.sum(omega_data[:idx, 0, 2]) * self.dt 
            theta2 = np.sum(omega_data[:idx, 1, 2]) * self.dt 
            line1.set_data([p1_pos[0], p1_pos[0] + R*np.cos(theta1)], [p1_pos[1], p1_pos[1] + R*np.sin(theta1)])
            line2.set_data([p2_pos[0], p2_pos[0] + R*np.cos(theta2)], [p2_pos[1], p2_pos[1] + R*np.sin(theta2)])

            if idx > 0:
                l_w1.set_data(t_full[:idx], omega_data[:idx, 0, 2])
                l_w2.set_data(t_full[:idx], omega_data[:idx, 1, 2])
            return c1, c2, line1, line2, l_w1, l_w2

        anim = animation.FuncAnimation(fig, animate, frames=len(indices), interval=30, blit=True)
        anim.save(os.path.join(output_dir, 'scenario_4_oblique.gif'), writer='pillow', fps=30)
        plt.close(fig)

if __name__ == "__main__":
    scenario = ObliqueCollisionScenario("Oblique Collision", duration=0.5, dt=5e-5)
    scenario.run() 
    scenario.plot_combined_animation(stride=50)