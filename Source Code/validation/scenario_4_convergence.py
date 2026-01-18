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
import matplotlib.tri as tri


# No boundary needed for this test

class ObliqueCollisionScenario(BaseValidationScenario):
    """
    Scenario 4: Oblique Particle-Particle Collision
    """

    def setup_simulation(self):

        mu = getattr(self, "mu", 0.5)
        CoR = getattr(self, "CoR", 0.8)

        # 1. Zero Gravity
        self.gravity = np.array([0, 0, 0]) 
        
        self.params = {
            "coeff_of_restitution": CoR,
            "mu": mu
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

    def _sphere_inertia(self, p):
        # Use p.inertia if already defined; otherwise use solid sphere inertia
        if hasattr(p, "inertia") and np.isfinite(p.inertia) and p.inertia > 0:
            return p.inertia
        return 0.4 * p.mass * (p.radius ** 2)

    def predicted_trajectory(self):
        """
        Predict the trajectories of two particles after collision.
        
        Parameters:
        - m1, m2: Masses of particle 1 and particle 2
        - R: Radius of the particles
        - v1_init, v2_init: Initial velocities of particle 1 and particle 2
        - pos1_init, pos2_init: Initial positions of particle 1 and particle 2
        - e: Coefficient of restitution (COR)
        - dt: Time step for simulation
        - t_max: Maximum time to simulate
        
        Returns:
        - pos1_trajectory, pos2_trajectory: Predicted position trajectories of both particles
        """
        p1, p2 = self.particles

        # Compute the normal vector between particles before collision
        delta_pos = np.array(p1.position) - np.array(p2.position)
        norm_delta_pos = np.linalg.norm(delta_pos)
        normal = delta_pos / norm_delta_pos
        
        # Compute the relative velocity along the normal
        v_rel = np.array(p2.velocity) - np.array(p1.velocity)
        v_rel_normal = np.dot(v_rel, normal)
        
        # Compute the new velocities based on the restitution equation
        v1_after = np.array(p1.velocity) + (1 + self.params["coeff_of_restitution"]) * p2.mass * v_rel_normal / p1.mass
        v2_after = np.array(p2.velocity) - (1 + self.params["coeff_of_restitution"]) * p1.mass * v_rel_normal / p2.mass
        
        # Now simulate the trajectories post-collision
        t_steps = int(self.duration / self.dt)
        t = np.linspace(0, self.duration, t_steps)
        
        # Predict the trajectories
        pos1_trajectory = np.array([p1.position + v1_after * t_i for t_i in t])
        pos2_trajectory = np.array([p2.position + v2_after * t_i for t_i in t])
        
        return pos1_trajectory, pos2_trajectory

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

    # Override run_simulation to handle multi-particle data recording
    def run_simulation(self):
        print(f"--- Starting Scenario (Multi-Particle): {self.name} ---")

        self.setup_simulation()
        boundaries = getattr(self, 'boundaries', [])
        contact_model = HertzMindlinDashpot(self.params)
        integrator = VelocityVerletIntegrator()
        solver = DEMSolver(self.particles, contact_model, integrator, self.gravity, boundaries)

        # Ensure result keys exist
        for k in ["time", "pos", "vel", "omega", "p", "ke_trans", "ke_rot",
                "P_total", "KE_total_trans", "KE_total_rot", "KE_total"]:
            if k not in self.results:
                self.results[k] = []

        steps = int(self.duration / self.dt)
        print(f"Simulating {steps} steps...")

        for i in range(steps):
            t = i * self.dt

            pos_list = [p.position.copy() for p in self.particles]
            vel_list = [p.velocity.copy() for p in self.particles]
            omg_list = [p.omega.copy() for p in self.particles]

            # Momentum and energies
            p_list = []
            ke_t_list = []
            ke_r_list = []

            for p in self.particles:
                mom = p.mass * p.velocity
                I = self._sphere_inertia(p)
                ke_t = 0.5 * p.mass * np.dot(p.velocity, p.velocity)
                ke_r = 0.5 * I * np.dot(p.omega, p.omega)

                p_list.append(mom)
                ke_t_list.append(ke_t)
                ke_r_list.append(ke_r)

            P_tot = np.sum(p_list, axis=0)
            KE_t_tot = float(np.sum(ke_t_list))
            KE_r_tot = float(np.sum(ke_r_list))
            KE_tot = KE_t_tot + KE_r_tot

            # Store
            self.results["time"].append(t)
            self.results["pos"].append(pos_list)
            self.results["vel"].append(vel_list)
            self.results["omega"].append(omg_list)

            self.results["p"].append(p_list)
            self.results["ke_trans"].append(ke_t_list)
            self.results["ke_rot"].append(ke_r_list)

            self.results["P_total"].append(P_tot)
            self.results["KE_total_trans"].append(KE_t_tot)
            self.results["KE_total_rot"].append(KE_r_tot)
            self.results["KE_total"].append(KE_tot)

            solver.solve_time_step(self.dt)

        print("Simulation finished.")

    def expected_post_collision_velocities_mu0(self, e):
        """
        Rigid oblique impact prediction for mu = 0 (no tangential impulse, no spin).
        Uses geometry implied by your initial setup (impact parameter b = R).
        """
        v1 = np.array([2.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 0.0])
        m1 = self.particles[0].mass
        m2 = self.particles[1].mass

        # Unit normal at first contact from particle 2 to particle 1 for b=R:
        n = np.array([-np.sqrt(3)/2, 0.5, 0.0])

        g = v1 - v2
        g_n = np.dot(g, n)  # should be negative for approaching
        if g_n >= 0:
            # If the sign convention doesn't match due to geometry changes, flip n
            n = -n
            g_n = np.dot(g, n)

        J = -(1.0 + e) * g_n / (1.0/m1 + 1.0/m2)

        v1p = v1 + (J/m1) * n
        v2p = v2 - (J/m2) * n
        return v1p, v2p

    def plot_results_LAWS(self):
        output_dir = self.ensure_output_directory()
        if not self.results["time"]:
            return

        t = np.array(self.results["time"])

        p_data = np.array(self.results["p"])           # (N, 2, 3)
        P_tot = np.array(self.results["P_total"])      # (N, 3)

        ke_t = np.array(self.results["ke_trans"])      # (N, 2)
        ke_r = np.array(self.results["ke_rot"])        # (N, 2)
        KE_tot = np.array(self.results["KE_total"])    # (N,)

        # Expected totals (always): constant total momentum
        P0 = P_tot[0].copy()
        P_expected = np.tile(P0, (len(t), 1))

        # Optional: expected post-collision translational velocities for mu=0
        mu = self.params.get("mu", None)
        e = self.params.get("coeff_of_restitution", None)

        # Build figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))

        # 1) Particle momenta (x and y)
        #axes[0, 0].plot(t, p_data[:, 0, 0], label="p1 Px")
        #axes[0, 0].plot(t, p_data[:, 1, 0], label="p2 Px")
        #axes[0, 0].plot(t, p_data[:, 0, 1], label="p1 Py")
        #axes[0, 0].plot(t, p_data[:, 1, 1], label="p2 Py")
        axes[0, 0].plot(t, P_tot[:, 0], "k-", lw=2, label="Sim Px_total")
        axes[0, 0].plot(t, P_expected[:, 0], "r--", lw=2, label="Expected Px_total")
        axes[0, 0].plot(t, P_tot[:, 1], "b-", lw=1.5, label="Sim Py_total")
        axes[0, 0].plot(t, P_expected[:, 1], "g--", lw=1.5, label="Expected Py_total")
        axes[0, 0].plot(t, p_data[:, 0, 0]+p_data[:, 0, 1], label="p1 (Total)")
        axes[0, 0].plot(t, p_data[:, 1, 0]+p_data[:, 1, 1], label="p2 (Total)")
        axes[0, 0].set_title("Particle Linear Momentum Components")
        axes[0, 0].set_xlabel("Time [s]")
        axes[0, 0].set_ylabel("Momentum [kg·m/s]")
        axes[0, 0].grid(True)
        axes[0, 0].legend(loc="upper right")

        # 2) Total momentum vs expected
        axes[0, 1].plot(t, P_tot[:, 0], "k-", lw=2, label="Sim Px_total")
        axes[0, 1].plot(t, P_expected[:, 0], "r--", lw=2, label="Expected Px_total")
        axes[0, 1].plot(t, P_tot[:, 1], "b-", lw=1.5, label="Sim Py_total")
        axes[0, 1].plot(t, P_expected[:, 1], "g--", lw=1.5, label="Expected Py_total")
        axes[0, 1].set_title("Total Linear Momentum Conservation")
        axes[0, 1].set_xlabel("Time [s]")
        axes[0, 1].set_ylabel("Total Momentum [kg·m/s]")
        axes[0, 1].grid(True)
        axes[0, 1].legend()

        # 3) Particle energies
        axes[1, 0].plot(t, ke_t[:, 0], label="P1 KE_trans")
        axes[1, 0].plot(t, ke_t[:, 1], label="P2 KE_trans")
        axes[1, 0].plot(t, ke_r[:, 0], "--", label="P1 KE_rot")
        axes[1, 0].plot(t, ke_r[:, 1], "--", label="P2 KE_rot")
        axes[1, 0].set_title("Per-Particle Kinetic Energies")
        axes[1, 0].set_xlabel("Time [s]")
        axes[1, 0].set_ylabel("Energy [J]")
        axes[1, 0].grid(True)
        axes[1, 0].legend()

        # 4) Total kinetic energy (trans+rot)
        axes[1, 1].plot(t, KE_tot, "k-", lw=2, label="Sim KE_total")
        axes[1, 1].axhline(KE_tot[0], color="r", linestyle="--", lw=2, label="Initial KE_total")
        axes[1, 1].set_title("Total Kinetic Energy (Trans + Rot)")
        axes[1, 1].set_xlabel("Time [s]")
        axes[1, 1].set_ylabel("Energy [J]")
        axes[1, 1].grid(True)
        axes[1, 1].legend()

        fig.suptitle(f"Collision V&V: mu={mu}, CoR={e}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "scenario_4_momentum_energy.png"))
        plt.close(fig)

        print("Saved scenario_4_momentum_energy.png")

    def apply_parameters(self, mu=None, CoR=None):
        if mu is not None:
            self.mu = mu
        if CoR is not None:
            self.CoR = CoR


## Parameter Analysis in Phase space (mu, CoR)
def sweep_mu_e(
    scenario_class,
    mu_values,
    e_values,
    duration=0.5,
    dt=5e-5
):
    rows = []
    for mu in mu_values:
        for e in e_values:
            sc = scenario_class(f"mu={mu}, CoR={e}", duration=duration, dt=dt)
            sc.setup_simulation()  # sets gravity/particles/params
            sc.apply_parameters(mu=mu, CoR=e)
            sc.run_simulation()

            t = np.array(sc.results["time"])
            P = np.array(sc.results["P_total"])
            KE = np.array(sc.results["KE_total"])
            omg = np.array(sc.results["omega"])  # (N,2,3)

            P0 = P[0]
            P_err = np.max(np.linalg.norm(P - P0, axis=1) / (np.linalg.norm(P0) + 1e-12))

            KE0 = KE[0]
            KEf = KE[-1]
            eta_E = KEf / (KE0 + 1e-12)

            Omega = abs(omg[-1, 0, 2]) + abs(omg[-1, 1, 2])

            rows.append({
                "mu": mu,
                "e": e,
                "P_err": P_err,
                "eta_E": eta_E,
                "Omega": Omega
            })
    return rows

def tricontour_metric(rows, metric_key, title, cbar_label):
    mu = np.array([r["mu"] for r in rows])
    e  = np.array([r["e"]  for r in rows])
    z  = np.array([r[metric_key] for r in rows])

    triang = tri.Triangulation(mu, e)

    plt.figure(figsize=(8, 6))
    cf = plt.tricontourf(triang, z, levels=25, cmap = 'plasma')
    plt.colorbar(cf, label=cbar_label)
    plt.xlabel("Friction μ")
    plt.ylabel("Restitution CoR")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_metric_vs_e_for_fixed_mu(
    rows,
    metric_key,
    mu_target,
    tol=1e-8,
    normalize=False
):
    # --- Filter rows ---
    filtered = [r for r in rows if abs(r["mu"] - mu_target) < tol]
    if len(filtered) == 0:
        raise ValueError(f"No data found for mu = {mu_target}")

    # --- Extract data ---
    e_vals = np.array([r["e"] for r in filtered])
    z_vals = np.array([r[metric_key] for r in filtered])

    # --- Optional normalization ---
    if normalize:
        z_vals = z_vals / np.max(z_vals)

    # --- Sort by e ---
    idx = np.argsort(e_vals)
    e_vals = e_vals[idx]
    z_vals = z_vals[idx]

    # --- Analytical curve ---
    e_ref = np.linspace(e_vals.min(), e_vals.max(), 300)
    theory = 1.0 - e_ref**2

    # --- Plot ---
    plt.figure(figsize=(7, 5))
    plt.plot(e_vals, z_vals, 'o-', lw=2, label="DEM")
    plt.plot(e_ref, theory, 'k--', lw=2, label=r"Theory: $1 - e^2$")

    plt.xlabel("Restitution coefficient e")
    plt.ylabel(metric_key)
    plt.title(f"{metric_key} vs restitution (μ = {mu_target})")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    scenario = ObliqueCollisionScenario("Oblique Collision", duration=0.5, dt=5e-5)
    scenario.run() 
    #scenario.plot_results()
    #scenario.plot_combined_animation(stride=50)

    # 1st PART
    # Plot Laws of Energy and Momentum
    FirstPart = True
    if FirstPart:
        print("Laws of energy and Momemtum plot!")
        scenario.plot_results_LAWS()

    # 2nd PART
    # Sweep analysis for parameters mu and CoR
    SecondPart = True
    if SecondPart:
        print("--- Staarting Sweep Analysis with Respect to (μ,CoR)")
        mu_values = [0.0, 0.1, 0.3, 0.5, 0.8]
        CoR_values  = [0.2, 0.5, 0.8, 0.95]

        rows = sweep_mu_e(ObliqueCollisionScenario, mu_values, CoR_values)

        tricontour_metric(rows, "eta_E",
                        "Energy Retention vs (μ,CoR)",
                        "KE_final / KE_initial [-]")

        tricontour_metric(rows, "Omega",
                        "Spin Generation vs (μ,CoR)",
                        "|ω1z|+|ω2z| [rad/s]")

        tricontour_metric(rows, "P_err",
                        "Momentum Conservation Error vs (μ,CoR)",
                        "max ||P(t)-P(0)|| / ||P(0)|| [-]")

        plot_metric_vs_e_for_fixed_mu(
            rows=rows,
            metric_key="eta_E",
            mu_target=0.5,
            normalize=True
        )
