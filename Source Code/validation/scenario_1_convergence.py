# Particle bouncing on a rigid floor scenario
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from dem.particle import Particle
from dem.boundary import Boundary
from dem.solver import DEMSolver
from dem.contact_model import get_contact_model
from dem.time_integration import get_time_integration_method
from results.plot_utils import plot_trajectories, plot_velocities

from results.plot_field import plot_field
import matplotlib.tri as mtri

def run_bouncing_simulation(E, delta_t, t_end=4.0):
    # --- Contact model parameters ---
    coeff_of_restitution = 1.0
    mu_friction = 0.0

    # --- Particle parameters ---
    mass = 1.0
    nu = 0.0
    radius = 0.5

    # --- External load ---
    gravity = np.array([0.0, -9.81, 0.0])

    # --- Contact and integration ---
    contact_params = {
        "coeff_of_restitution": coeff_of_restitution,
        "mu": mu_friction
    }

    contact_model = get_contact_model("HM+D", contact_params)
    time_integration = get_time_integration_method("velocity-verlet")

    # --- Particle ---
    particle = Particle(
        position=[5.0, 1.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        radius=radius,
        mass=mass,
        E=E,
        nu=nu,
        particle_id=1
    )

    # --- Boundary ---
    floor_boundary = Boundary(
        boundary_id_in=1,
        point_in=np.array([0.0, 0.0, 0.0]),
        normal_in=np.array([0.0, 1.0, 0.0]),
        E_in=1e10,
        nu_in=0.0,
        mu_in=mu_friction
    )

    solver = DEMSolver(
        [particle],
        contact_model,
        time_integration,
        gravity,
        boundaries=[floor_boundary]
    )

    # --- Logs ---
    times = []
    y_pos = []
    y_vel = []

    t = 0.0
    while t < t_end:
        solver.solve_time_step(delta_t)
        times.append(t)
        y_pos.append(particle.position[1])
        y_vel.append(particle.velocity[1])
        t += delta_t

    return np.array(times), np.array(y_pos), np.array(y_vel)

def compute_apex_heights(y, radius):
    apex_indices = np.where(
        (y[1:-1] > y[:-2]) &
        (y[1:-1] > y[2:])
    )[0] + 1

    return y[apex_indices] - radius  # height above contact

def compute_restitution(y_vel):
    restitution = []
    for i in range(1, len(y_vel) - 1):
        if y_vel[i-1] < 0 and y_vel[i+1] > 0:
            restitution.append(-y_vel[i+1] / y_vel[i-1])
    return np.array(restitution)

def compute_measured_restitution(vy):
    e_measured = []

    for i in range(1, len(vy)-1):
        if vy[i-1] < 0.0 and vy[i+1] > 0.0:
            e_measured.append(-vy[i+1] / vy[i-1])

    return np.array(e_measured)

def run_bouncing_simulation_restitution(e, delta_t, t_end=4.0, E=1e4):
    # --- Parameters ---
    mass = 1.0
    nu = 0.0
    radius = 0.5
    gravity = np.array([0.0, -9.81, 0.0])
    E = 10000

    contact_params = {
        "coeff_of_restitution": e,
        "mu": 0.0
    }

    contact_model = get_contact_model("HM+D", contact_params)
    time_integration = get_time_integration_method("velocity-verlet")

    particle = Particle(
        position=[0.0, 1.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        radius=radius,
        mass=mass,
        E=E,
        nu=nu,
        particle_id=1
    )

    floor = Boundary(
        boundary_id_in=1,
        point_in=np.array([0.0, 0.0, 0.0]),
        normal_in=np.array([0.0, 1.0, 0.0]),
        E_in=1e10,
        nu_in=0.0,
        mu_in=0.0
    )

    solver = DEMSolver([particle], contact_model, time_integration, gravity, boundaries=[floor])

    times, y_pos, y_vel = [], [], []

    t = 0.0
    while t < t_end:
        solver.solve_time_step(delta_t)
        times.append(t)
        y_pos.append(particle.position[1])
        y_vel.append(particle.velocity[1])
        t += delta_t

    return np.array(times), np.array(y_pos), np.array(y_vel)

def restitution_sensitivity_analysis():
    e_values = [0.3, 0.5, 0.7, 0.9]
    delta_t = 1e-4
    radius = 0.5

    plt.figure(figsize=(14, 4))

    # --- Plot 1: Apex height vs bounce index ---
    plt.subplot(1, 3, 1)
    for e in e_values:
        print("e: ", e)
        _, y, _ = run_bouncing_simulation_restitution(e, delta_t)
        h = compute_apex_heights(y, radius)
        plt.plot(range(len(h)), h, 'o-', label=f"e = {e}")

    plt.xlabel("Bounce index k")
    plt.ylabel("Apex height above floor [m]")
    plt.title("Apex height decay")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # --- Plot 2: Log(apex height) vs bounce index ---
    plt.subplot(1, 3, 2)
    for e in e_values:
        _, y, _ = run_bouncing_simulation_restitution(e, delta_t)
        h = compute_apex_heights(y, radius)
        h = h[h > 0]  # avoid log(0)
        plt.plot(range(len(h)), np.log(h), 'o-', label=f"e = {e}")

    plt.xlabel("Bounce index k")
    plt.ylabel("log(h_k)")
    plt.title("Log-linear decay (theory: slope = 2 log e)")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # --- Plot 3: Measured restitution per bounce ---
    plt.subplot(1, 3, 3)
    for e in e_values:
        _, _, vy = run_bouncing_simulation_restitution(e, delta_t)
        e_meas = compute_measured_restitution(vy)
        plt.plot(range(len(e_meas)), e_meas, 'o-', label=f"e = {e}")

    plt.xlabel("Bounce index k")
    plt.ylabel("Measured restitution")
    plt.title("Restitution consistency")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def generate_phase_space_data(
        E_values,
        e_values,
        bounce_index=1,
        delta_t=1e-4,
        t_end=4.0,
        radius=0.5
    ):
    """
    Generates scattered data points (E, e, h_k) where h_k
    is the apex height after bounce_index.
    """

    E_list = []
    e_list = []
    h_list = []

    for E in E_values:
        for e in e_values:
            print("E, e: ", E, e)
            _, y, _ = run_bouncing_simulation_restitution(e, delta_t, t_end=t_end, E=E)
            h = compute_apex_heights(y, radius)

            if len(h) > bounce_index:
                print("inside")
                E_list.append(E)
                e_list.append(e)
                h_list.append(h[bounce_index])

    return np.array(E_list), np.array(e_list), np.array(h_list)

def plot_phase_diagram(E, e, h, bounce_index):
    """
    Creates a phase diagram using Delaunay triangulation
    """

    triang = mtri.Triangulation(E, e)

    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(
        triang,
        h,
        levels=20,
        cmap="viridis"
    )

    plt.colorbar(contour, label=f"Apex height after bounce {bounce_index} [m]")
    plt.xlabel("Young's modulus E")
    plt.ylabel("Coefficient of restitution e")
    plt.title(f"Phase diagram: Apex height after bounce {bounce_index}")

    plt.xscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def phase_space_analysis():
    E_values = np.logspace(3, 5, 6)      # 1e3 → 1e5
    e_values = np.linspace(0.3, 0.95, 8) # restitution range
    bounce_index = 1                     # second bounce (k=1)

    E, e, h = generate_phase_space_data(
        E_values,
        e_values,
        bounce_index=bounce_index
    )

    plot_phase_diagram(E, e, h, bounce_index)

def run_bouncing_simulation_contact(e, delta_t, t_end=4.0, E=1e4):
    mass = 1.0
    nu = 0.0
    radius = 0.5
    gravity = np.array([0.0, -9.81, 0.0])

    contact_params = {
        "coeff_of_restitution": e,
        "mu": 0.0
    }

    contact_model = get_contact_model("HM+D", contact_params)
    time_integration = get_time_integration_method("velocity-verlet")

    particle = Particle(
        position=[0.0, 1.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        omega=[0.0, 0.0, 0.0],
        radius=radius,
        mass=mass,
        E=E,
        nu=nu,
        particle_id=1
    )

    floor = Boundary(
        boundary_id_in=1,
        point_in=np.array([0.0, 0.0, 0.0]),
        normal_in=np.array([0.0, 1.0, 0.0]),
        E_in=1e10,
        nu_in=0.0,
        mu_in=0.0
    )

    solver = DEMSolver([particle], contact_model, time_integration, gravity, boundaries=[floor])

    times, y_pos, y_vel, overlap = [], [], [], []

    t = 0.0
    while t < t_end:
        solver.solve_time_step(delta_t)

        y = particle.position[1]
        delta = max(0.0, radius - y)

        times.append(t)
        y_pos.append(y)
        y_vel.append(particle.velocity[1])
        overlap.append(delta)

        t += delta_t

    return (
        np.array(times),
        np.array(y_pos),
        np.array(y_vel),
        np.array(overlap)
    )

def compute_contact_duration(times, overlap):
    """
    Returns contact duration of the FIRST resolved impact.
    """

    in_contact = overlap > 0.0

    if not np.any(in_contact):
        return np.nan

    indices = np.where(in_contact)[0]

    t_start = times[indices[0]]
    t_end   = times[indices[-1]]

    return t_end - t_start

def compute_max_overlap(overlap):
    if np.any(overlap > 0.0):
        return np.max(overlap)
    return np.nan

def generate_phase_data_contact_duration(E_values, e_values, delta_t=1e-4):
    E_list, e_list, tc_list = [], [], []

    for E in E_values:
        for e in e_values:
            print("E, e", E, e)
            t, _, _, overlap = run_bouncing_simulation_contact(e, delta_t, E=E)
            tc = compute_contact_duration(t, overlap)

            if not np.isnan(tc):
                E_list.append(E)
                e_list.append(e)
                tc_list.append(tc)

    return np.array(E_list), np.array(e_list), np.array(tc_list)

def generate_phase_data_max_overlap(E_values, e_values, delta_t=1e-4):
    E_list, e_list, delta_list = [], [], []

    for E in E_values:
        for e in e_values:
            print("E, e", E, e)
            _, _, _, overlap = run_bouncing_simulation_contact(e, delta_t, E=E)
            delta_max = compute_max_overlap(overlap)

            if not np.isnan(delta_max):
                E_list.append(E)
                e_list.append(e)
                delta_list.append(delta_max)

    return np.array(E_list), np.array(e_list), np.array(delta_list)

def plot_phase_diagram(E, e, Z, label, title):
    triang = mtri.Triangulation(E, e)

    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(triang, Z, levels=20, cmap="viridis")

    plt.colorbar(contour, label=label)
    plt.xlabel("Young's modulus E")
    plt.ylabel("Coefficient of restitution CoR")
    plt.xscale("log")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():

    # 1st PART
    # Study the influence of E values on the solution
    FirstPart = True
    if FirstPart:
        E_values = [1e3, 1e5]
        delta_t = 1e-4
        t_end = 2.0

        plt.figure()
        for E in E_values:
            t, y, v = run_bouncing_simulation(E, delta_t, t_end=t_end)
            plt.plot(t, abs(v), label=f"E = {E:.0e}")

        plt.xlabel("Time [s]")
        plt.ylabel("Velocity v [m/s]")
        plt.legend()
        plt.title("Effect of Young's modulus on bouncing motion")
        plt.grid()
        plt.show()

    # 2nd PART
    SecondPart = True
    # Study the convergence of error with respect to time step
    if SecondPart:
        E = 1e4
        dt_values = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5]
        apex_errors = []

        reference_dt = dt_values[-1]
        _, y_ref, _ = run_bouncing_simulation(E, reference_dt)
        apex_ref = compute_apex_heights(y_ref, 0.5).mean()

        for dt in dt_values[:-1]:
            _, y, _ = run_bouncing_simulation(E, dt)
            apex = compute_apex_heights(y, 0.5).mean()
            apex_errors.append(abs(apex - apex_ref))

        plt.loglog(dt_values[:-1], apex_errors, 'o-')
        plt.xlabel("Time step Δt [s]")
        plt.ylabel("Apex height error [m]")
        plt.title("Time-step convergence of bouncing particle")
        plt.grid(which="both")
        plt.show()

    # 3rd PART
    ThirdParth = True
    # sensitivity analysis on the value of restitution
    if ThirdParth:
        restitution_sensitivity_analysis()

    # 5th PART
    FourthPart = True
    # countour plot for sensitivity analysis on pair of values (coeff_restitution, E)
    if FourthPart:

        ## For height apex
        #phase_space_analysis()

        ## for contact duration
        E_vals = np.logspace(3, 5, 6)
        e_vals = np.linspace(0.3, 0.95, 8)

        #E, e, tc = generate_phase_data_contact_duration(E_vals, e_vals)
        #plot_phase_diagram(
        #    E, e, tc,
        #    label="Contact duration [s]",
        #    title="Phase plot: Contact duration vs (E, e)"
        #)

        ## for maximum overlap
        E, e, delta = generate_phase_data_max_overlap(E_vals, e_vals)
        plot_phase_diagram(
            E, e, delta,
            label="Maximum overlap [m]",
            title="Phase plot: Maximum overlap vs (E, CoR)"
        )


# --- Run the main function ---
if __name__ == "__main__":
    main()