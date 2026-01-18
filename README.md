
# DEM Project Work - Particle Methods in Engineering

**Technical University of Munich (TUM)**
**Chair of Structural Analysis**
**Course:** Particle Methods in Engineering (Winter Semester 2025)

## 📌 Project Overview

This project is a Python implementation of the **Discrete Element Method (DEM)** for simulating granular material mechanics. It is developed as part of the "Particle Methods in Engineering" course coursework.

The goal of this project is to deepen the understanding of DEM theory and numerical concepts by extending a basic educational code framework. The extensions focus on realistic contact mechanics (tangential forces, friction) and environmental interactions (rigid boundaries), followed by rigorous verification and validation (V&V).

## ✨ Key Features & Tasks

This software implements a 2D DEM solver with the following core functionalities and extensions derived from the project requirements:

### 🔹 Core Functionality
* **Particle Representation**: 2D spherical particles with mass, radius, and material properties ($E$, $\nu$).
* **Time Integration**: Explicit **Velocity-Verlet** scheme for translational and rotational equations of motion.
* **Contact Detection**: Efficient neighbor searching and overlap detection.
* **Visualization**: Real-time animation of particle trajectories and post-processing plots (velocity, force, torque).

### 🚀 Implemented Extensions

#### Task 1: Tangential Contact & Friction
* **Hertz-Mindlin + Dashpot (HM+D) Model**: Implemented tangential contact force contributions based on relative tangential velocity at the contact point.
* **Coulomb Friction**: Modeled slip conditions where the elastic tangential force is capped by the friction limit ($\mu F_n$).
* **Rotational Physics**: Computed torque resulting from tangential forces and integrated it into the particle's rotational dynamics.

#### Task 2: Particle-Boundary Interaction
* **Rigid Boundaries**: Introduction of 2D rigid walls (lines/planes) acting as boundaries.
* **Wall Contact Resolution**: Adapted the HM+D contact law to handle particle-wall interactions (treating walls as stationary objects with infinite mass).
* **Boundary Forces**: Computation of normal and tangential reaction forces and resulting torques on particles upon collision with boundaries.

#### Task 3: Verification & Validation
The implementation is verified through specific test scenarios:
1.  **Particle-Particle Collision**: Oblique impact verification with friction effects.
2.  **Bouncing Ball**: Energy conservation and restitution coefficient checks against a rigid floor.
3.  **Sliding Friction**: A block/particle sliding on a frictional surface to verify the Coulomb limit.

## 📂 Project Structure

```text
DEM-Project-Work/
├── Document/                 # Project assignment and theoretical background
│   ├── Project Work Topic DEM.pdf
│   └── Project Work Information.pdf
├── Source Code/              # Source code for the DEM solver
│   ├── MainDEM.py            # Entry point: Configuration and Simulation Loop
│   ├── dem/                  # Core library
│   │   ├── particle.py       # Particle class (properties, state)
│   │   ├── solver.py         # Main DEM solver class (force loop)
│   │   ├── contact_model.py  # Contact laws (Hertz-Mindlin, Friction)
│   │   └── time_integration.py # Velocity-Verlet integrator
│   ├── results/              # Visualization and logging tools
│   │   ├── plot_utils.py     # Plotting functions using Matplotlib
│   │   ├── plot_field.py     # Plotting fields using Matplotlib 
│   │   └── Plot_Results/     # Folder containing results for exmaple scenarios
│   ├── test/                 # Folder containing unittests for implementation of Task 1 & 2 
│   │   └── ...
│   ├── validation/           # Scenarios for Task 3 and Analysis of Results
│   │   ├── NumericalResults/ # Folder containing analysis for scenarios
│   │   ├── scenario_1_bouncing.py # File with exemplar implementation of scenario
│   │   ├── scenario_1_convergence.py # Convergence and Analysis of such scenario
│   │   └── ...
├── LICENSE                   # MIT License
└── README.md                 # Project documentation

```

## ⚙️ InstallationTo run this project, you need **Python 3.x** and the following scientific computing libraries:

* **NumPy**: For vector and matrix operations.
* **Matplotlib**: For plotting and animation.

You can install the dependencies using pip:

```bash
pip install numpy matplotlib

```

## 🚀 Usage1. **Navigate to the Source Code directory:**
```bash
cd "Source Code"

```


2. **Run the simulation:**
```bash
python MainDEM.py

```


3. **Configuration:**
You can modify simulation parameters directly in `MainDEM.py` under the `Inputs` section:
* `coeff_of_restitution`: Coefficient of restitution (e).
* `mu_friction`: Coefficient of friction (\mu).
* `particles`: Initial positions, velocities, and properties of particles.
* `delta_t`: Time step size.

4. **Scenarios:**
You can also run the different scenarios by entenring the validation/ folder and using the following command:
```bash
python scenario_1_bouncing.py
```
This is also available for any other scenario, as well for the convergence and results analysis:
```bash
python scenario_1_convergence.py
```

## 📊 Theory

### Contact Model (Hertz-Mindlin)
The contact force $\mathbf{F}$ is decomposed into normal ($F_n$) and tangential ($F_t$) components:
* **Normal Force**: Based on the non-linear Hertz theory with a dissipative dashpot term.
* **Tangential Force**: Incremental spring-dashpot model bounded by the Coulomb friction limit $|F_t| \le \mu |F_n|$.

### Time Integration
The simulation uses the **Velocity-Verlet** algorithm, a symplectic integrator that offers good stability and energy conservation properties for N-body systems.

## Task 3: Verification and Validation Strategy

To assess the correctness and robustness of the implemented **Hertz-Mindlin (HM+D) contact model** (Task 1) and **Particle-Wall boundaries** (Task 2), we have devised four test scenarios. These scenarios compare numerical results against analytical solutions to ensure physical plausibility.

We have split the workload to cover both particle-particle and particle-wall interactions efficiently.

### Work Division Plan

| Scenario | Description | Type | Assignee |
| :--- | :--- | :--- | :--- |
| **1, 3** | Vertical Bouncing on Floor & Rolling on Inclined Plane | Implementation of Tools | **Qinfei** |
| **2, 4** | Horizontal Sliding with Friction & Oblique 2-Particle Collision | Implementation of Tools | **Di** |
| **1,2,3,4** | Analysis of All Scenarios | Convergence and Results Analysis | **Eduardo** |

---

### Detailed Test Scenarios

#### 1. Vertical Bouncing (Verification of Normal Force & COR)
**Goal:** Verify the implementation of the coefficient of restitution ($e$) and normal wall contact forces.
* **Setup:** A single particle drops from height $h_0$ onto a fixed horizontal floor (gravity $g$ active).
* **Theoretical Expectation:**
    * Rebound height: $h_{final} = e^2 \cdot h_0$
    * Rebound velocity: $v_{out} = -e \cdot v_{in}$


#### 2. Horizontal Sliding (Verification of Coulomb Friction)
**Goal:** Validate that the tangential force is correctly capped by the Coulomb limit ($F_t \le \mu F_n$) during sliding.
* **Setup:** A particle is given an initial horizontal velocity $v_0$ on a floor with friction coefficient $\mu$. Gravity $g$ is active to provide normal force. Rotation is fixed (or ignored) to ensure pure sliding.
* **Theoretical Expectation:**
    * The particle should undergo constant deceleration: $a = -\mu \cdot g$
    * Stopping distance: $d = \frac{v_0^2}{2 \mu g}$


#### 3. Rolling on Inclined Plane (System Validation)
**Goal:** Verify the coupling between tangential forces and torque generation (rotational equations of motion).
* **Setup:** A particle is released from rest on a rigid wall inclined at angle $\alpha$.
* **Theoretical Expectation:**
    * For a 2D disc (Moment of Inertia $I = 0.5 m r^2$) under pure rolling condition:
    * Linear acceleration of center of mass: $a_{CM} = \frac{2}{3} g \sin(\alpha)$


#### 4. Oblique Particle-Particle Collision
**Goal:** Verify tangential force generation and momentum conservation between two particles.
* **Setup:** Two identical particles collide at an off-center angle (impact parameter $b > 0$).
* **Theoretical Expectation:**
    * Post-collision, both particles must acquire angular velocity ($\omega \neq 0$) due to tangential friction.
    * Total linear momentum of the system must be conserved.


## 👥 Contributors

* **Di Liu** - *Task 1: Tangential Contact Force, Friction Model & Rotational Dynamics*
* **Qinfei Ran** - *Task 2: Particle-Boundary Contact (Walls) & Solver Extensions*
* **Eduardo Silva** - *Task 3: Implementation of Analysis of Scenarios & Comparison of Analytical and Numerical Results*

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

