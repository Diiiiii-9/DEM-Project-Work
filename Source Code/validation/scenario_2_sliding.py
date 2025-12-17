# validation/scenario_2_sliding.py

import numpy as np
import matplotlib.pyplot as plt
from base_validation import BaseValidationScenario
from dem.particle import Particle

class SlidingFrictionScenario(BaseValidationScenario):
    def setup_simulation(self):
        """
        Implementation of the specific setup for Sliding Friction.
        """
        # 1. Define Parameters
        self.gravity = np.array([0, -9.81, 0])
        self.params = {
            "coeff_of_restitution": 0.8,
            "mu": 0.3  # The friction coefficient we want to test
        }
        
        # 2. Create Particles
        # Example: One particle on the ground
        # TODO: Define correct initial velocity v0 for sliding
        p1 = Particle(
            position=[0, 0, 0],  
            velocity=[5.0, 0, 0], # Initial slide speed
            omega=[0, 0, 0],
            radius=0.1,
            mass=1.0,
            E=1e7,
            nu=0.3,
            particle_id=1
        )
        self.particles = [p1]
        
        # TODO: Remember to add the WALL/FLOOR in your Solver logic 
        # (Since Task 2 is needed for this, ensure your solver handles boundaries 
        # or simulate a "floor" using a very large fixed particle if Task 2 isn't ready)

    def get_analytical_solution(self, time_array):
        """
        Calculate expected velocity/position based on Coulomb friction.
        v(t) = v0 - mu * g * t
        """
        mu = self.params["mu"]
        g = 9.81
        v0 = 5.0 # Must match setup
        
        # TODO: Implement the formula
        # velocity_analytical = ...
        # position_analytical = ...
        
        return None # Return the arrays

    def plot_results(self):
        """
        Compare Numerical vs Analytical.
        """
        time_sim = np.array(self.results["time"])
        vel_sim = np.array([v[0] for v in self.results["vel"]]) # X-velocity
        
        # Get theory
        # vel_theory = self.get_analytical_solution(time_sim)
        
        plt.figure(figsize=(10, 6))
        plt.plot(time_sim, vel_sim, 'b-', label='DEM Simulation')
        # plt.plot(time_sim, vel_theory, 'r--', label='Analytical Solution')
        
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity X [m/s]')
        plt.title('Validation Scenario 2: Sliding Friction')
        plt.legend()
        plt.grid(True)
        
        # Save or Show
        plt.savefig('scenario_2_result.png')
        plt.show()

if __name__ == "__main__":
    # Create and run the scenario
    # Duration 1.0s, Time step 0.001s
    scenario = SlidingFrictionScenario("Sliding Test", duration=1.0, dt=1e-4)
    scenario.run()