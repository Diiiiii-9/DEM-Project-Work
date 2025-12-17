# validation/base_validation.py

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import sys
import os

# Add parent directory to path to import 'dem' modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dem.solver import DEMSolver
from dem.particle import Particle
from dem.contact_model import HertzMindlinDashpot
from dem.time_integration import VelocityVerletIntegrator

class BaseValidationScenario(ABC):
    """
    Abstract Base Class for all verification scenarios.
    This defines the standard workflow: Setup -> Run -> Analyze -> Plot.
    """
    def __init__(self, name, duration, dt):
        self.name = name
        self.duration = duration
        self.dt = dt
        self.particles = []
        self.results = {
            "time": [],
            "pos": [],  # Store position history
            "vel": [],  # Store velocity history
            "energy": []
        }
    
    @abstractmethod
    def setup_simulation(self):
        """
        TODO: Implement particle creation and parameter definition here.
        Must populate self.particles.
        """
        pass

    @abstractmethod
    def get_analytical_solution(self, time_array):
        """
        TODO: Return the theoretical values for comparison.
        """
        pass

    def run_simulation(self):
        """
        Standard simulation loop. No need to modify this usually.
        """
        print(f"--- Starting Scenario: {self.name} ---")
        
        # 1. Setup Solver
        self.setup_simulation()

        boundaries = getattr(self, 'boundaries', [])
        
        # Initialize dependencies
        # Note: You might need to customize gravity per scenario in setup, 
        # but here we use a default or allow override.
        contact_model = HertzMindlinDashpot(self.params)
        integrator = VelocityVerletIntegrator()
        
        # Create solver
        solver = DEMSolver(self.particles, contact_model, integrator, self.gravity, boundaries)
        
        # 2. Time Loop
        steps = int(self.duration / self.dt)
        print(f"Simulating {steps} steps...")
        
        for i in range(steps):
            current_time = i * self.dt
            
            # Record Data (Example: recording particle 0)
            self.results["time"].append(current_time)
            self.results["pos"].append(self.particles[0].position.copy())
            self.results["vel"].append(self.particles[0].velocity.copy())
            
            # Advance Step
            solver.solve_time_step(self.dt)
            
        print("Simulation finished.")

    @abstractmethod
    def plot_results(self):
        """
        TODO: Implement plotting logic (Matplotlib) here.
        Compare self.results with analytical solution.
        """
        pass

    def run(self):
        """
        Main execution method.
        """
        self.run_simulation()
        self.plot_results()