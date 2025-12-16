# dem/time_integration.py

from abc import ABC, abstractmethod
import numpy as np



class TimeIntegration(ABC):
    @abstractmethod
    def pre_force_update(self, particles, delta_t):
        pass

    @abstractmethod
    def post_force_update(self, particles, delta_t):
        pass

# --- Velocity-Verlet integrator ---
class VelocityVerletIntegrator(TimeIntegration):
    def pre_force_update(self, particles, delta_t):
        for p in particles:
            # velocity-verlet time integration step predict
            # previous velocity is required to update velocity
            p.prev_acceleration = p.acceleration.copy()

            # position update
            p.position += p.velocity * delta_t + 0.5 *delta_t**2 * p.acceleration 

            # angle update
            p.rotation += p.omega* delta_t + 0.5 * delta_t**2 * p.angular_acc

            

    def post_force_update(self, particles, delta_t):
        for p in particles:
            # Update velocity
            p.velocity += 0.5 * (p.prev_acceleration + p.acceleration) * delta_t

            # Update angular velocity using Velocity Verlet
            p.omega += 0.5 * (p.prev_angular_acc + p.angular_acc) * delta_t

            # Save for next step
            p.prev_angular_acc = p.angular_acc

def get_time_integration_method(name: str) -> TimeIntegration:
    name = name.lower()
    if name in ["velocity-verlet", "verlet"]:
        return VelocityVerletIntegrator()
    else:
        raise ValueError(f"Unknown integration scheme: {name}")
