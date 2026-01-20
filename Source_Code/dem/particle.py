# dem/particle.py

import numpy as np

class Particle:
    def __init__(self, 
                 position, 
                 velocity,
                 omega, 
                 radius, 
                 mass, 
                 E,
                 nu, 
                 particle_id=None):
        """
        Initialize a 2D spherical particle.
        """
        self.id = particle_id
        self.radius = radius
        self.mass = mass
        self.E =E
        self.nu = nu
        self.inertia = 0.5 * mass * radius**2               # 2D disc moment of inertia
        self.kn = 2* self.E * np.sqrt(self.radius)

        self.position = np.array(position, dtype=float)     # 2D vector
        self.velocity = np.array(velocity, dtype=float)     # 2D vector
        self.acceleration = np.zeros(3, dtype=float)        # 2D vector
        self.prev_acceleration = np.zeros(3, dtype=float)   # Needed for Verlet

        self.rotation = np.zeros(3, dtype=float)                               # Scalar angle (theta)
        self.omega = np.array(omega, dtype=float)                   # Angular velocity
        self.angular_acc  = np.zeros(3, dtype=float)              # Angular acceleration
        self.prev_angular_acc = np.zeros(3, dtype=float)   # Needed for Verlet

        self.force = np.zeros(3, dtype=float)               # Force
        self.torque = np.zeros(3, dtype=float)                                  # Torque

        # key: neighbor_particle_id, value: numpy array (tangential vector) - for task 1
        self.tangential_overlaps = {}

        
    def reset_force(self):
        self.force = np.zeros(3, dtype=float)
        self.torque = np.zeros(3, dtype=float)

    def __repr__(self):
        return (f"Particle(id={self.id}, pos={self.position}, vel={self.velocity}, "
                f"radius={self.radius}, mass={self.mass})")
