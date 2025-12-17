# dem/boundary.py
# rigid boundaries in 2D
import numpy as np

class Boundary:
    def __init__(self, boundary_id_in, point_in, normal_in, E_in, nu_in, mu_in=0.5):
        """
        initialize boundary properties
        the boundary is defined by a point and a normal vector
        For rigid boundaries, we assume infinite mass and stiffness (just very large) and fixed position and just never update them
        """
        self.boundary_id = boundary_id_in      # unique identifier
        self.point = np.array(point_in)        # a point on the boundary, used to define position and distance
        self.normal = np.array(normal_in) / np.linalg.norm(normal_in)     # normal vector (unit vector)
        self.E = E_in          # Young's modulus of the boundary material
        self.nu = nu_in        # Poisson's ratio of the boundary material
        self.mu = mu_in        # Coefficient of friction of the boundary material
        self.velocity = np.zeros(3)     # Rigid boundary assumed stationary
        self.omega = np.zeros(3)        # No rotation for rigid boundary

    def __repr__(self):
        return f"Boundary(ID={self.boundary_id}, Point={self.point}, Normal={self.normal})"