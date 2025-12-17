import unittest
import numpy as np
from types import SimpleNamespace

# Assuming the HertzMindlinDashpot class is in a file named contact_model.py
# from contact_model import HertzMindlinDashpot 

# ------------------------------------------------------------------------------
# MOCK CLASSES (To simulate Particle and Boundary behavior)
# ------------------------------------------------------------------------------

class MockParticle:
    def __init__(self, position, velocity, radius, mass, material_props):
        self.id = 1
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.radius = radius
        self.mass = mass
        self.omega = np.zeros(3)  # Angular velocity
        
        # Material properties
        self.E = material_props['E']
        self.nu = material_props['nu']
        
        # Storage for history
        self.tangential_overlaps = {} 

class MockBoundary:
    def __init__(self, point, normal, velocity, material_props):
        self.boundary_id = 101
        self.point = np.array(point, dtype=float)   # A point on the plane
        self.normal = np.array(normal, dtype=float) # Normal vector
        self.velocity = np.array(velocity, dtype=float)
        
        # Material properties
        self.E = material_props['E']
        self.nu = material_props['nu']
        self.mu = material_props['mu']

# ------------------------------------------------------------------------------
# UNIT TEST SUITE
# ------------------------------------------------------------------------------

class TestHertzMindlinBoundary(unittest.TestCase):
    
    def setUp(self):
        """
        Setup run before every test.
        Initializes the model and common parameters.
        """
        # Model parameters
        self.params = {
            "coeff_of_restitution": 0.7,
            "mu": 0.5
        }
        
        # Instantiate the model (Assuming the class is available in scope)
        # In your actual project, import the class properly.
        # self.model = HertzMindlinDashpot(self.params) 
        
        # --- FOR DEMONSTRATION, I will define a wrapper to simulate the instance ---
        # In real usage, verify your import matches this structure.
        class HertzMindlinDashpotWrapper:
            def __init__(self, params):
                self.params = params
                self.mu_friction = params['mu']
                self.coeff_of_restitution = params['coeff_of_restitution']
            
            # Helper to calculate damping coefficient (copied from your logic)
            def GammaForHertzThornton(self):
                # Simplified for test stability, or copy exact logic if needed
                # Ideally, testing the math should use the real method.
                # Here we assume a fixed small value for testing flow.
                return 0.1 

            # Paste the provided method here or bind it
            # compute_boundary_contact = <The Method You Provided>
        
        self.model = HertzMindlinDashpotWrapper(self.params)
        
        # Inject the method provided by Qinfei into the class for testing
        # (This is just a trick to run this script standalone; in your project, just import it)
        from types import MethodType
        # We assume the method 'compute_boundary_contact' is defined in the file context
        # Since I cannot import your file, I assume the method provided in the prompt
        # is part of the class. 
        # Please ensure the method is indented inside the class in your actual code.

    def test_no_contact(self):
        """
        Test Case 1: Particle is far away from the boundary.
        Expectation: Zero force, Zero torque.
        """
        # Particle at z=10, Boundary at z=0, Radius=1. No overlap.
        p = MockParticle([0, 0, 10], [0, 0, -1], 1.0, 1.0, {'E': 1e7, 'nu': 0.3})
        b = MockBoundary([0, 0, 0], [0, 0, 1], [0, 0, 0], {'E': 1e7, 'nu': 0.3, 'mu': 0.5})
        
        # Inject method logic locally for the test runner to work 
        # (In your code, you just call self.model.compute_boundary_contact)
        F, T = self.compute_boundary_contact_proxy(self.model, p, b, 0.01)
        
        np.testing.assert_array_equal(F, np.zeros(3), "Force should be zero when no contact")
        np.testing.assert_array_equal(T, np.zeros(3), "Torque should be zero when no contact")

    def test_normal_repulsion(self):
        """
        Test Case 2: Particle penetrates boundary vertically (Normal Force).
        Expectation: Positive Force in Z direction (repelling the particle).
        """
        # Boundary at z=0, Normal=[0,0,1]
        # Particle at z=0.9, Radius=1.0 -> Overlap = 0.1
        p = MockParticle([0, 0, 0.9], [0, 0, -1], 1.0, 1.0, {'E': 1e7, 'nu': 0.3})
        b = MockBoundary([0, 0, 0], [0, 0, 1], [0, 0, 0], {'E': 1e7, 'nu': 0.3, 'mu': 0.5})
        
        F, T = self.compute_boundary_contact_proxy(self.model, p, b, 0.01)
        
        # Force should be [0, 0, +Value]
        self.assertAlmostEqual(F[0], 0.0)
        self.assertAlmostEqual(F[1], 0.0)
        self.assertGreater(F[2], 0.0, "Force should push particle upwards")
        
        # Since contact is purely vertical through center, Torque should be zero
        np.testing.assert_allclose(T, np.zeros(3), atol=1e-10)

    def test_tangential_stick(self):
        """
        Test Case 3: Particle has horizontal velocity (Shear).
        Expectation: Tangential force opposes velocity, stored in history.
        """
        # Particle touching boundary, moving +X
        p = MockParticle([0, 0, 0.9], [0.1, 0, 0], 1.0, 1.0, {'E': 1e7, 'nu': 0.3})
        b = MockBoundary([0, 0, 0], [0, 0, 1], [0, 0, 0], {'E': 1e7, 'nu': 0.3, 'mu': 0.5})
        
        dt = 0.001
        F, T = self.compute_boundary_contact_proxy(self.model, p, b, dt)
        
        # Tangential force should be in -X direction (opposing motion)
        self.assertLess(F[0], 0.0, "Tangential force should oppose velocity")
        
        # Check if history is updated
        self.assertIn(b.boundary_id, p.tangential_overlaps, "Tangential overlap should be stored")
        # Overlap should be roughly v * dt in negative direction (or positive depending on sign convention)
        # Logic: delta_s = v_rel * dt. 
        self.assertNotEqual(np.linalg.norm(p.tangential_overlaps[b.boundary_id]), 0)

    def test_friction_limit_sliding(self):
        """
        Test Case 4: Coulomb Friction Limit (Sliding).
        Expectation: Tangential force magnitude should be capped at mu * F_normal.
        """
        # Very high velocity to force sliding
        p = MockParticle([0, 0, 0.9], [100.0, 0, 0], 1.0, 1.0, {'E': 1e7, 'nu': 0.3})
        # Low friction coefficient
        mu_b = 0.1
        b = MockBoundary([0, 0, 0], [0, 0, 1], [0, 0, 0], {'E': 1e7, 'nu': 0.3, 'mu': mu_b})
        
        F, T = self.compute_boundary_contact_proxy(self.model, p, b, 0.001)
        
        F_n_mag = F[2] # Normal force (Z)
        F_t_mag = abs(F[0]) # Tangential force (X)
        
        # Allow small numerical tolerance
        expected_limit = mu_b * F_n_mag
        
        # Check if F_t is clamped to the limit
        self.assertAlmostEqual(F_t_mag, expected_limit, places=5, 
                               msg="Tangential force should be capped by Coulomb friction")

    # --------------------------------------------------------------------------
    # Helper to run the provided code snippet logic
    # (In your real file, you don't need this, you call the method directly)
    # --------------------------------------------------------------------------
    def compute_boundary_contact_proxy(self, model, p, boundary, delta_t):
        """
        This contains Qinfei's exact logic for testing purposes.
        """
        vec_to_boundary = p.position - boundary.point
        distance = np.dot(vec_to_boundary, boundary.normal)
        interpenetration = p.radius - distance
        
        if interpenetration <= 0:
            if boundary.boundary_id in p.tangential_overlaps:
                del p.tangential_overlaps[boundary.boundary_id]
            return np.zeros(3), np.zeros(3)
            
        n = boundary.normal
        interpenetration_vel = -np.dot(p.velocity, n)

        # Equivalent properties
        R_eq = p.radius
        m_eq = p.mass
        E_eq = 1 / ((1 - p.nu**2)/p.E + (1 - boundary.nu**2)/boundary.E)

        G1 = p.E / (2 * (1 + p.nu))
        G2 = boundary.E / (2 * (1 + boundary.nu))
        G_eq = 1 / ((2 - p.nu)/G1 + (2 - boundary.nu)/G2)

        # Normal Force
        xi = model.GammaForHertzThornton()

        k_n = 2 * E_eq * np.sqrt(R_eq * interpenetration)
        c_n = 2 * xi * np.sqrt(m_eq * k_n)

        F_n_mag = (2/3) * k_n * interpenetration + c_n * interpenetration_vel

        if F_n_mag < 0: 
            F_n_mag = 0.0
        F_n = F_n_mag * n 

        # Lever arm & Velocity
        x_ip = (p.radius - interpenetration) * n 
        v_ip = p.velocity + np.cross(p.omega, x_ip)

        v_rel = v_ip - boundary.velocity
        v_rel_n = np.dot(v_rel, n) * n
        v_rel_t = v_rel - v_rel_n
        v_rel_t[2] = 0.0 

        k_t = 8 * G_eq * np.sqrt(R_eq * interpenetration)
        c_t = 2 * xi * np.sqrt(m_eq * k_t) 

        if not hasattr(p, 'tangential_overlaps'):
            p.tangential_overlaps = {}
        delta_t_vec = p.tangential_overlaps.get(boundary.boundary_id, np.zeros(3))

        delta_t_vec += v_rel_t * delta_t
        F_t_trial = -k_t * delta_t_vec - c_t * v_rel_t

        mu_effective = min(model.mu_friction, getattr(boundary, 'mu', 0.5))

        F_t_mag_trial = np.linalg.norm(F_t_trial)
        F_n_mag_abs = np.abs(F_n_mag)
        F_t_limit = mu_effective * F_n_mag_abs

        F_t = np.zeros(3)
        if F_t_mag_trial > F_t_limit and F_t_mag_trial > 1e-12:
            # Sliding
            ratio = F_t_limit / F_t_mag_trial
            F_t = F_t_trial * ratio
            delta_t_vec = delta_t_vec * ratio
        else:
            # Stick
            F_t = F_t_trial
            
        p.tangential_overlaps[boundary.boundary_id] = delta_t_vec
        F_total = F_n + F_t
        torque = np.cross(x_ip, F_t)
        
        return F_total, torque

if __name__ == '__main__':
    unittest.main()