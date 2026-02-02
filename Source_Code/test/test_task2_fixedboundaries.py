import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dem')))
import unittest
from particle import Particle
from boundary import Boundary
from contact_model import HertzMindlinDashpot

class TestHertzMindlinPhysics(unittest.TestCase):
    
    def setUp(self):

        self.params = {
            "coeff_of_restitution": 0.7,
            "mu": 0.5
        }
        self.model = HertzMindlinDashpot(self.params)
        self.dt = 0.001  
        self.mat_props = {
            'E': 1e7,
            'nu': 0.3
        }

    def create_test_particle(self, pos, vel, radius=1.0, mass=1.0):
        return Particle(
            position=pos,
            velocity=vel,
            omega=[0, 0, 0],
            radius=radius,
            mass=mass,
            E=self.mat_props['E'],
            nu=self.mat_props['nu'],
            particle_id=1
        )

    def test_no_contact(self):
        p = self.create_test_particle(pos=[0, 0, 10], vel=[0, 0, -1])
        b = Boundary(101, [0, 0, 0], [0, 0, 1], self.mat_props['E'], self.mat_props['nu'], mu_in=0.5)
        
        force, torque = self.model.compute_boundary_contact(p, b, self.dt)
        
        np.testing.assert_array_equal(force, np.zeros(3))
        np.testing.assert_array_equal(torque, np.zeros(3))

    def test_normal_repulsion(self):
        p = self.create_test_particle(pos=[0, 0, 0.9], vel=[0, 0, -1])
        b = Boundary(101, [0, 0, 0], [0, 0, 1], self.mat_props['E'], self.mat_props['nu'])
        
        force, torque = self.model.compute_boundary_contact(p, b, self.dt)

        self.assertGreater(force[2], 0.0)
        self.assertAlmostEqual(force[0], 0.0)

    def test_tangential_stick(self):

        p = self.create_test_particle(pos=[0, 0, 0.9], vel=[0.1, 0, 0])
        b = Boundary(101, [0, 0, 0], [0, 0, 1], self.mat_props['E'], self.mat_props['nu'])
        
        force, _ = self.model.compute_boundary_contact(p, b, self.dt)

        self.assertLess(force[0], 0.0)
        self.assertIn(b.boundary_id, p.tangential_overlaps)

    def test_friction_limit_sliding(self):
        
        # Slideing condition: high tangential velocity
        p = self.create_test_particle(pos=[0, 0, 0.9], vel=[100.0, 0, 0])
        mu_val = 0.2
        b = Boundary(101, [0, 0, 0], [0, 0, 1], self.mat_props['E'], self.mat_props['nu'], mu_in=mu_val)
        
        force, _ = self.model.compute_boundary_contact(p, b, self.dt)
        
        f_normal = force[2]
        f_tangential = abs(force[0])
        
        # test: |Ft| <= mu * Fn
        expected_limit = mu_val * f_normal
        self.assertAlmostEqual(f_tangential, expected_limit, places=4)

if __name__ == '__main__':
    unittest.main()