# dem/contact_model.py

from abc import ABC, abstractmethod
import numpy as np

def get_contact_model(name, params):
    """
    Factory function to create contact model instances.
    """
    if name.lower() in ["hm+d", "hertz", "hertzmindlin", "hertz-mindlin"]:
        return HertzMindlinDashpot(params)
    else:
        raise ValueError(f"Unknown contact model: {name}")

class ContactModel(ABC):
    def __init__(self, params):
        self.params = params 

    @abstractmethod
    def compute_contact(self, p1, p2, delta_t):
        """
        Calculate contact force and torque on p1 due to p2.
        Returns:
            force_on_p1 (np.array)
            torque_on_p1 (np.array)
            torque_on_p2 (np.array)
        """
        pass

class HertzMindlinDashpot(ContactModel):
    def __init__(self, params):
        super().__init__(params)
        self.coeff_of_restitution = params.get("coeff_of_restitution", 0.7) # Default safely
        self.mu_friction = params.get("mu", 0.3)

    def compute_contact(self, p1, p2, delta_t):
        """
        Implementation of the Hertz-Mindlin + Dashpot model with Coulomb friction.
        """
        
        # --- 1. Geometric Calculations ---
        r_vec = p2.position - p1.position
        l_c = np.linalg.norm(r_vec)

        # Handle zero distance to avoid division by zero (though unlikely in DEM)
        if l_c == 0:
            return np.zeros(3), np.zeros(3), np.zeros(3)

        # Normal vector pointing from p1 to p2
        n = r_vec / l_c
        n[2] = 0.0 # Enforce 2D calculation in XY plane if needed

        # Overlap (interpenetration) calculation
        interpenetration = p1.radius + p2.radius - np.dot((p2.position - p1.position), n)

        # --- 2. Check for Contact Loss ---
        # Initialize storage for tangential history if it doesn't exist
        if not hasattr(p1, 'tangential_overlaps'):
            p1.tangential_overlaps = {}

        # If particles are not touching
        if interpenetration <= 0:
            # IMPORTANT: Clear history! If we don't clear this, the next time they touch,
            # the spring will "remember" the old stretch, which is physically wrong.
            if p2.id in p1.tangential_overlaps:
                del p1.tangential_overlaps[p2.id]
            return np.zeros(3), np.zeros(3), np.zeros(3)

        # --- 3. Relative Velocity ---
        interpenetration_vel = -np.dot((p2.velocity - p1.velocity), n)

        # Equivalent properties
        R_eq = (p1.radius * p2.radius) / (p1.radius + p2.radius)
        m_eq = (p1.mass * p2.mass) / (p1.mass + p2.mass)
        E_eq = 1 / ((1 - p1.nu**2)/p1.E + (1 - p2.nu**2)/p2.E)
        
        # Shear modulus calculation (G = E / (2*(1+nu)))
        G1 = p1.E / (2 * (1 + p1.nu))
        G2 = p2.E / (2 * (1 + p2.nu))
        G_eq = 1 / ((2 - p1.nu)/G1 + (2 - p2.nu)/G2)

        # --- 4. Normal Force (Hertzian) ---
        xi = self.GammaForHertzThornton()
        
        # Stiffness and Damping for Normal direction
        k_n = 2 * E_eq * np.sqrt(R_eq * interpenetration)
        c_n = 2 * xi * np.sqrt(m_eq * k_n)
        
        # Calculate Normal Force magnitude (Elastic + Damping)
        F_n_mag = (2/3) * k_n * interpenetration + c_n * interpenetration_vel
        
        # Normal force usually shouldn't be attractive (adhesive) in basic HM models
        if F_n_mag < 0: 
            F_n_mag = 0.0
            
        F_n = -F_n_mag * n  # Force on p1 is opposite to n

        # --- 5. Tangential Force (Mindlin with Friction) ---
        
        # Calculate contact points (lever arms) relative to particle centers
        k_n_1 = k_n * p1.E / (p1.E + p2.E) # Simplified stiffness split
        k_n_2 = k_n * p2.E / (p1.E + p2.E)
        
        # Lever arms (vectors from center to contact point)
        x_ip = (p1.radius - (k_n_2 / (k_n_1 + k_n_2)) * interpenetration) * n
        x_jp = - (p2.radius - (k_n_1 / (k_n_1 + k_n_2)) * interpenetration) * n

        # Velocity at the exact contact point (v + omega x r)
        # Note: Assuming 2D, omega is [0, 0, w], position is [x, y, 0]
        v_ip = p1.velocity + np.cross(p1.omega, x_ip)
        v_jp = p2.velocity + np.cross(p2.omega, x_jp)
        
        v_rel = v_ip - v_jp
        
        # Tangential relative velocity
        v_rel_n = np.dot(v_rel, n) * n
        v_rel_t = v_rel - v_rel_n
        v_rel_t[2] = 0.0 # Ensure 2D plane
        
        # Stiffness for Tangential direction
        # Mindlin: k_t depends on normal force/overlap
        k_t = 8 * G_eq * np.sqrt(R_eq * interpenetration)
        c_t = 2* xi * np.sqrt(m_eq * k_t)
        
        # Retrieve previous tangential spring displacement (overlap)
        delta_t_vec = p1.tangential_overlaps.get(p2.id, np.zeros(3))
        
        # Integrate tangential velocity to get displacement increment
        delta_s_inc = v_rel_t * delta_t
        
        # Update the spring elongation
        delta_t_vec += delta_s_inc
        
        # Compute "Trial" Tangential Force (Elastic + Damping)
        F_t_trial = -k_t * delta_t_vec - c_t * v_rel_t
        
        # --- 6. Coulomb Friction Law (Task 1) ---
        F_t_mag_trial = np.linalg.norm(F_t_trial)
        F_n_mag_abs = np.abs(F_n_mag)
        F_t_limit = self.mu_friction * F_n_mag_abs
        
        F_t = np.zeros(3)
        
        if F_t_mag_trial > F_t_limit and F_t_mag_trial > 1e-12:
            # SLIDING condition: Cap the force to the friction limit
            ratio = F_t_limit / F_t_mag_trial
            F_t = F_t_trial * ratio
            
            # IMPORTANT: Reset the spring! 
            # If sliding occurs, the spring cannot stretch further than the friction allows.
            # We adjust delta_t_vec so that k_t * delta_t_vec equals the limit force.
            # (Ignoring damping for the reset logic is standard approximation)
            delta_t_vec = delta_t_vec * ratio
            
        else:
            # STICK condition: Use the calculated elastic force
            F_t = F_t_trial

        # Store the updated (and potentially clipped) history back to p1
        p1.tangential_overlaps[p2.id] = delta_t_vec

        # --- 7. Total Force and Torque ---
        F_total = F_n + F_t
        
        # Torque = r x F (Only tangential force creates torque on spheres as Normal passes through center)
        # Torque on p1
        torque1 = np.cross(x_ip, F_t)
        
        # Torque on p2 (Action-Reaction: Force is -F_t at contact point x_jp)
        # Note: The force applied TO p2 is -F_total. 
        # The contact point relative to p2 is x_jp.
        # So torque = x_jp x (-F_t)
        torque2 = np.cross(x_jp, -F_t)

        return F_total, torque1, torque2
    
    def GammaForHertzThornton(self):
        """
        Calculates dissipation factor based on coefficient of restitution.
        Source: Thornton et al. (see doc/references)
        """
        e = self.coeff_of_restitution
        
        if e < 0.001: e = 0.001
        if e > 0.999: return 0.0

        h1 = -6.918798
        h2 = -16.41105
        h3 = 146.8049
        h4 = -796.4559
        h5 = 2928.711
        h6 = -7206.864
        h7 = 11494.29
        h8 = -11342.18
        h9 = 6276.757
        h10 = -1489.915

        alpha = e * (h1 + e * (h2 + e * (h3 + e * (h4 + e * (h5 + e * (h6 + e * (h7 + e *(h8 + e * (h9 + e * h10)))))))))

        return np.sqrt(1.0/(1.0 - (1.0+e)*(1.0+e) * np.exp(alpha)) - 1.0)
    
    def compute_boundary_contact(self, p, boundary, delta_t):
        """
        Compute contact force and torque between a particle and a rigid boundary.
        """
        vec_to_boundary = p.position - boundary.point
        distance = np.dot(vec_to_boundary, boundary.normal)
        interpenetration = p.radius - distance
        if interpenetration <= 0:
            # No contact
            if boundary.boundary_id in p.tangential_overlaps:
                del p.tangential_overlaps[boundary.boundary_id]
            return np.zeros(3), np.zeros(3)
        n = boundary.normal

        interpenetration_vel = -np.dot(p.velocity, n)

        # Equivalent properties
        # Equivalent radius and mass for particle-boundary contact
        R_eq = p.radius
        m_eq = p.mass

        # Equivalent Young's modulus
        E_eq = 1 / ((1 - p.nu**2)/p.E + (1 - boundary.nu**2)/boundary.E)

        # Equivalent Shear modulus
        G1 = p.E / (2 * (1 + p.nu))
        G2 = boundary.E / (2 * (1 + boundary.nu))
        G_eq = 1 / ((2 - p.nu)/G1 + (2 - boundary.nu)/G2)

        # Normal Force (Hertzian)
        xi = self.GammaForHertzThornton()

        k_n = 2 * E_eq * np.sqrt(R_eq * interpenetration)
        c_n = 2 * xi * np.sqrt(m_eq * k_n)

        # contact force in normal direction
        F_n_mag = (2/3) * k_n * interpenetration + c_n * interpenetration_vel

        if F_n_mag < 0: 
            F_n_mag = 0.0
        F_n = F_n_mag * n   #? Force on particle from boundary (along n)

        
        x_ip = (p.radius - interpenetration) * n    #? Lever arm from particle center to contact point
        v_ip = p.velocity + np.cross(p.omega, x_ip) #? Velocity at contact point

        v_rel = v_ip - boundary.velocity    # Relative velocity at contact point
        v_rel_n = np.dot(v_rel, n) * n      # Velocity component normal to boundary
        v_rel_t = v_rel - v_rel_n           # Tangential relative velocity

        v_rel_t[2] = 0.0 # Ensure 2D plane

        k_t = 8 * G_eq * np.sqrt(R_eq * interpenetration)   # Stiffness for Tangential direction
        c_t = 2* xi * np.sqrt(m_eq * k_t)                   # Tangential damping

        if not hasattr(p, 'tangential_overlaps'):
            p.tangential_overlaps = {}
        delta_t_vec = p.tangential_overlaps.get(boundary.boundary_id, np.zeros(3))

        delta_t_vec += v_rel_t * delta_t
        F_t_trial = -k_t * delta_t_vec - c_t * v_rel_t      # Trial Tangential Force

        mu_effective = min(self.mu_friction, getattr(boundary, 'mu', 0.5))  # Friction coefficient (minimum)

        F_t_mag_trial = np.linalg.norm(F_t_trial)   # Magnitude of trial tangential force

        F_n_mag_abs = np.abs(F_n_mag)           # Magnitude of normal force
        F_t_limit = mu_effective * F_n_mag_abs  # Friction limit

        F_t = np.zeros(3)
        if F_t_mag_trial > F_t_limit and F_t_mag_trial > 1e-12:
            # SLIDING condition
            ratio = F_t_limit / F_t_mag_trial
            F_t = F_t_trial * ratio
            delta_t_vec = delta_t_vec * ratio
        else:
            # STICK condition
            F_t = F_t_trial
            
        # Store updated tangential overlap
        p.tangential_overlaps[boundary.boundary_id] = delta_t_vec
        F_total = F_n + F_t
        torque = np.cross(x_ip, F_t)
        return F_total, torque  

