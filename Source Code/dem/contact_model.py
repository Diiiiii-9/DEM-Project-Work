# dem/contact_model.py

from abc import ABC, abstractmethod
import numpy as np

def get_contact_model(name, params):
    if name.lower() in ["hm+d", "hertz", "hertzmindlin", "hertz-mindlin"]:
        return HertzMindlinDashpot(params)
    else:
        raise ValueError(f"Unknown contact model: {name}")

class ContactModel(ABC):
    def __init__(self, params):
        self.params = params  # dictionary for flexibility

    @abstractmethod
    def compute_contact(self, p1, p2, delta_t):
        """
        Calculate contact force and torque on p1 due to p2.
        Returns:
            force_on_p1 (np.array)
            torque_on_p1 (np.array)
        """
        pass

    

class HertzMindlinDashpot(ContactModel):
    def __init__(self, params):
        super().__init__(params)
        self.coeff_of_restitution = params["coeff_of_restitution"]
        self.mu_friction = params["mu"]

    def compute_contact(self, p1, p2, delta_t):

        r_vec = p2.position - p1.position
        l_c = np.linalg.norm(r_vec)

        # compute normal vector
        n = r_vec / l_c
        n[2] = 0.0

        # compute equivalent quantitites
        R_eq=(p1.radius*p2.radius)/(p1.radius+p2.radius)
        m_eq=(p1.mass*p2.mass)/(p1.mass+p2.mass)
        E_eq=1/((1-p1.nu**2)/p1.E+ (1-p2.nu**2)/p2.E)
        G_eq = (1)/((2-p1.nu)/(p1.E/(2*(1+p1.nu))) + (2-p2.nu)/(p2.E/(2*(1+p2.nu))))

        # compute normal interpenetration
        interpenetration = p1.radius + p2.radius - np.dot((p2.position - p1.position), n)

        interpenetration_vel = -np.dot((p2.velocity - p1.velocity), n)

        if interpenetration <= 0:
            return np.zeros(3), np.zeros(3), np.zeros(3)  # no contact
        
        # compute fraction xi
        xi = self.GammaForHertzThornton()

        # compute rheolocical parameter
        # for normal force
        k_n = 2 * E_eq * np.sqrt(R_eq  * interpenetration)
        c_n = 2 * xi * np.sqrt(m_eq * k_n)

        # compute rheolocical parameter
        # for tangential force
        # ToDo
        
        k_n_1 = k_n*p1.E/(p1.E+p2.E)
        k_n_2 = k_n*p2.E/(p1.E+p2.E)

        # calculate distance from sphere center to contact point
        x_ip = (p1.radius - (k_n_2/ (k_n_1 + k_n_2)) * interpenetration) * n
        x_jp = - (p2.radius - (k_n_1 / (k_n_1 + k_n_2)) * interpenetration) * n

        # velocity at contact point
        v_ip = p1.velocity  + np.cross(p1.omega, x_ip)
        v_jp = p2.velocity  + np.cross(p2.omega, x_jp)

        # relative velocity at contact point
        v_rel = v_ip - v_jp

        # relative velocity normal and tangential direction
        v_rel_n = np.array(np.dot(v_rel, n) * n)
        v_rel_t = v_rel - v_rel_n
        v_rel_t[2]=0.0
        
        # zero out small components relative to total speed
        if np.linalg.norm(v_rel_t) < 1e-15:
            v_rel_t[:] = 0.0
        

        # compute tangential unit vector t
        if np.linalg.norm(v_rel_t) != 0:
            t = v_rel_t / np.linalg.norm(v_rel_t)
        else:
            t = np.array([0, 0, 0])

        # incremental displacement tangential direction
        delta_s = v_rel_t * delta_t

        # calculate Hertz normal contact force
        F_n = 2/3 * k_n * interpenetration + c_n * interpenetration_vel
        F_n = -F_n * n


        # Compute Tangential force
        # ToDo
        F_t = np.zeros(3)


        # Total force
        F_total = F_n + F_t

        # Compute Torque from tangential force (T=x_p x F_t)
        # ToDo
        torque1 = np.zeros(3)
        torque2 = np.zeros(3)

        return F_total, torque1, torque2
    
    def GammaForHertzThornton(self):
        e = self.coeff_of_restitution
        # see calculation: https://www.sciencedirect.com/science/article/pii/S0032591012005670#s0030
        if e < 0.001:
            e = 0.001

        if e > 0.999:
            return 0.0

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
