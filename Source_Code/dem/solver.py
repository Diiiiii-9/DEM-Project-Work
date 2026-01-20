class DEMSolver:
    def __init__(self, particles, contact_model, integrator, gravity, boundaries=None):
        self.particles = particles
        self.contact_model = contact_model
        self.integrator = integrator
        self.gravity = gravity
        self.boundaries = boundaries if boundaries is not None else []

    def solve_time_step(self, delta_t):
        # --- Step 1: Pre-force integration (update position, angle) ---
        self.integrator.pre_force_update(self.particles, delta_t)

        # --- Step 2: Reset forces and torques ---
        for p in self.particles:
            p.force[:] = 0.0
            p.torque[:] = 0.0

        # --- Step 3: Contact forces between all particle pairs ---
        for i, p1 in enumerate(self.particles):
            for j, p2 in enumerate(self.particles):
                if i < j:
                    force, torque1, torque2 = self.contact_model.compute_contact(p1, p2, delta_t)
                    p1.force += force
                    p1.torque += torque1
                    p2.force -= force
                    p2.torque += torque2
        for p in self.particles:
            for boundary in self.boundaries:
                # Compute contact force and torque with boundary # NEW FUNCTION!
                force, torque = self.contact_model.compute_boundary_contact(p, boundary, delta_t)
                # Accumulate forces and torques
                p.force += force
                p.torque += torque

        # --- Step 4: Add external forces (e.g. gravity) ---
        for p in self.particles:
            # add body forces
            p.force += p.mass * self.gravity
            
            #Update acceleration
            p.acceleration = p.force / p.mass
            
            # Update angular velocity
            p.angular_acc = p.torque / p.inertia

        # --- Step 5: Post-force integration (update velocity, omega) ---
        self.integrator.post_force_update(self.particles, delta_t)
