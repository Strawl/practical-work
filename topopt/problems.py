from feax import Problem
import jax.numpy as jnp

class DensityElasticityProblem(Problem):
    def custom_init(self,
                    E0: float,
                    E_eps: float,
                    nu: float,
                    p: float,
                    T: float) -> None:
        """
        E0 : float
            Young's modulus of solid material.
        E_eps : float
            Small stiffness for void regions (E_min).
        nu : float
            Poisson's ratio.
        p : float
            SIMP penalization exponent.
        T : float
            Traction magnitude (load intensity).
        """

        # Initialize subclass-specific attributes
        self.E0 = E0
        self.E_eps = E_eps
        self.nu = nu
        self.p = p
        self.T = T

    def get_tensor_map(self):
        """
        Returns a stress function σ(u_grad, ρ) implementing SIMP interpolation.
        """
        def stress(u_grad, rho):
            # Material interpolation: E(ρ) = (E0 - E_eps) * ρ^p + E_eps
            E = (self.E0 - self.E_eps) * rho**self.p + self.E_eps
            mu = E / (2.0 * (1.0 + self.nu))
            lam = E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

            strain = 0.5 * (u_grad + u_grad.T)
            sigma = lam * jnp.trace(strain) * jnp.eye(self.dim) + 2.0 * mu * strain
            return sigma

        return stress


    def get_surface_maps(self):
        """
        Returns a list of surface traction mapping functions.
        Each map computes a traction vector applied at a boundary point x.

        Automatically adapts to 2D or 3D based on self.dim.
        """
        def surface_map(u, x, load: float) -> jnp.ndarray:
            # Initialize zero traction vector
            traction = jnp.zeros(self.dim)

            # Apply traction in the last axis direction by convention:
            # - For 2D → y-direction
            # - For 3D → z-direction
            traction = traction.at[-1].set(load)

            return traction

        return [surface_map]

class PlaneStressElasticityProblem(Problem):
    def custom_init(self,
                 Emax: float,
                 Emin: float,
                 nu: float,
                 penal: float,
                 T: float):
        """
        Parameters
        ----------
        Emax : float
            Young's modulus of solid material.
        Emin : float
            Young's modulus of void material.
        nu : float
            Poisson's ratio.
        penal : float
            SIMP penalization exponent.
        T : float
            Traction magnitude.
        """
        self.Emax = Emax
        self.Emin = Emin
        self.nu = nu
        self.penal = penal
        self.T = T

    def get_tensor_map(self):
        """
        Returns the stress tensor σ(u_grad, θ) for plane stress elasticity.
        """
        def stress(u_grad: jnp.ndarray, rho: float) -> jnp.ndarray:
            # SIMP interpolation for Young's modulus
            E = self.Emin + (self.Emax - self.Emin) * rho ** self.penal

            # Small strain tensor
            epsilon = 0.5 * (u_grad + u_grad.T)
            eps11, eps22, eps12 = epsilon[0, 0], epsilon[1, 1], epsilon[0, 1]

            # Plane stress constitutive relation
            sig11 = E / (1 + self.nu) / (1 - self.nu) * (eps11 + self.nu * eps22)
            sig22 = E / (1 + self.nu) / (1 - self.nu) * (self.nu * eps11 + eps22)
            sig12 = E / (1 + self.nu) * eps12

            return jnp.array([[sig11, sig12],
                             [sig12, sig22]])

        return stress

    def get_surface_maps(self):
        """
        Returns a surface traction mapping function for 2D plane stress.
        """
        def surface_map(u, x, load: float) -> jnp.ndarray:
            # Traction in y-direction
            return jnp.array([0.0, load])
        return [surface_map]