import jax

import jax.numpy as jnp
import gpjax

### SETTINGS

# PILCO

timeHorizon = 25 # time horizon
Delta_t = 1 # time discretization

# CART POLE

pendulumLength = 0.6
massCart = 0.5
massPendulum = 0.5 # mass of pendulum

# Physics

a_g = 9.82 # accleration due to gravity
mu = 0.1 # friction coeff


### CART POLE IMPLEMENTATION

def CartPole():
    def __init__(self):
        self.l = pendulumLength
        self.m_c = massCart
        self.m_p = massPendulum

    def step(self, state, control, dt):
        """
        calculates next state via euler difference
        args: state at time t-1, control signal applied, timestep
        returns: state time t 
        """

        r, r_d, theta, theta_d = state

        s = jnp.sin(theta)
        c = jnp.cos(theta)

        r_dd = (((-2 * self.m_p * self.l * (theta_d ** 2) * s) + (3 * self.m_p * a_g * s * c) + (4 * control) - (4 * mu*r_d))/( (4*(self.m_c + self.m_p)) - (3 * self.m_p * self.l * (c ** 2))))
        theta_dd = (((-3 * self.m_p * self.l * (theta_d ** 2) * s * c) + ((6 * (self.m_c + self.m_p) ) * a_g * s) + (( 6 * (control - mu*self.r_d) ) * c))/( (4*self.l*(self.m_c + self.m_p)) - (3*self.m_p*self.l * (c ** 2))))

        r = r + r_d*dt
        r_d = r_d + r_dd*dt

        theta = theta_d * dt
        theta_d = theta_dd * dt

        return jnp.array([r, r_d, theta, theta_d])

    def calculate_cost(self, state, variance=0.25):
        """
        saturating cost function, sqaured euclidean distance from end point and goal.
        args: state, variance
        returns: numerical cost
        """
        r, r_d, theta, theta_d = state
        p_x = self.l*jnp.sin(theta)
        p_y = self.l*jnp.cos(theta)
        tip = jnp.array([r + p_x, p_y])

        target = jnp.array([0.0, self.l])
        dist = jnp.sum((tip - target)**2)
        cost = 1 - jnp.exp(-0.5 * dist * 1/(variance**2))
        return cost