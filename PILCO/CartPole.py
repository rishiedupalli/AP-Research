import jax
import jax.numpy as jnp
import jax.random as jr
from jax import jit
from jaxutils import Dataset
import objax
import optax as ox
import jaxkern as jk
import gpjax as gpx
import matplotlib.pyplot as plt

### SETTINGS

# General

key = jr.PRNGKey(123)
simTimesteps = 25 # number of seconds cartpole is ran

# PILCO

T = 40 # PILCO run time, how long to stay under policy \pi
num_random_rollouts = 5 # number of random rollouts
num_rollouts = 15 # Number of PILCO rollouts
numBasisFunctions = 50 # number of RBF basis functions

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
        self.dt = 0.1

    def step(self, state, control):
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

        r = r + r_d * self.dt
        r_d = r_d + r_dd * self.dt

        theta = theta_d * self.dt
        theta_d = theta_dd * self.dt

        return jnp.array([r, r_d, theta, theta_d])

    def calculate_cost(self, state, variance=0.25):
        """
        cost function, sqaured euclidean distance from end point and goal.
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


### PILCO

class MGPR():
    def __init__(self, D):
        super(MGPR, self).__init__()
        self.num_inputs = D.X.shape[1]
        self.num_ouputs = D.y.shape[1]
        self.n_data_points = D.X.shape[0]

        self.modelparams = []
        self.posteriors = []

        self.create_models(D)

    def create_models(self, D):
        for i in range(self.num_ouputs):
            pass

    def optimize(self, D):
        pass

class PILCO():
    def __init__(self, D, policy, cost, horizon, m_init=None, s_init=None, name="PILCO"):
        super(PILCO, self).__init__(name)

        self.state_dim = D.y.shape[1]
        self.control_dim = D.X.shape[1] - D.y.shape[1]
        self.horizon = horizon

        if m_init is None or s_init is None:
            self.m_init = D.X[0:1, 0:self.state_dim]
            self.s_init = jnp.diag(jnp.ones(self.state_dim) * 0.1)
        else:
            self.m_init = m_init
            self.s_init = s_init

        self.dynamics_model = MGPR(D)

        self.policy = policy
        
        self.cost = cost

class RBFN(MGPR):
    def __init__(self, state_dim, control_dim, n, name="RBFN Policy"):
        MGPR.__init__(self)
    

### UTILITY

def random_rollout():
    X = []
    Y = []

    x_t1 = jnp.array([0, 0, 0, 0]) # reset environment

    for dt in range(simTimesteps):
        u = jr.uniform(key, 1, int, -10,10) # apply a random force of x \in [-10.10] newtons
        x_t2 = env.step(x_t1, u)

        X.append(jnp.hstack((x_t1,u)))
        Y.append(x_t2 - x_t1)

        x_t1 = x_t2

    return jnp.stack(X), jnp.stack(Y)

### DRIVER

env = CartPole()

# Generate Random Rollouts

X1, Y1 = random_rollout()

for rollout in range(1, num_random_rollouts):
    X2, Y2 = random_rollout()
    X1 = jnp.vstack((X1,X2))
    Y1 = jnp.vstack((Y1,Y2))

D = Dataset(X=X1,y=X2)

state_dim = D.y.shape[1]
control_dim = D.X.shape[1] - state_dim

learnedPolicy = RBFN(state_dim=state_dim, control_dim=control_dim, n=numBasisFunctions)

pilco = PILCO(D=D, controller=learnedPolicy, horizon=T) # init pilco with data D, RBF controller with 50 basis functions, with T = 40

for rollout in range(num_rollouts):
    import pdb
    pdb.set_trace()
    pass