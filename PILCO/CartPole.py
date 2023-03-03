import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr

import objax
import optax as ox
import jaxkern as jk
import gpjax as gpx

from jax import jit, grad, vmap
from jaxutils import Dataset

import matplotlib.pyplot as plt

### SETTINGS

# General

key = jr.PRNGKey(123)
simTimesteps = 25 # number of deciseconds cartpole is ran

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
    """
    multiple gaussian process regression
    creates a gp for every dimension in input for dynamics learning
    """
    def __init__(self, D, name=None):
        super(MGPR, self).__init__()
        self.num_inputs = D.X.shape[1]
        self.num_ouputs = D.y.shape[1]
        self.n_data_points = D.X.shape[0]

        self.models = []

        self.create_models(D)

    def create_models(self, D):
        for i in range(self.num_ouputs):
            kernel = jk.RBF(active_dims=[0, 1, 2, 3])
            prior = gpx.Prior(kernel=kernel)

            likelihood = gpx.Gaussian(num_datapoints=D.n)
            posterior = prior * likelihood

            parameter_state = gpx.initialise(
                posterior, key, kernel
            )

            self.models.append(parameter_state)

    def optimize(self, D):
        # optimize gp models
        new_models = []

        for model in self.models:
            kernel = jk.RBF(active_dims=[0, 1, 2, 3])
            prior = gpx.Prior(kernel=kernel)

            likelihood = gpx.Gaussian(num_datapoints=D.n)
            posterior = prior * likelihood

            negative_mll = jit(posterior.marginal_log_likelihood(D, negative=True))
            optimizer = ox.adam(learning_rate=0.01)

            inference_state = gpx.fit(
                objective=negative_mll,
                parameter_state=model,
                optax_optim=optimizer,
                num_iters=500
            )

            new_models.append(inference_state)

        self.models = new_models

    def calculate_factorizations(self):
        pass
    

class RBFN(MGPR):
    # RBF controller but its actually a degenerate GP
    
    def __init__(self, state_dim, control_dim, num_basis, name="RBFN Policy"):
        self.vars = Dataset(X=objax.random.normal((num_basis, state_dim)),y=0.1*objax.random.normal(num_basis, control_dim))
        MGPR.__init__(self, self.vars)

    def create_models(self, data):
        self.models = []
        for i in range(self.num_ouputs):
            kernel = jax.RBF(active_dims=[0, 1, 2, 3])
            prior = gpx.Prior(kernel=kernel)

            likelihood = gpx.Gaussian(num_datapoints=D.n)
            posterior = prior * likelihood

            parameter_state = gpx.initialise(
                posterior, key, kernel
            )

            self.models.append(parameter_state)

    def compute_action(self, m, s, squash=True):
        iK, beta = self.calculate_factorizations()



class PILCO():
    # Probabilistic Inference for Learning Control algorithm

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

        self.controller = policy
        
        self.cost = cost

    def optimize(self):
        self.dynamics_model.optimize()

    def optimize_policy(self, maxiter=500, restarts=1):
        
        lr = 0.01
        if not self.optimizer:
            opt_hyperparams = objax.optimizer.Adam(self.controller.vars)
            energy = objax.GradValues(self.training_loss, self.controller.vars)

    def training_loss(self):
        predictions = self.predict(self.m_init, self.s_init, self.horizon)
        _, _, reward = predictions
        return -reward
    
    def predict(self, m_x, s_x, n):
        init_val = (m_x, s_x, n)

        def body_fun(i, v):
            m_x, s_x, reward = v
            return (
                *self.propagate(m_x, s_x),
                
            )
        
    def propagate(self, m_x, s_x):
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

### UTILITY

def squash_sin(m, s, max_action=10):
    k = jnp.shape(m)[1]
    max_action = max_action * jnp.ones((1, k))

    M = max_action * jnp.exp(-0.5 * jnp.diag(s)) * jnp.sin(m)

    lq = -0.5 * (jnp.diag(s)[:, None] + jnp.diag(s)[None, :])
    q = jnp.exp(lq)
    mT = jnp.transpose(m, (1, 0))
    S = (jnp.exp(lq + s) - q) * jnp.cos(mT - m) - (jnp.exp(lq - s) - q) * jnp.cos(
        mT + m
    )
    S = 0.5 * max_action * jnp.transpose(max_action, (1, 0)) * S

    C = max_action * objax.Vectorize(lambda x: jnp.diag(x, k=0), objax.VarCollection())(
        jnp.exp(-0.5 * jnp.diag(s)) * jnp.cos(m)
    )

    return M, S, C.reshape((k, k))

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

Policy = RBFN(state_dim=state_dim, control_dim=control_dim, n=numBasisFunctions)

pilco = PILCO(D=D, controller=Policy, horizon=T) # init pilco with data D, RBF controller with 50 basis functions, with T = 40

for rollout in range(num_rollouts):

    import pdb
    pdb.set_trace()

    pass