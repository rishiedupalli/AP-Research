# Jax Implementation of PILCO for solving the Cart-Pole Swing-Up Problem

# IMPORTS
import jax

import objax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.random as jr
import distrax as dx

import numpy as np

import optax as ox
import jaxkern as jk
import gpjax as gpx

from jax import jit, grad, vmap
from jaxutils import Dataset

import matplotlib.pyplot as plt
import random

### SETTINGS

# General

simTimesteps = 2.5 # x number of seconds cartpole is ran
key = jr.PRNGKey(12345)

# PILCO

num_random_rollouts = 5 # number of random rollouts
num_rollouts = 15 # Number of PILCO rollouts
T = 40 # PILCO run time, how long to stay under policy \pi
numBasisFunctions = 50 # number of RBF basis functions

# CART POLE

pendulumLength = 0.6
massCart = 0.5
massPendulum = 0.5 # mass of pendulum

# Physics

a_g = 9.82 # accleration due to gravity
mu = 0.1 # friction coeff

### CART POLE IMPLEMENTATION

class CartPole():
    def __init__(self):
        self.l = pendulumLength
        self.m_c = massCart
        self.m_p = massPendulum
        self.dt = 0.1

    def step(self, state, control):
        """
        calculates next state via euler difference
        args: state at arbitrary time t, control signal applied
        returns: state at time t+1
        """

        r, r_d, theta, theta_d = state

        s = jnp.sin(theta)
        c = jnp.cos(theta)

        r_dd = (((-2 * self.m_p * self.l * (theta_d ** 2) * s) + (3 * self.m_p * a_g * s * c) + (4 * control) - (4 * mu*r_d))/( (4*(self.m_c + self.m_p)) - (3 * self.m_p * self.l * (c ** 2))))
        theta_dd = (((-3 * self.m_p * self.l * (theta_d ** 2) * s * c) + ((6 * (self.m_c + self.m_p) ) * a_g * s) + (( 6 * (control - mu*r_d) ) * c))/( (4*self.l*(self.m_c + self.m_p)) - (3*self.m_p*self.l * (c ** 2))))

        r = r + r_d * self.dt
        r_d = r_d + r_dd * self.dt

        theta = theta_d * self.dt
        theta_d = theta_dd * self.dt

        return jnp.array([r, r_d, theta, theta_d])
    
### RBF Controller

class RBFController():
    def __init__(self, state_dim, num_basis_functions):
        self.theta_init = {
            'weights': np.random.normal(size=(1, num_basis_functions)),
            'lambda': np.random.normal(size=(1, state_dim)),
            'centers': np.random.normal(size=(1, num_basis_functions))
        }

    def Random_Policy(self, x):
        return random.randrange(-10,10)

    def Network(x: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        u = 0
        return u

### PILCO

Env = CartPole()

Controller = RBFController(state_dim=4, num_basis_functions=50)

# Step 1: Apply random control signals u and record initial data

def random_rollout():
    X = []
    Y = []

    x_t1 = jnp.array([0., 0., 0., 0.]) # reset environment

    for dt in range(int(simTimesteps*10)):
        u = Controller.Random_Policy(x_t1) # apply a random force of x \in [-10,10] newtons based on RBF controller intialized with \theta \sim \mathcal{N}(0,I)
        x_t2 = Env.step(x_t1, u)

        X.append(jnp.hstack((x_t1,u)))
        Y.append(x_t2 - x_t1)

        x_t1 = x_t2

    return jnp.stack(X), jnp.stack(Y)

X1, Y1 = random_rollout()

for rollout in range(1, num_random_rollouts):
    X2, Y2 = random_rollout()
    X1 = jnp.vstack((X1,X2))
    Y1 = jnp.vstack((Y1,Y2))

X_Final = X1
Y_Final = Y1

# Step 2: PILCO Main Loop

RBF_Kernel = jk.RBF(active_dims=[0,1,2,3,4])

for rollout in range(num_rollouts):
    print("PILCO Iteration " + str(rollout + 1))
    
    ### Step 3: Learn GP Dynamics Models

    ## Settings
    num_outputs = 4 # For multivariate targets, we train conditionally independent GPs for each target dimension. For Cart-Pole Swing-Up, x = [*,*,*,*] -> d = 4

    optimizer_lr = 0.01 # optimizer learning rate
    optimizer_iterations = 1 # how many iterations hyperparameters are optimized

    ## Variables
    DynamicsModels = []
    DynamicsModelsData = []

    M_ts1 = X_Final[0] + Y_Final[0].mean()
    S_ts1 = np.cov(Y_Final[0])

    # Create Dynamics Model Data for each GP
    for i in range(num_outputs):
        DynamicsModelsData.append(Dataset(X=X_Final, y=Y_Final[:, i:i + 1]))

    # Create & Optimize GP Dynamics Models
    for Data in DynamicsModelsData:
        prior = gpx.Prior(kernel=RBF_Kernel)

        likelihood = gpx.Gaussian(num_datapoints=Data.n)
        posterior = prior * likelihood

        parameter_state = gpx.initialise(
            posterior, key
        )

        params, _, _ = parameter_state.unpack()
        print(params)

        negative_mll = jit(posterior.marginal_log_likelihood(Data, negative=True))
        negative_mll(params)

        optimizer = ox.adam(learning_rate=optimizer_lr)

        print("Model Hyperparameter Tuning in Progress")
        print(Data)
        
        inference_state = gpx.fit(
            objective=negative_mll,
            parameter_state=parameter_state,
            optax_optim=optimizer,
            num_iters=optimizer_iterations
        )

        DynamicsModels.append(inference_state)

        learned_params, _ = inference_state.unpack()
        print(learned_params)
        latent_dist = posterior(learned_params, Data)(Data.X)
        predictive_dist = likelihood(learned_params, latent_dist)


    ### Step 4: Policy Generation & Optimization

    ## Constants

    ## Helper Functions

    for T_Step in range(T):

        # Propagate through GP dynamics models to calculate J^{\pi}(\theta)
        
        # Calculate Mean of N(x_t+1 | \mu_{t-1}, \Sigma_{x_{t-1})
        
        DeltaMu = []

        for a in range(num_outputs): # run loop for each dynamics model a

            params, _ = DynamicsModels[a].unpack()
            GP_x = DynamicsModelsData[a].X
            GP_y = DynamicsModelsData[a].y

            K = RBF_Kernel.cross_covariance(params['kernel'], GP_x, GP_x)

            Beta = jnp.linalg.inv(K) @ GP_y

            q = []

            for x in X_Final:

                Lambda = jnp.diag(params['kernel']['lengthscale'])

                v_i = x - M_ts1      
            
                A =  (1 / jnp.sqrt(jnp.abs(jnp.linalg.det(M_ts1 @ jnp.linalg.inv(Lambda) + jnp.eye(num_outputs + 1)))))

                B = jnp.exp( -0.5 * (jnp.transpose(v_i) @ jnp.linalg.inv((S_ts1 + Lambda)) @ v_i))

                q.append(A * B)

            q = jnp.asarray(q)

            DeltaMu.append(jnp.transpose(Beta) @ q)

        New_Mean = jnp.asarray(DeltaMu)

        # Calculate Covariance of N(x_t+1 | \mu_{t-1}, \Sigma_{x_{t-1})

        DeltaSigma = [num_outputs][num_outputs]

        for a in range(num_outputs):
            for b in range(num_outputs):

                GP_Params_A, _ = DynamicsModels[a].unpack()
                GP_Params_B, _ = DynamicsModels[b].unpack()

                GP_Ax = DynamicsModelsData[a].X
                GP_Ay = DynamicsModelsData[a].y

                GP_Bx = DynamicsModelsData[b].X
                GP_By = DynamicsModelsData[b].y

                K_A = RBF_Kernel.cross_covariance(GP_Params_A['kernel'], GP_Ax, GP_Ax)
                K_B = RBF_Kernel.cross_covariance(GP_Params_B['kernel'], GP_Bx, GP_Bx)

                Beta_A = jnp.linalg.inv(K_A) @ GP_Ay

                Beta_B = jnp.linalg.inv(K_B) @ GP_Bx

                Q = [X_Final][X_Final]

                Lambda_A = jnp.diag(GP_Params_A['kernel']['lengthscale'])

                Lambda_B = jnp.diag(GP_Params_B['kernel']['lengthscale'])

                for i in X_Final: # Calculate Entries of Q_ij
                    for j in X_Final:

                        k_a = RBF_Kernel.__call__(GP_Params_A['kernel'], i, M_ts1)

                        k_b = RBF_Kernel.__call__(GP_Params_B['kernel'], j, M_ts1)

                        R = S_ts1 * (jnp.linalg.inv(Lambda_A) + jnp.linalg.inv(Lambda_B)) + jnp.eye(num_outputs + 1)

                        z = (jnp.linalg.inv(Lambda_A) @ (i - M_ts1)) + (jnp.linalg.inv(Lambda_B) @ (j - M_ts1))

                        A = (k_a * k_b) / jnp.sqrt(jnp.abs(jnp.linalg.det(R)))

                        B = jnp.exp(0.5 * (jnp.transpose(z) @ jnp.linalg.inv(R) @ (S_ts1 * z)))

                        Q[i][j] = A*B

                    E_fts1 = 1 - jnp.trace(Beta_A @ Q)

                    E_fxts1 = (jnp.transpose(Beta_A) @ Q @ Beta_B)

                if (a == b):
                    s_aa = E_fts1 + E_fxts1 - (DeltaMu[a] * DeltaMu[a])
                    DeltaSigma[a][b] = s_aa
                else:
                    s_ab = E_fxts1 - DeltaMu[a] * DeltaMu[b]
                    DeltaSigma[a][b] = s_ab

        # Calculate N(x_t+1 | \mu_{t-1}, \Sigma_{x_{t-1})

    # Evaluate J^{\pi}