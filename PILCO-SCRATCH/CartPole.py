# Jax Implementation of PILCO for solving the Cart-Pole Swing-Up Problem

# IMPORTS
import jax

import objax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy
import jax.random as jr

import numpy as np

import optax as ox
import jaxkern as jk
import gpjax as gpx

from jax import jit, grad, vmap
from jaxutils import Dataset

import matplotlib.pyplot as plt
import scipy.integrate as spint

### SETTINGS

# General

jax.config.update("jax_enable_x64", True)

simTimesteps = 2.5 # x number of seconds cartpole is ran
key = jr.PRNGKey(12345)

# PILCO

num_random_rollouts = 1 # number of random rollouts
num_rollouts = 15 # Number of PILCO rollouts
T = 25 # PILCO run time, how long to stay under policy \pi
numBasisFunctions = 50 # number of RBF basis functions
input_dim = 4

# CART POLE

pendulumLength = 0.6
massCart = 0.5
massPendulum = 0.5 # mass of pendulum
Target = jnp.array([0, 0, jnp.pi, 0])

# Physics

a_g = 9.82 # accleration due to gravity
mu = 0.1 # friction coeff

# Testing

CalcCov = False

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
    def __init__(self, input_dim, num_basis_functions):
        self.num_basis_functions = num_basis_functions
        self.input_dim = input_dim
        self.centers = jr.normal(key, (self.num_basis_functions, self.input_dim))
        self.weights = jr.normal(key, (self.num_basis_functions,))
        self.Lambda = jnp.diag(jr.normal(key, (1, self.input_dim)).flatten())

    def rbf_kernel(self, x, mu_i):
        dist = (x - mu_i).T @ self.Lambda @ (x - mu_i)
        return jnp.exp(-.05 * dist)
    
    def compute_action(self, x):
        sum = []
        for mu_i, w in zip(self.centers, self.weights):
            phi = self.rbf_kernel(x, mu_i)
            dot = jnp.dot(w, phi)
            print(phi)
            print(dot)
            sum.append(dot)

        return jnp.sum(jnp.asarray(sum))
    
def Cost(x, t1, t2, m, s):
    t = Target
    return spint.quad_vec(1 - (jnp.exp(-((  jnp.linalg.norm(x - t)**2 )/0.25))) * ((jnp.exp(-0.5 * (x - m).T @ s.T @ (x-m))) / (jnp.sqrt( 2 * jnp.pi )**4 * jnp.linalg.det(s))), t1, t2, args=(m, s) )

### PILCO

Env = CartPole()

Controller = RBFController(input_dim=input_dim, num_basis_functions=numBasisFunctions)

# Step 1: Apply random control signals u and record initial data

def rollout():
    X = []
    Y = []

    x_t1 = jnp.array([0., 0., 0., 0.]) # reset environment

    for dt in range(int(simTimesteps*10)):
        u = Controller.compute_action(x_t1) # apply  x \in [-10,10] newtons based on RBF controller intialized with \theta \sim \mathcal{N}(0,I)
        x_t2 = Env.step(x_t1, u)

        X.append(jnp.hstack((x_t1,u)))
        Y.append(x_t2 - x_t1)

        x_t1 = x_t2

    return jnp.stack(X), jnp.stack(Y)

X1, Y1 = rollout()

for rollout in range(1, num_random_rollouts):
    X2, Y2 = rollout()
    X1 = jnp.vstack((X1,X2))
    Y1 = jnp.vstack((Y1,Y2))

X_Final = X1
Y_Final = Y1

# Step 2: PILCO Main Loop

RBF_Kernel = jk.RBF(active_dims=[0, 1, 2, 3, 4])

def CalculateMean(M_ts1, S_ts1, DynamicsModels, DynamicsModelsData, num_outputs=4):
    DeltaMu = []

    for a in range(num_outputs): # run loop for each dynamics model a
        params, _ = DynamicsModels[a].unpack()
        GP_x = DynamicsModelsData[a].X
        GP_y = DynamicsModelsData[a].y


        GP_Correction = np.asscalar(1 / params['kernel']['variance'])

        K = RBF_Kernel.cross_covariance(params['kernel'], GP_x, GP_x)

        Beta = GP_Correction * jnp.linalg.inv(K) @ GP_y

        q = []

        for x in X_Final:

            Lambda = jnp.diag(params['kernel']['lengthscale'])

            v_i = x - M_ts1      
            
            A =  (1 / jnp.sqrt(jnp.abs(jnp.linalg.det(M_ts1 @ jnp.linalg.inv(Lambda) + jnp.eye(num_outputs + 1)))))

            B = jnp.exp( -0.5 * (jnp.transpose(v_i) @ jnp.linalg.inv((S_ts1 + Lambda)) @ v_i))

            q.append(A * B)   

        q = jnp.asarray(q)
        DeltaMu.append(jnp.transpose(Beta) @ q)

    DeltaMu = jnp.asarray(DeltaMu)
    return DeltaMu

def CalculateCovariance(M_ts1, S_ts1, DynamicsModels, DynamicsModelsData, DeltaMu, num_outputs=4):

    DeltaSigma = []

    for a in range(num_outputs):

        DeltaSigma_Row = []

        for b in range(num_outputs):

            GP_Params_A, _ = DynamicsModels[a].unpack()
            GP_Params_B, _ = DynamicsModels[b].unpack()

            GP_Ax = DynamicsModelsData[a].X
            GP_Ay = DynamicsModelsData[a].y

            GP_Bx = DynamicsModelsData[b].X

            GP_Correction_A =  np.asscalar(1 / GP_Params_A['kernel']['variance'])
            GP_Correction_B = np.asscalar(1 / GP_Params_B['kernel']['variance'])

            K_A = GP_Correction_A * RBF_Kernel.cross_covariance(GP_Params_A['kernel'], GP_Ax, GP_Ax)
            K_B = GP_Correction_B * RBF_Kernel.cross_covariance(GP_Params_B['kernel'], GP_Bx, GP_Bx)

            Beta_A = jnp.linalg.inv(K_A) @ GP_Ay

            Beta_B = jnp.linalg.inv(K_B) @ GP_Bx

            Q = []

            Lambda_A = jnp.diag(GP_Params_A['kernel']['lengthscale'])

            Lambda_B = jnp.diag(GP_Params_B['kernel']['lengthscale'])

            R = S_ts1 * (jnp.linalg.inv(Lambda_A) + jnp.linalg.inv(Lambda_B)) + jnp.eye(num_outputs + 1)

            R_Prime = jnp.sqrt(jnp.abs(jnp.linalg.det(R)))

            if CalcCov:
                for i in X_Final: # Calculate Entries of Q_ij

                    Q_Row = []

                    for j in X_Final:

                        k_a = GP_Correction_A * RBF_Kernel.__call__(GP_Params_A['kernel'], i, M_ts1)

                        k_b = GP_Correction_B * RBF_Kernel.__call__(GP_Params_B['kernel'], j, M_ts1)

                        z = (jnp.linalg.inv(Lambda_A) @ (i - M_ts1)) + (jnp.linalg.inv(Lambda_B) @ (j - M_ts1))

                        A = (k_a * k_b) / R_Prime

                        B = jnp.exp(0.5 * (jnp.transpose(z) @ jnp.linalg.inv(R) @ (S_ts1 * z)))

                        Product = jnp.nan_to_num(A*B)

                        Q_Row.append(Product)

                        print('Calculated Q_' + str(i) + str(j) + ' @ ' + '(' + str(a) + ', ' + str(b) + ') = ' + str(Product))

                    Q.append(Q_Row)

                Q = jnp.asarray(Q)
            else:
                Q = 0.0001 * np.random.uniform(0,1,(25,25))

            E_fts1 = 1 - jnp.trace(K_A @ Q)
            print(E_fts1)

            E_fxts1 =  jnp.mean((jnp.transpose(Beta_A) @ Q @ Beta_B))
            print(E_fxts1)

            if (a == b):
                s_aa = E_fts1 + E_fxts1 - (DeltaMu[a] * DeltaMu[a])
                DeltaSigma_Row.append(jnp.nan_to_num(s_aa))
            else:
                s_ab = E_fxts1 - (DeltaMu[a] * DeltaMu[b])
                DeltaSigma_Row.append(jnp.nan_to_num(s_ab)) 

        DeltaSigma_Row = jnp.asarray(DeltaSigma_Row)
        DeltaSigma.append(DeltaSigma_Row)

    DeltaSigma = jnp.asarray(DeltaSigma)
    return DeltaSigma

for rollout in range(num_rollouts):
    print("PILCO Iteration " + str(rollout + 1))

    import pdb
    
    ### Step 3: Learn GP Dynamics Models

    ## Settings
    num_outputs = 4 # For multivariate targets, we train conditionally independent GPs for each target dimension. For Cart-Pole Swing-Up, x = [*,*,*,*] -> d = 4

    optimizer_lr = 0.01 # optimizer learning rate
    optimizer_iterations = 1 # how many iterations hyperparameters are optimized

    ## Variables
    DynamicsModels = []
    DynamicsModelsData = []

    States = []
    DeltaT = []
    Mean_t = []
    Sigma_t = []
    
    M_ts1 = X_Final[0] + Y_Final[0].mean()
    S_ts1 = np.cov(Y_Final[0])

    States.append(X_Final[0][:4])
    DeltaT.append(Y_Final[0])
    Mean_t.append(M_ts1)
    Sigma_t.append(S_ts1)

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

    for T_Step in range(T): # Propagate through GP dynamics models to calculate J^{\pi}(\theta)
        print('PILCO Iteration ' + str(rollout +1) + ' at t = ' + str(T_Step))
        
        # Calculate Mean of N(x_t+1 | \mu_{t-1}, \Sigma_{x_{t-1})
        
        DeltaMu = CalculateMean(M_ts1, S_ts1, DynamicsModels, DynamicsModelsData)

        print('Calculated Delta Mu as')
        print(DeltaMu)

        # Calculate Covariance of N(x_t+1 | \mu_{t-1}, \Sigma_{x_{t-1})

        DeltaSigma = CalculateCovariance(M_ts1, S_ts1, DynamicsModels, DynamicsModelsData, DeltaMu)
        print('Calculated Delta Sigma as')
        print(DeltaSigma)

        # Calculate N(x_t | \mu_{t}, \Sigma_{x_{t})
        print()
        print()
        
        XY = jnp.stack((States[T_Step], DeltaT[T_Step]), axis=1)
        Cdx = jnp.cov(XY)

        M_t = M_ts1[:4] + DeltaMu.flatten()
        S_t = S_ts1 + DeltaSigma.flatten().reshape(4,4) + Cdx + Cdx.T

        x_t = jr.multivariate_normal(key, M_t, S_t, method='svd')
        u_t = Controller.compute_action(x_t)

        next_state = Env.step(x_t, u_t)

        dt = next_state - States[T_Step]

        M_ts1 = jnp.append(x_t, u_t) + jnp.mean(dt)
        S_ts1 = jnp.cov(dt)

        States.append(next_state)
        DeltaT.append(dt)
        Mean_t.append(M_t)
        Sigma_t.append(S_t)

    # Evaluate J^{\pi}

    