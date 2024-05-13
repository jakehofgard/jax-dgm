# JAX imports
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jaxopt import OptaxSolver
import optax
import jax.lax as lax
from flax import linen as nn
from flax.serialization import to_state_dict

# Miscellaneous imports
import numpy as np
from tqdm.notebook import trange
import matplotlib.pyplot as plt

# LSTM-like layer used in DGM
class LSTMLayer(nn.Module):
    units: int
    trans: str = 'tanh'

    @nn.compact
    def __call__(self, X, S):
        # Define the transformation function
        if self.trans == 'tanh':
            trans_fn = jnp.tanh
        elif self.trans == 'relu':
            trans_fn = jax.nn.relu
        elif self.trans == 'sigmoid':
            trans_fn = jax.nn.sigmoid

        # Initialize weights and biases
        Uz = self.param('Uz', nn.initializers.normal(), (X.shape[-1], self.units))
        Ug = self.param('Ug', nn.initializers.normal(), (X.shape[-1], self.units))
        Ur = self.param('Ur', nn.initializers.normal(), (X.shape[-1], self.units))
        Uh = self.param('Uh', nn.initializers.normal(), (X.shape[-1], self.units))

        Wz = self.param('Wz', nn.initializers.normal(), (self.units, self.units))
        Wg = self.param('Wg', nn.initializers.normal(), (self.units, self.units))
        Wr = self.param('Wr', nn.initializers.normal(), (self.units, self.units))
        Wh = self.param('Wh', nn.initializers.normal(), (self.units, self.units))

        bz = self.param('bz', nn.initializers.normal(), (self.units,))
        bg = self.param('bg', nn.initializers.normal(), (self.units,))
        br = self.param('br', nn.initializers.normal(), (self.units,))
        bh = self.param('bh', nn.initializers.normal(), (self.units,))

        # Compute components of LSTMLayer output
        Z = trans_fn(jnp.dot(X, Uz) + jnp.dot(S, Wz) + bz)
        G = trans_fn(jnp.dot(X, Ug) + jnp.dot(S, Wg) + bg)
        R = trans_fn(jnp.dot(X, Ur) + jnp.dot(S, Wr) + br)
        H = trans_fn(jnp.dot(X, Uh) + jnp.dot(S * R, Wh) + bh)

        # Compute LSTMLayer outputs
        S_new = (1 - G) * H + Z * S
        return S_new

# Fully connected (dense) layer
class DenseLayer(nn.Module):
    units: int
    trans: str = None

    @nn.compact
    def __call__(self, X):
        # Define the transformation function
        if self.trans == 'tanh':
            trans_fn = jnp.tanh
        elif self.trans == 'relu':
            trans_fn = jax.nn.relu
        elif self.trans == 'sigmoid':
            trans_fn = jax.nn.sigmoid
        else:
            trans_fn = lambda x: x

        # Initialize weights and biases
        W = self.param('W', nn.initializers.normal(), (X.shape[-1], self.units))
        b = self.param('b', nn.initializers.normal(), (self.units,))

        # Compute DenseLayer output
        S = jnp.dot(X, W) + b
        return trans_fn(S)

# Neural network architecture used in DGM
class DGMNet(nn.Module):
    units: int
    n_layers: int
    final_trans: str = None

    @nn.compact
    def __call__(self, x):
        # Initial inputs are measure-time pairs [m, t]
        X = x

        # Initial layer
        S = DenseLayer(self.units, trans='tanh')(X)

        # Intermediate LSTM layers
        for _ in range(self.n_layers):
            S = LSTMLayer(self.units, trans='tanh')(X, S)

        # Final layer
        result = DenseLayer(1, trans=self.final_trans)(S)
        return result
    
# Terminal cost function g
@jit
def g(m):
    return 10 * m

# Auxiliary function for Hamiltonian
@jit
def a_star(r, M):
    return jnp.clip(r, 0, M)

# Hamiltonian for Example A.1
@jit
def hamiltonian(stack, c):
    m = stack[:, -1]
    z = stack[:, :d]

    # Store diagonal elements that shouldn't be included in Hamiltonian computation
    C = c[:d, :d]
    diag_C = jnp.diag(C)
    diag_z = jnp.diag(z)

    # Vectorized computation of Hamiltonian
    s = -jnp.sum(a_star(-z * C, M) * z + 1 / 2 * jnp.square(a_star(-z * C, M)), axis=1) + \
        a_star(-diag_z * diag_C, M) * diag_z + 1 / 2 * jnp.square(a_star(-diag_z * diag_C, M)) - \
        2 * m

    return s

# Uniformly sample points for the PDE and terminal conditions
def sampler(key, nSim_interior, nSim_terminal, t_low, T, m_low, m_high, d):
    # Sampler for the interior
    key, subkey = random.split(key)
    t_interior = random.uniform(subkey, (nSim_interior, 1), minval=t_low, maxval=T)
    m_interior = random.uniform(key, (nSim_interior, d), minval=m_low, maxval=m_high)
    m_interior_sum = jnp.sum(m_interior, axis=1, keepdims=True)
    m_interior = m_interior / m_interior_sum

    interior_pts = jnp.concatenate([m_interior, t_interior], axis=1)

    # Sampler for the terminal time
    key, subkey = random.split(key)
    t_terminal = T * jnp.ones((nSim_terminal, 1))
    m_terminal = random.uniform(subkey, (nSim_terminal, d), minval=m_low, maxval=m_high)
    m_terminal_sum = jnp.sum(m_terminal, axis=1, keepdims=True)
    m_terminal = m_terminal / m_terminal_sum

    terminal_pts = jnp.concatenate([m_terminal, t_terminal], axis=1)

    return interior_pts, terminal_pts

# MFCP parameters
nSim_interior = 10000  # Number of interior samples
nSim_terminal = 10000  # Number of terminal samples
t_low = 0.0  # Lower time bound
T = 1.0  # Terminal time
m_low = 0.0  # Lower space bound
m_high = 1.0  # Upper space bound
d = 2  # Dimension of space
M = 20 # Upper bound on controls

# Set cost matrix, values are not particularly important
if d > 10:
  costs = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
  c = np.random.choice(costs, (d, d))
  np.fill_diagonal(c, 0)
  c = jnp.array(c)
else:
  c = jnp.array([[0.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [2.0, 0.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [10.0, 2.0, 0.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [3.0, 2.0, 5.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [1.0, 2.0, 5.0, 3.0, 0.0, 2.0, 3.0, 5.0, 10.0, 1.0],
              [10.0, 2.0, 5.0, 3.0, 1.0, 0.0, 3.0, 5.0, 10.0, 1.0],
              [2.0, 2.0, 5.0, 3.0, 1.0, 2.0, 0.0, 5.0, 10.0, 1.0],
              [2.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 0.0, 10.0, 1.0],
              [5.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 0.0, 1.0],
              [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 0.0]])

# Network hyperparameters
n_layers = 3
nodes_per_layer = 50

# Training parameters
sampling_stages = 200  # number of times to resample new time-space domain points
steps_per_sample = 10  # number of SGD steps to take before re-sampling

key = random.PRNGKey(0)
interior_pts, terminal_pts = sampler(key, nSim_interior, nSim_terminal, 
                                     t_low, T, m_low, m_high, d)

# Jitted loss function for HJB equation of the MFC problem
@jit
def loss(params, interior_pts, terminal_pts, c, M, use_unif=True):
    # Compute derivatives at current sampled points in the interior
    V = lambda p: model.apply(params, p).reshape()
    dV = vmap(grad(V))(interior_pts)
    Vm, Vt = dV[:, :d], dV[:, -1]

    m_interior = interior_pts[:, :d]
    m_terminal = terminal_pts[:, :d]

    # Compute PDE loss, vectorized
    Vms = jnp.tile(Vm[:, None, :], (1, d, 1))
    Vmi = jnp.tile(Vm[:, :, None], (1, 1, d))
    diffs = Vms - Vmi
    stack = jnp.concatenate([diffs, m_interior[:, :, None]], axis=-1)

    sum = jnp.sum(m_interior * vmap(hamiltonian, in_axes=(0, None))(stack, c), 
                  axis=1, 
                  keepdims=True)

    diff_V = -Vt + sum
    L1 = jnp.max(jnp.abs(diff_V)) if use_unif else jnp.mean(diff_V ** 2)

    # Compute terminal loss
    target_value = jnp.sum(m_terminal * g(m_terminal), axis=1, keepdims=True)
    fitted_value = model.apply(params, terminal_pts)

    L3 = jnp.max(jnp.abs(fitted_value - target_value)) if use_unif else jnp.mean((fitted_value - target_value) ** 2)

    return L1 + L3, L1

total_steps = steps_per_sample * sampling_stages

# Tuned peak values
# Uniform: 0.0008
# L^2: 0.0005
lr_scheduler = optax.cosine_onecycle_schedule(transition_steps=total_steps, 
                                              peak_value=0.0008, 
                                              pct_start=0.30, 
                                              div_factor=30., 
                                              final_div_factor=100.)

# Set optimizer, Adam with weight decay and gradient clipping
opt = optax.chain(
  optax.clip(2.0), # best value = 2.0
  optax.adamw(learning_rate=lr_scheduler),
)

# Initialize solver
solver = OptaxSolver(opt=opt, fun=loss,
                    maxiter=steps_per_sample * sampling_stages, has_aux=True)

# Initialize parameters
rng = jax.random.PRNGKey(0)
model = DGMNet(units=nodes_per_layer, n_layers=n_layers, final_trans=None)
params = model.init(key, jnp.zeros((1, d+1)))

# Run training loop
state = solver.init_state(params, interior_pts, terminal_pts, c, M)
jitted_update = jax.jit(solver.update)

# Save all three loss metrics
total_losses = []
pde_losses = []
terminal_losses = []

# Run training loop, saving best model at each step
min_loss = np.inf
# Time epochs
for epoch in trange(sampling_stages):
  # Sample according to the above scheme
  interior_pts, terminal_pts = sampler(key, 
                                      nSim_interior, 
                                      nSim_terminal, 
                                      t_low, T, m_low, m_high, d)
  for _ in range(steps_per_sample):
    # Take the specified number of gradient steps on a sample in each epoch
    params, state = jitted_update(params=params,
                                  state=state,
                                  interior_pts=interior_pts,
                                  terminal_pts=terminal_pts,
                                  c=c,
                                  M=M)
    
    # Save loss at each iteration, and best model loss decreases
    total_loss = state.value
    pde_loss = state.aux
    terminal_loss = total_loss - pde_loss

    # Save best model and losses
    if total_loss < min_loss:
      best_model = to_state_dict(model)
    total_losses.append(total_loss)
    pde_losses.append(pde_loss)
    terminal_losses.append(terminal_loss)