"""# Mean Field Control Problem"""

import tensorflow as tf
import numpy as np

# Neural network parameters
n_layers = 3
nodes_per_layer = 50
learning_rate = 0.001
num_samples = 100

# Training parameters
sampling_stages = 40  # number of times to resample new time-space domain points
steps_per_sample = 10  # number of SGD steps to take before re-sampling

# MFCP parameters

d = 10

t_low = 0.0
T = 20.0

m_low = 0.0
m_high = 1.0
M = 20

# c = np.ones(shape = [d, d], dtype = np.float32)

# c = np.array([[0.0, 10.0],
#               [10.0, 0.0]])

# Create cost array for d > 10
costs = np.array([1.0, 2.0, 3.0, 5.0, 10.0])
c = np.random.choice(costs, (d, d))
np.fill_diagonal(c, 0)

# Cost array for d <= 10
# c = np.array([[0.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
#               [2.0, 0.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
#               [10.0, 2.0, 0.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
#               [3.0, 2.0, 5.0, 0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 1.0],
#               [1.0, 2.0, 5.0, 3.0, 0.0, 2.0, 3.0, 5.0, 10.0, 1.0],
#               [10.0, 2.0, 5.0, 3.0, 1.0, 0.0, 3.0, 5.0, 10.0, 1.0],
#               [2.0, 2.0, 5.0, 3.0, 1.0, 2.0, 0.0, 5.0, 10.0, 1.0],
#               [2.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 0.0, 10.0, 1.0],
#               [5.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 0.0, 1.0],
#               [1.0, 2.0, 5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 10.0, 0.0]])


# Terminal cost function g
def g(m):
    return 10 * m


# Hamiltonian function H
def a_star(r):
    result = tf.clip_by_value(r, 0, M)
    return result


def hamiltonian(stack):
    m = tf.squeeze(stack[:, -1])
    z = stack[:, :d]

    # Store diagonal elements that shouldn't be included in Hamiltonian computation
    C = c[:d, :d]
    diag_C = tf.cast(tf.linalg.diag_part(C), tf.float32)
    diag_z = tf.cast(tf.linalg.diag_part(z), tf.float32)

    # Vectorized computation of Hamiltonian
    s = -tf.math.reduce_sum(a_star(-z * C) * z + 1 / 2 * tf.math.square(a_star(-z * C)), axis=1) + \
        a_star(-diag_z * diag_C) * diag_z + 1 / 2 * tf.math.square(a_star(-tf.math.multiply(diag_z, diag_C))) - \
        2 * m

    return s


# Sampling parameters
nSim_interior = num_samples
nSim_terminal = num_samples


# Sampling function - randomly sample time-space pairs
def sampler(nSim_interior, nSim_terminal):
    ''' Sample time-space points from the function's domain;
        here each space point is the probability vector of staying at each state;
        points are sampled uniformly on the interior of the domain as well as at the terminal time points.
    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
    '''

    # Sampler 1st: domain interior
    t_interior = np.random.uniform(low=t_low, high=T, size=[nSim_interior, 1])
    m_interior = np.random.uniform(low=m_low, high=m_high, size=[nSim_interior, d])
    m_interior_sum = np.sum(m_interior, axis=1).reshape([nSim_interior, 1])
    m_interior = m_interior / m_interior_sum

    # Sampler 2nd: spatial boundary
    # no spatial boundary condition for this problem

    # Sampler 3rd: initial/terminal condition
    t_terminal = T * np.ones((nSim_terminal, 1))
    m_terminal = np.random.uniform(low=m_low, high=m_high, size=[nSim_terminal, d])
    m_terminal_sum = np.sum(m_terminal, axis=1).reshape([nSim_terminal, 1])
    m_terminal = m_terminal / m_terminal_sum

    t_interior = tf.convert_to_tensor(t_interior, dtype="float32")
    m_interior = tf.convert_to_tensor(m_interior, dtype="float32")
    t_terminal = tf.convert_to_tensor(t_terminal, dtype="float32")
    m_terminal = tf.convert_to_tensor(m_terminal, dtype="float32")

    return t_interior, m_interior, t_terminal, m_terminal


# Loss function for HJB equation of the MFC problem
def loss(model, t_interior, m_interior, t_terminal, m_terminal, nSim_interior, nSim_terminal, use_unif=True):
    ''' Compute total loss for training.
    Args:
        model:         DGM model object
        t_interior:    sampled time points in the interior of the function's domain
        m_interior:    sampled space points in the interior of the function's domain
        t_terminal:    sampled time points at terminal time (vector of terminal times)
        m_terminal:    sampled space points at terminal time
        nSim_interior: number of space points in the interior of the function's domain to sample
        nSim_terminal: number of space points at terminal time to sample (terminal condition)
        use_unif:          if True, use uniform loss instead of squared error
    '''

    t = t_interior
    m = m_interior

    # compute derivatives at current sampled points in the interior
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(m)
        V = model(m, t)
    Vm = tape.gradient(V, m)

    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(t)
        V = model(m, t)
    Vt = tape2.gradient(V, t)

    # Compute PDE loss, vectorized

    Vms = tf.tile(tf.reshape(Vm, [nSim_interior, 1, d]), tf.constant([1, d, 1], tf.int32))
    Vmi = tf.tile(tf.reshape(Vm, [nSim_interior, d, 1]), tf.constant([1, 1, d], tf.int32))
    diffs = Vms - Vmi
    stack = tf.concat([diffs, tf.reshape(m, [nSim_interior, d, 1])], axis=-1)

    sum = tf.reduce_sum(m * tf.map_fn(hamiltonian, stack), axis=1, keepdims=True)

    diff_V = -Vt + sum
    if use_unif:
        L1 = tf.reduce_max(tf.abs(diff_V))
    else:
        L1 = tf.reduce_mean(tf.square(diff_V))

    # Loss term 3rd: initial/terminal condition
    target_value = tf.reduce_sum(m_terminal * g(m_terminal), axis=1, keepdims=True)
    fitted_value = model(m_terminal, t_terminal)

    # compute average L2-error (resp. uniform) of terminal condition
    if use_unif:
        L3 = tf.reduce_max(tf.abs(fitted_value - target_value))
    else:
        L3 = tf.reduce_mean(tf.square(fitted_value - target_value))

    return L1, L3


# Gradient function (of loss function)
def grad(model, t_interior, m_interior, t_terminal, m_terminal, nSim_interior, nSim_terminal):
    V = model(m_interior, t_interior)
    W1 = model.initial_layer.W
    b1 = model.initial_layer.b
    W_last = model.final_layer.W
    b_last = model.final_layer.b
    Uz_list = []
    Ug_list = []
    Ur_list = []
    Uh_list = []
    Wz_list = []
    Wg_list = []
    Wr_list = []
    Wh_list = []
    bz_list = []
    bg_list = []
    br_list = []
    bh_list = []
    for i in range(n_layers):
        Uz_list.append(model.LSTMLayerList[i].Uz)
        Ug_list.append(model.LSTMLayerList[i].Ug)
        Ur_list.append(model.LSTMLayerList[i].Ur)
        Uh_list.append(model.LSTMLayerList[i].Uh)
        Wz_list.append(model.LSTMLayerList[i].Wz)
        Wg_list.append(model.LSTMLayerList[i].Wg)
        Wr_list.append(model.LSTMLayerList[i].Wr)
        Wh_list.append(model.LSTMLayerList[i].Wh)
        bz_list.append(model.LSTMLayerList[i].bz)
        bg_list.append(model.LSTMLayerList[i].bg)
        br_list.append(model.LSTMLayerList[i].br)
        bh_list.append(model.LSTMLayerList[i].bh)

    parameter_set = ([W1, b1, W_last, b_last] + Uz_list + Ug_list + Ur_list + Uh_list + Wz_list + Wg_list
                     + Wr_list + Wh_list + bz_list + bg_list + br_list + bh_list)

    with tf.GradientTape(persistent=True) as tape:
        L1, L3 = loss(model, t_interior, m_interior, t_terminal, m_terminal, nSim_interior, nSim_terminal)
        Loss = L1 + L3
    return Loss, L1, L3, tape.gradient(Loss, parameter_set), parameter_set