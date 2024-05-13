"""# Train"""
import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer

import mfcp
from mfcp import grad, sampler
from arch import DGMNet

# Save paths for trial data, weights, figures
TRIAL_PATH = f"tests/loss_uniferror_d={mfcp.d}.csv"
CHECK_PATH = f"checkpoints/best_model_unif_d={mfcp.d}"

# Only save figure if d = 2
FIG_PATH = f"figures/mfcp_value_d=2_uniferror.eps"


# Train neural network and save outputs
def train_dgm(trial_path=TRIAL_PATH, check_path=CHECK_PATH, fig_path=FIG_PATH, save_trial=True, plot_surface=True):
    # Use DGM learning rate schedule
    start = timer()
    step = tf.Variable(0, trainable=False)
    boundaries = [5000, 10000, 20000, 30000, 40000, 45000]
    values = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
    learning_rate_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    lr_schedule = learning_rate_fn(step)

    # Set up network
    model = DGMNet(mfcp.nodes_per_layer, mfcp.n_layers)
    optimizer = tf.keras.optimizers.Adam(learning_rate=mfcp.learning_rate)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

    L_min = np.inf
    losses = np.zeros((mfcp.sampling_stages * mfcp.steps_per_sample, 3))

    # Train network
    for i in range(mfcp.sampling_stages):
        # sample uniformly from the required regions
        t_interior, m_interior, t_terminal, m_terminal = sampler(mfcp.nSim_interior, mfcp.nSim_terminal)

        # for a given sample, take the required number of SGD steps
        for j in tqdm(range(mfcp.steps_per_sample)):
            # Gradient function computes loss, so no need to do it twice
            Loss, L1, L3, grads, parameters = grad(
                model,
                t_interior, m_interior,
                t_terminal, m_terminal,
                mfcp.nSim_interior, mfcp.nSim_terminal
            )

            optimizer.apply_gradients(zip(grads, parameters))
            L1 = L1.numpy()
            L3 = L3.numpy()
            Loss = Loss.numpy()

            # Save all loss metrics for plotting
            losses[i * mfcp.steps_per_sample + j, :] = np.array([L1, L3, Loss])

        # Print trial information
        print(f"\nFinished epoch {i}")
        print(f"PDE loss: {L1},", f"Terminal loss: {L3},", f"Combined loss: {Loss}")

        # Save trial backup
        if save_trial:
            np.savetxt(trial_path, losses)

        # Save weights at end of epoch if best loss is achieved
        if save_trial and Loss <= L_min:
            print("Saving losses and weights")
            L_min = Loss
            model.save(check_path, save_format="tf")

    if plot_surface:
        # Make 3D plot for the value function (applicable for d = 2)
        t_num = int(mfcp.T)
        m_num = 100

        t_plot = np.linspace(0.0, mfcp.T, t_num, dtype=np.float32)
        m1 = np.linspace(0.0, 1.0, m_num, dtype=np.float32)
        m_plot = np.array([m1, 1.0 - m1]).T
        V = np.zeros((t_num, m_num))

        # Load best model
        best_model = load_model(check_path, compile=False)
        for i, t in enumerate(t_plot):
            V_i = best_model(m_plot, np.repeat(t, m_num).reshape(-1, 1)).numpy()
            V[i, :] = np.squeeze(V_i)

        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection='3d')
        ax.grid()

        ts, ms = np.meshgrid(t_plot, m1)
        surf = ax.plot_surface(ts, ms, V.T, cmap=plt.cm.cividis)
        ax.set_title(f'Approximate Value Function, Trained with $L^2$ Loss')

        # Set axes label
        ax.set_xlabel('Time ($t$)', labelpad=20)
        ax.set_ylabel('$m_1$', labelpad=20)
        ax.set_zlabel('$V(t, m_1, 1 - m_1)$', labelpad=20)

        ax.view_init(10, 40)
        plt.savefig(fig_path, format='eps')
        plt.show()

    end = timer()

    print(f"Total train time: {end - start} seconds")
    print(f"Final combined loss: {L_min}")


# Train DGM network using specifications in MFCP
train_dgm(TRIAL_PATH, CHECK_PATH, FIG_PATH, save_trial=True, plot_surface=False)