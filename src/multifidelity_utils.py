# multifidelity_utils.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import scipy.io
import mat73
from scipy.interpolate import griddata
import os
import pickle
import time
from sklearn.utils import extmath
import h5py
import matplotlib.tri as tri

def custom_loss(y_pred, y_true):
    """Custom loss function that handles NaN values in predictions"""
    goodind = tf.math.logical_not(tf.math.is_nan(y_pred))
    y_pred_loss = tf.boolean_mask(y_pred, goodind)
    y_pred_true = tf.boolean_mask(y_true, goodind)
    return K.mean(K.square(y_pred_loss - y_pred_true))

def sliding_windows(data_input, data_output, seq_length, freq=1):
    """
    Create sliding windows for sequence data with padding
    """
    pad_width = seq_length
    padded_input = np.pad(data_input, ((0,0), (pad_width, pad_width), (0,0)), 
                         'constant', constant_values=(0.))
    padded_output = np.pad(data_output, ((0,0), (pad_width, pad_width), (0,0)), 
                          'constant', constant_values=(np.nan,))
    
    x, y = [], []
    
    for i in range(padded_input.shape[0]):
        for j in range(0, padded_input.shape[1] - seq_length + 1, freq):
            _x = padded_input[i, j:(j + seq_length), :]
            _y = padded_output[i, j:(j + seq_length), :]
            x.append(_x)
            y.append(_y)

    return np.array(x), np.array(y)

def compute_randomized_SVD(S, N_POD, N_h, n_channels, name='', verbose=False):
    if verbose:
        print('Computing randomized POD...')
    U = np.zeros((n_channels * N_h, N_POD))
    start_time = time.time()
    for i in range(n_channels):
        U[i * N_h: (i + 1) * N_h], Sigma, Vh = extmath.randomized_svd(S[i * N_h: (i + 1) * N_h, :],
                                                                      n_components=N_POD, transpose=False,
                                                                      flip_sign=False, random_state=123)
        if verbose:
            print('Done... Took: {0} seconds'.format(time.time() - start_time))

    if verbose:
        I = 1. - np.cumsum(np.square(Sigma)) / np.sum(np.square(Sigma))
        print(I[-1])

    if name:
        sio.savemat(name, {'V': U[:, :N_POD]})

    return U, Sigma

def load_reaction_diffusion(params, fidelity, path, splitted = False):
    u_list = []
    for param in params:
        name = path + 'u_' + fidelity + '_' + "{:.3f}".format(param) 
        
        if splitted:
            u_test_list = []
            for i in [1,2]:
                u_test = scipy.io.loadmat(name + '_' + str(i) + '.mat')['u']
                u_test_list.append(u_test)
            u = np.concatenate(u_test_list, axis = 2)
        else:
            u = scipy.io.loadmat(name + '.mat')['u']
            
        u_list.append(u)
    
    data_u = np.stack(u_list, axis=3)
    x = scipy.io.loadmat(path + 'x_' + fidelity + '.mat')['x']
    t = scipy.io.loadmat(path + 't_' + fidelity + '.mat')['t']
    
    return data_u, x.flatten(), t.flatten()

def load_navier_stokes(path, params, train_test, t0, T, dt, verbose=0):
    data_snap = []
    data_drag = []
    data_lift = []

    # time range
    start = int(t0 / dt)
    end = int(T / dt)

    dofs_vx = list(range(3899))
    dofs_vy = list(range(15270, 15270 + 3899))
    dovs_p = list(range(15270 * 2, 15270 * 2 + 3899))

    dofs = dofs_vx + dofs_vy  # + dovs_p
    for param in params:

        # Load snapshot data
        name_snap = (
            path + "/snapshots" + "/snap_" + str(int(param)) + "_" + train_test + ".mat"
        )
        with h5py.File(name_snap, "r") as file:
            snap = file["snapshots"][:].T
        data_snap.append(snap[start:end, :])

        # Load drag and lift data
        name_drag = (
            path + "/drag" + "/drag_" + str(int(param)) + "_" + train_test + ".mat"
        )
        with h5py.File(name_drag, "r") as file:
            drag = file["Drag"][:].T
        data_drag.append(drag[start:end, :])

        name_lift = (
            path + "/lift" + "/lift_" + str(int(param)) + "_" + train_test + ".mat"
        )
        with h5py.File(name_lift, "r") as file:
            lift = file["Lift"][:].T
        data_lift.append(lift[start:end, :])

    data_snap = np.array(data_snap)[:, :, dofs]
    data_drag = np.array(data_drag)
    data_lift = np.array(data_lift)
    if verbose:
        print("Loaded data for param = ", params)
        print("Snap shape = ", data_snap.shape)
        print("Drag shape = ", data_drag.shape)
        print("Lift shape = ", data_lift.shape)

    return data_snap, data_drag, data_lift

def plot_pod_results(t_hf_test, y_true, mean_pred, std_pred, n_modes=9, title="POD Results"):
    """
    Plot POD mode predictions with uncertainty
    """
    fig = plt.figure(figsize=(15, 15))
    for i in range(min(n_modes, 9)):
        ax = plt.subplot(331 + i)
        plt.plot(t_hf_test, y_true[:, i], label='HF', color='red')
        plt.plot(t_hf_test, mean_pred[:, i], label='mean', color='blue')
        plt.fill_between(t_hf_test, 
                        mean_pred[:, i] - 2*std_pred[:, i], 
                        mean_pred[:, i] + 2*std_pred[:, i], 
                        color='lightblue', alpha=0.5, label='mean ± std')
        ax.title.set_text(f'POM {i+1}')
        plt.xlabel('t')
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def create_model_inputs(t_data, mu_data, scale_time=1.0, scale_param=1.0, model_type='dense'):
    """
    Create model inputs in appropriate format
    """
    if model_type.lower() == 'dense':
        t_in = np.tile(t_data, len(mu_data)) / scale_time
        mu_in = np.repeat(mu_data, len(t_data)) / scale_param
        return np.vstack((t_in, mu_in)).T
    else:  # LSTM
        t_lstm = np.tile(t_data, len(mu_data)).T.reshape(len(mu_data), -1, 1) / scale_time
        mu_lstm = np.repeat(mu_data, len(t_data)).reshape(len(mu_data), -1, 1) / scale_param
        return np.concatenate((t_lstm, mu_lstm), axis=2)

def setup_sensor_data(u_hf, u_hf_test, n_sensors=2, noise=False, noise_sigma=0.4):
    """
    Setup sensor data from HF solutions
    """
    if noise:
        scale_noise = np.exp(0)  # mu=0 for lognormal
        u_hf_noise = u_hf * np.random.lognormal(mean=0, sigma=noise_sigma, size=u_hf.shape) * scale_noise
        u_hf_test_noise = u_hf_test * np.random.lognormal(mean=0, sigma=noise_sigma, size=u_hf_test.shape) * scale_noise
    else:
        u_hf_noise, u_hf_test_noise = u_hf, u_hf_test
    
    # Select sensors from boundary points
    x2_sensors = np.stack((u_hf_noise[0,0,:,:], u_hf_noise[0,-1,:,:], 
                          u_hf_noise[-1,0,:,:], u_hf_noise[-1,-1,:,:]), axis=0).swapaxes(0, 2)
    x2_sensors_test = np.stack((u_hf_test_noise[0,0,:,:], u_hf_test_noise[0,-1,:,:],
                              u_hf_test_noise[-1,0,:,:], u_hf_test_noise[-1,-1,:,:]), axis=0).swapaxes(0, 2)
    
    return x2_sensors, x2_sensors_test

def calculate_errors(y_true, y_pred):
    """
    Calculate various error metrics
    """
    mse = np.mean(np.square(y_pred - y_true))
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(mse)
    
    return {'mse': mse, 'mae': mae, 'rmse': rmse}

def save_model_results(model, predictions, errors, filepath):
    """
    Save model results to file
    """
    results = {
        'model_config': model.get_config() if hasattr(model, 'get_config') else None,
        'predictions': predictions,
        'errors': errors
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

def load_model_results(filepath):
    """
    Load saved model results
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    

def load_mesh(mesh_file_path):
    """Load mesh data from a .msh file."""
    with open(mesh_file_path, 'r') as f:
        lines = f.readlines()
    
    # --- Nodes ---
    nodes_start = next(i for i, line in enumerate(lines) if '$Nodes' in line) + 1
    nnodes = int(lines[nodes_start].strip())
    vertex_data = np.array([list(map(float, lines[nodes_start + 1 + i].split())) 
                            for i in range(nnodes)])
    vertex_data = vertex_data[vertex_data[:, 0].argsort()]  # sort by index
    vertices = vertex_data[:, [1, 2]]

    # --- Elements ---
    elements_start = next(i for i, line in enumerate(lines) if '$Elements' in line) + 1
    nelem = int(lines[elements_start].strip())
    connectivity_data, element_centers = [], []
    
    for i in range(nelem):
        line_data = lines[elements_start + 1 + i].split()
        if len(line_data) == 8:  # triangular
            nodes = [int(x) - 1 for x in line_data[-3:]]
            connectivity_data.append(nodes)
            element_centers.append(np.mean(vertices[nodes], axis=0))
    
    connectivity = np.array(connectivity_data)
    triangulation = tri.Triangulation(vertices[:, 0], vertices[:, 1], connectivity)

    return vertices, connectivity, np.array(element_centers), triangulation


def _prepare_data_for_plotting(vertices, connectivity, data_values):
    """Handle whether data is vertex-based or element-based."""
    n_vertices = len(vertices)
    n_elements = len(connectivity)
    data_length = len(data_values)

    if data_length == n_vertices:
        return data_values
    elif data_length == n_elements:
        # Interpolate element data to vertices
        vertex_data = np.zeros(n_vertices)
        vertex_counts = np.zeros(n_vertices)
        for elem_idx, element in enumerate(connectivity):
            for vertex_idx in element:
                vertex_data[vertex_idx] += data_values[elem_idx]
                vertex_counts[vertex_idx] += 1
        vertex_counts[vertex_counts == 0] = 1
        return vertex_data / vertex_counts
    else:
        raise ValueError("Data size does not match vertices or elements")


def plot_snapshot(data, timestep, n_simulation, triangulation,
                  vertices, connectivity, ax=None, title=None, cmap="viridis",
                  minmax=None, show_colorbar=True, show_y=True, figsize=(10, 6)):
    """
    Plot a single timestep of data (2D/3D array) on a mesh.
    """
    # Select slice
    if data.ndim == 3:
        plot_data = data[n_simulation, timestep, :]
    elif data.ndim == 2:
        plot_data = data[timestep, :]
    else:
        raise ValueError("Data must be 2D or 3D")

    # Interpolate if needed
    plot_data = _prepare_data_for_plotting(vertices, connectivity, plot_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    tcf = ax.tricontourf(
        triangulation, plot_data, levels=50, cmap=cmap,
        vmin=minmax[0] if minmax else None,
        vmax=minmax[1] if minmax else None
    )

    if show_colorbar:
        cbar = fig.colorbar(tcf, ax=ax)
        cbar.set_label("Value")
        if minmax:
            cbar.set_ticks(minmax)
            cbar.set_ticklabels([f"{v:.2f}" for v in minmax])

    ax.triplot(triangulation, 'k-', alpha=0.1, linewidth=0.3)
    ax.set_aspect("equal")
    ax.set_xlabel("X", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)

    if show_y:
        ax.set_ylabel("Y", rotation=0, fontsize=14)
    else:
        ax.set_yticks([])
        ax.set_ylabel("")

    if title:
        ax.set_title(title)

    return fig, ax
