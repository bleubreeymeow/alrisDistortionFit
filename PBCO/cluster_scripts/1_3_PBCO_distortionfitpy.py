import os
os.environ['PYTHONHASHSEED'] = '1'  # Set a fixed seed for reproducibility

# os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # For within-op parallelism
# os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # For between-op parallelism

import numpy as np
import tensorflow as tf
import pandas as pd
from alris_one_third_functions import shift_atoms, transform_list_hkl_p63_p65, get_structure_factors , atom_position_list
import multiprocessing as mp
from time import time
from datetime import datetime

num_threads = 1 

# Configure TensorFlow to use multiple threads
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)


global features ,labels , labels_err, matrix , max_mode_amps , epochs , lr

def init_worker(seed, _features, _labels, _labels_err, _matrix, _max_mode_amps , _epochs , _lr):
    global features, labels, labels_err, matrix, max_mode_amps, epochs, lr
    features = _features
    labels = _labels
    labels_err = _labels_err
    matrix = _matrix
    max_mode_amps = _max_mode_amps
    epochs = _epochs
    lr = _lr

    # Create separate seed
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    worker_seed = seed + worker_id
    tf.keras.utils.set_random_seed(seed=worker_seed)

    print(f"worker {worker_id} initialised")



def run_iteration(iteration):
    global features, labels, labels_err, matrix, max_mode_amps , epochs , lr
    n_dim = 3

    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    n_epochs = epochs

 # Create the model
    inputs = tf.keras.Input(shape=(n_dim,))
    outputs = FunAsLayer(matrix , max_mode_amps)(inputs)
    model = tf.keras.Model(inputs, outputs)

    # Compile the model with the custom loss function and metric
    model.compile(
        optimizer=optim,
        loss= 'mse', # MSE_weighted() if using errors
        metrics=[r_factor_metric],
        run_eagerly=False,  # Set to True for debugging, False for performance
    )

    history = model.fit(
        x=features,
        y=labels,  # replace with combined_labels if using errors
        batch_size = features.shape[0], # Use a smaller batch size features.shape[0]
        epochs=n_epochs,
        verbose=0,
        shuffle=True, # not sure whether this matters
        # callbacks=[cb]
        sample_weight=labels_err  # Use sample weights if you have errors
    )

    final_loss = history.history['loss'][-1]
    best_model_pars = max_mode_amps * tf.tanh(model.layers[-1].get_weights()[0])
    y_pred = fun_tf(features, best_model_pars , matrix)
    labels = tf.reshape(labels, [-1])
    rf = r_factor_metric(labels, y_pred)

    return best_model_pars, final_loss , iteration , history.history['loss'] , rf

def fun_tf(hkl_list, pars, matrix):
    """
    Fast computation of structure factors with parameter-dependent structure.
    """

    # Get modified structure
    pars_tensor = tf.stack(pars)  # shape (params,)

    atom_shift_list = shift_atoms(matrix , (pars_tensor))
    atom_shift_list = atom_shift_list[:,0]
    atom_shift_list = tf.unstack(atom_shift_list)

    modified_struct = atom_position_list(*atom_shift_list)

    hkl_list = transform_list_hkl_p63_p65(hkl_list)

    sf_hkl = get_structure_factors(hkl_list, modified_struct)
    intensity = (abs(sf_hkl)) ** 2
    w = tf.constant(0.0004877332589381476, dtype=tf.float32)  # Debye-Waller factor 
    qnorms = tf.norm(tf.cast(hkl_list, tf.float32), axis=1)
    intensity = intensity * tf.exp(- w* qnorms ** 2)  # Apply Debye-Waller factor
    max_intensity = tf.reduce_max(intensity)
    intensity = tf.where(intensity < max_intensity * 0.05, tf.zeros_like(intensity), intensity)  # remove intensities below a threshold
    intensity = intensity / tf.reduce_max(intensity)  # Normalize to max
    intensity = intensity / tf.reduce_sum(intensity) * 60
    return intensity


class FunAsLayer(tf.keras.layers.Layer):
    def __init__(self, matrix , max_mode_amps,**kwargs):
        super().__init__(**kwargs)
        self.max_mode_amps = max_mode_amps
        self.matrix = matrix

    def build(self, input_shape):
        self.param = self.add_weight(name='param', shape=(156,), initializer=tf.keras.initializers.RandomNormal(mean=0.,stddev=0.2), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        # Apply tanh to ensure parameters stay within the [-1, 1] range then multiply by max_mode_amps so each parameter is scaled corresponding to the element in max_mode_amps
        pretransform = tf.tanh(self.param)
        transformed_params = pretransform * self.max_mode_amps  # Scale parameters

        output = fun_tf(inputs, (transformed_params) , self.matrix)
        return tf.reshape(output , [-1])  # Ensure output is 1D
    

"""
# R-Score based on intensity
class RFactorLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_sum(tf.abs(y_true - y_pred)) / tf.reduce_sum(y_true)
"""
    
# mean squared error
class PerSampleMSE(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__(reduction=tf.keras.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        squared_error = tf.square(y_true - y_pred)
        per_sample_mse = tf.reduce_mean(squared_error, axis=-1)
        return per_sample_mse  # shape (batch_size,)
  
# Define the custom metric function
def r_factor_metric(y_true, y_pred):
    labels = y_true
    return tf.reduce_sum(tf.abs(labels - y_pred)) / tf.reduce_sum(labels)


def make_sample_weights(experimental_data):
    labels = experimental_data["intensity_exp"].tolist()
    labels = labels / np.max(labels)  # Normalize labels
    labels = labels / np.sum(labels) * 60 #Normalize labels
    vol_err = experimental_data["intensity_exp_err"].tolist()

    labels_err = []

    for label, err in zip(labels, vol_err):
        if label == 0:
            labels_err.append(1)  # Assign a high error for zero labels
        else:
            labels_err.append(1000)  # Inverse error for each label

    labels_err = tf.convert_to_tensor(labels_err, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    labels = tf.expand_dims(labels, axis=-1)  # Ensure labels are 2D
    labels_err = tf.expand_dims(labels_err, axis=-1)  # Ensure labels_err are 2D

    return labels, labels_err



if __name__ == "__main__":
    t0 = time()
    # Load experimental data
    # experimental_data = pd.read_csv('alrisDistortionFit/PBCO/raw_data/combined_peaks.csv')
    # matrix = np.loadtxt('alrisDistortionFit/PBCO/matrix.txt', dtype=np.float32)
    # max_mode_amps = np.loadtxt('alrisDistortionFit/PBCO/new_PBCO_fit/new_PBCO_max_bound_vectors.txt', dtype=np.float32 , delimiter=',')

    experimental_data = pd.read_csv('1_3_LOGcombined_peaks.csv')
    matrix = np.loadtxt('1_3_matrix.txt', dtype=np.float32)
    max_mode_amps = np.loadtxt('PBCO_1_3_max_bound_vectors.txt', dtype=np.float32 , delimiter=',')

    number_of_modes = 156
    n_features = experimental_data.shape[0]
    n_dim = 3
    iteration_num = 2000
    seed = 1
    n_cores = 32
    epochs = 75
    lr = 0.08
    hkl_list = experimental_data[["h", "k", "l"]].values.tolist()

    features = tf.convert_to_tensor(hkl_list, dtype=tf.float32)
    matrix = tf.convert_to_tensor(matrix, dtype=tf.float32)
    max_mode_amps = tf.convert_to_tensor(max_mode_amps, dtype=tf.float32)

    labels, labels_err = make_sample_weights(experimental_data)

    # Instantiate multiprocessing pool

    mp.set_start_method('spawn', force=True)  # Use 'spawn' to avoid issues with TensorFlow and multiprocessing

    pool = mp.Pool(
        processes=n_cores,
        initializer=init_worker,
        initargs=(seed,features, labels, labels_err, matrix, max_mode_amps , epochs , lr)
    )
    #spawn n processes

    # Start the evaluation
    results = []
    progress_interval = max(1, iteration_num // 10)
    for idx, result in enumerate(pool.imap_unordered(run_iteration, range(iteration_num), 1)):
        results.append(result)
        if idx % progress_interval == 0 or idx == iteration_num -1:
            print(f"Progress: {(idx/iteration_num*100):.0f}% completed.")
    
    # Close the pool
    pool.close()
    pool.join()

    histogram_matrix = np.zeros((number_of_modes, iteration_num), dtype=np.float32)
    loss_matrix = np.zeros((iteration_num,), dtype=np.float32)
    r_factors = np.zeros((iteration_num,), dtype=np.float32)
    each_iteration_loss = np.zeros((iteration_num,epochs), dtype=np.float32)

    for i, res in enumerate(results):
        histogram_matrix[: , i] = res[0]
        loss_matrix[i] = res[1]
        each_iteration_loss[i] = res[3]
        r_factors[i] = res[4]

    savedir = f'results/LOG{datetime.now().strftime("%Y%m%d_%H%M%S")}_iters{iteration_num}_epochs{epochs}_lr{lr}'
    os.makedirs(savedir, exist_ok=True)  # Ensure the directory exists
    np.savez(os.path.join(savedir, 'all_result_matrix.npz'), histogram_matrix=histogram_matrix , loss_matrix=loss_matrix , each_iteration_loss=each_iteration_loss, r_factors=r_factors)

    print(f"Total time taken: {time() - t0:.2f} seconds")