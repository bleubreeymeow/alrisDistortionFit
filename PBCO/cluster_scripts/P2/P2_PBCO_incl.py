import os
os.environ['PYTHONHASHSEED'] = '1'  # Set a fixed seed for reproducibility
slurm_array_task_id=int(os.environ["SLURM_ARRAY_TASK_ID"])
# os.environ['TF_NUM_INTRAOP_THREADS'] = '1'  # For within-op parallelism
# os.environ['TF_NUM_INTEROP_THREADS'] = '1'  # For between-op parallelism

import numpy as np
import tensorflow as tf
import pandas as pd
from P2_alris_one_third_functions import shift_atoms, transform_list_hkl_p63_p65, get_structure_factors , atom_position_list
import multiprocessing as mp
from time import time
from datetime import datetime

num_threads = 1 

# Configure TensorFlow to use multiple threads
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)


global features ,labels , labels_err, matrix , max_mode_amps , epochs , lr , hkl_indices

def init_worker(seed, _features, _labels, _labels_err, _matrix, _max_mode_amps , _epochs , _lr , _hkl_indices):
    global features, labels, labels_err, matrix, max_mode_amps, epochs, lr , hkl_indices
    features = _features
    labels = _labels
    labels_err = _labels_err
    matrix = _matrix
    max_mode_amps = _max_mode_amps
    epochs = _epochs
    lr = _lr
    hkl_indices = _hkl_indices

    # Create separate seed
    worker_id = mp.current_process()._identity[0] if mp.current_process()._identity else 0
    worker_seed = seed + worker_id
    tf.keras.utils.set_random_seed(seed=worker_seed)

    print(f"worker {worker_id} initialised")



def run_iteration(iteration):
    global features, labels, labels_err, matrix, max_mode_amps , epochs , lr , hkl_indices
    n_dim = 3

    optim = tf.keras.optimizers.Adam(learning_rate=lr)
    n_epochs = epochs

 # Create the model
    inputs = tf.keras.Input(shape=(n_dim,))
    outputs = FunAsLayer(matrix , max_mode_amps , hkl_indices)(inputs)
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
        shuffle=False, # not sure whether this matters
        # callbacks=[cb]
        sample_weight=labels_err  # Use sample weights if you have errors
    )

    final_loss = history.history['loss'][-1]
    best_model_pars = max_mode_amps * tf.tanh(model.layers[-1].get_weights()[0])
    y_pred = fun_tf(features, best_model_pars , matrix , hkl_indices)
    labels = tf.reshape(labels, [-1])
    rf = r_factor_metric(labels, y_pred)

    return best_model_pars, final_loss , iteration , history.history['loss'] , rf

def fun_tf(hkl_list, pars, matrix , hkl_indices):
    """
    Fast computation of structure factors with parameter-dependent structure.
    """
    # stack parameters
    pars_tensor = tf.stack(pars)  # shape (params,)


    norm_factors = [0.05038, 0.04953, 0.03502, 0.03562, 0.04035, 0.03502, 0.03562, 0.02017, 0.02853, 0.02477, 0.02853, 0.02519, 0.05038, 0.04953, 0.03502, 0.03562, 0.04035, 0.03502, 0.03562, 0.02017, 0.02853, 0.02477, 0.02853, 0.02519, 0.05038, 0.04953, 0.03502, 0.03562, 0.04035, 0.03502, 0.03562, 0.02017, 0.02853, 0.02477, 0.02853, 0.02519, 0.04035, 0.03502, 0.03562, 0.02017, 0.02853, 0.02477, 0.02853, 0.02519, 0.04035, 0.03502, 0.03562, 0.02017, 0.02853, 0.02477, 0.02853, 0.02519]
    norm_factors = tf.convert_to_tensor(norm_factors, dtype=tf.float32)
    pars_tensor = pars_tensor * norm_factors
    
    pars_tensor = tf.reshape(pars_tensor, (52,))  # Ensure shape is (params,)
    pars_tensor = tf.unstack(pars_tensor)

    # atom_shift_list = shift_atoms(matrix , (pars_tensor))
    # atom_shift_list = atom_shift_list[:,0]
    # atom_shift_list = tf.unstack(atom_shift_list)

    # modified_struct = atom_position_list(*atom_shift_list)

    modified_struct = atom_position_list( *pars_tensor )
    all_hkl_list = fn_all_hkl_list()
    all_hkl_list = transform_list_hkl_p63_p65(all_hkl_list)

    # Get structure factors
    sf_hkl = get_structure_factors(all_hkl_list, modified_struct)
    intensity = (abs(sf_hkl)) ** 2

    w = tf.constant(0.000811509257682975, dtype=tf.float32)  # Debye-Waller factor 
    qnorms = tf.norm(tf.cast(all_hkl_list, tf.float32), axis=1)
    intensity = intensity * tf.exp(- w * qnorms ** 2)  # Apply Debye-Waller factor

    all_sim_intensity = intensity
    all_sim_intensity = all_sim_intensity / tf.reduce_sum(all_sim_intensity) * 10000

    # Extract only the intensities corresponding to the input hkl_list
    sim_intensity = tf.gather(all_sim_intensity, hkl_indices)

    return sim_intensity

class FunAsLayer(tf.keras.layers.Layer):
    def __init__(self, matrix , max_mode_amps,hkl_indices,**kwargs):
        super().__init__(**kwargs)
        self.max_mode_amps = max_mode_amps
        self.matrix = matrix
        self.hkl_indices = hkl_indices

    def build(self, input_shape):
        max_mode_amps = self.max_mode_amps
        self.param = (self.add_weight(name='param', shape=(52,), initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.06), trainable=True, constraint=lambda x: tf.clip_by_value(x, -max_mode_amps, max_mode_amps)))
        super().build(input_shape)

    def call(self, inputs):
        output = fun_tf(inputs, self.param , self.matrix, self.hkl_indices)
        return tf.reshape(output , [-1])  # Ensure output is 1D
    

"""
# R-Score based on intensity
class RFactorLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_sum(tf.abs(y_true - y_pred)) / tf.reduce_sum(y_true)
"""
    
     
# Define the custom metric function
def r_factor_metric(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    return tf.reduce_sum(tf.abs(y_true - y_pred)) / tf.reduce_sum(y_true)


def make_sample_weights(experimental_data):
    labels = experimental_data["intensity_exp"].tolist()
    labels = labels / np.float32(275455.80456422985) * 10000  # Normalize to sum to 10000 
    vol_err = experimental_data["intensity_exp_err"].tolist()

    labels_err = []

    for label, err in zip(labels, vol_err):
        if label == 0:
            labels_err.append(1e-9)  # Assign a high error for zero labels
        else:
            labels_err.append(1/label) # Inverse error for each label

    labels_err = tf.convert_to_tensor(labels_err, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels, dtype=tf.float32)

    labels = tf.expand_dims(labels, axis=-1)  # Ensure labels are 2D
    labels_err = tf.expand_dims(labels_err, axis=-1)  # Ensure labels_err are 2D

    return labels, labels_err

def fn_all_hkl_list():
    all_h_list = np.linspace(0 , -4.666666 , 15)
    all_k_list = np.linspace(0 , 4.666666 , 15) 
    all_l_list = np.linspace(0 , 12 , 5) 
    all_hkl_list = np.array(np.meshgrid(all_h_list, all_k_list, all_l_list)).T.reshape(-1, 3)
    all_hkl_list = tf.convert_to_tensor(all_hkl_list, dtype=tf.float32)
    return all_hkl_list


def make_hkl_indices(hkl_list , experimental_data): 
    hkl_list = np.array(hkl_list, dtype=np.float32) * 3
    hkl_list = np.round(hkl_list).astype(int)
    hkl_list = pd.DataFrame(hkl_list, columns=['h', 'k', 'l'])

    all_h_list = np.linspace(0 , -4.666666 , 15) * 3
    all_k_list = np.linspace(0 , 4.666666 , 15) * 3
    all_l_list = np.linspace(0 , 12 , 5) * 3
    all_hkl_list = np.array(np.meshgrid(all_h_list, all_k_list, all_l_list)).T.reshape(-1, 3)
    all_hkl_list = np.round(all_hkl_list).astype(int)
    all_hkl_list = pd.DataFrame(all_hkl_list, columns=['h', 'k', 'l'])

    merged = all_hkl_list.merge(hkl_list, how='left', indicator=True)
    hkl_indices = merged[merged['_merge'] == 'both'].index.tolist()

    #retrive hkl values from all_hkl_list using hkl_indices
    hkl_from_indices = all_hkl_list.iloc[hkl_indices].values.tolist()

    reordered_hkl_list = pd.DataFrame(hkl_from_indices, columns=['h', 'k', 'l'])


    experimental_data[['h', 'k', 'l']] = experimental_data[['h', 'k', 'l']] * 3 
    experimental_data[['h', 'k', 'l']] = np.round(experimental_data[['h', 'k', 'l']]).astype(int)

    #find the intensity values from experimental_data corresponding to the reordered_hkl_list
    intensity = []
    for index, row in reordered_hkl_list.iterrows():
        h = row['h']
        k = row['k']
        l = row['l']
        intensity_value = experimental_data[(experimental_data['h'] == h) & (experimental_data['k'] == k) & (experimental_data['l'] == l)]['intensity_exp'].values
        
        if len(intensity_value) > 0:
            intensity.append(intensity_value[0])
        else:
            intensity.append(0)

    reordered_hkl_list = np.array(hkl_from_indices, dtype=np.float32) / 3

    hkl_indices = tf.convert_to_tensor(hkl_indices, dtype=tf.int32)

    return hkl_indices , reordered_hkl_list , intensity


if __name__ == "__main__":
    t0 = time()

    pre_experimental_data = pd.read_csv('1_3_combined_peaks_300K_no_bragg.csv')
    matrix = np.loadtxt('P2_matrix.txt', dtype=np.float32)

    hkl_list = pre_experimental_data[["h", "k", "l"]].values.tolist()
    hkl_indices , hkl_list , intensity_reordered = make_hkl_indices(hkl_list , pre_experimental_data)
    experimental_data = pd.DataFrame(hkl_list , columns=['h', 'k', 'l'])
    experimental_data['intensity_exp'] = intensity_reordered
    experimental_data['intensity_exp_err'] = pre_experimental_data['intensity_exp_err'].values

    max_mode_amps = np.loadtxt('PBCO_1_3_P2_max_bound_vectors.txt', dtype=np.float32 , delimiter=',')
    max_mode_amps2 = max_mode_amps.copy()
    indices_to_zero = [3 , 4 , 9 , 10 , 11 , 12 , 15 , 21 , 22 , 23 , 24]
    for index in indices_to_zero:
        max_mode_amps[index - 1] = 0.0
    this_included_mode = indices_to_zero[slurm_array_task_id]

    #recover the max amp of the mode which is included.
    max_mode_amps[this_included_mode - 1] = max_mode_amps2[this_included_mode - 1]

    number_of_modes = 52
    n_features = experimental_data.shape[0]
    n_dim = 3
    iteration_num = 500
    seed = 1
    n_cores = 32
    epochs = 1000
    lr = 7e-3
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
        initargs=(seed,features, labels, labels_err, matrix, max_mode_amps , epochs , lr , hkl_indices)
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

    savedir = f'results/run_1_inc_{this_included_mode}_iters{iteration_num}_epochs{epochs}_lr{lr}'
    os.makedirs(savedir, exist_ok=True)  # Ensure the directory exists
    np.savez(os.path.join(savedir, 'all_result_matrix.npz'), histogram_matrix=histogram_matrix , loss_matrix=loss_matrix , each_iteration_loss=each_iteration_loss, r_factors=r_factors)

    print(f"Total time taken: {time() - t0:.2f} seconds")