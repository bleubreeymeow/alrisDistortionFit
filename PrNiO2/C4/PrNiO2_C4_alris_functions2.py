import numpy as np
import tensorflow as tf


def transform_list_hkl_p63_p65(hkl_list):
    """
    Function to transform a list of hkl vectors from the old cell to the new cell, in the P63 structure as well as
    in the p65 structure.
    :param hkl_list: List of hkl vectors
    :return: List of hkl vectors in the new cell
    -9a,c,9b
    """
    # Convert hkl_list to a TensorFlow tensor
    hkl_list = tf.convert_to_tensor(hkl_list, dtype=tf.float32)

    # Apply the transformation using TensorFlow operations
    h_new = 2 * hkl_list[:, 2]
    k_new = - 2 * hkl_list[:, 0] - 2 * hkl_list[:, 1]
    l_new = 2 * hkl_list[:, 0] - 2 * hkl_list[:, 1]

    # Stack the new h, k, l components into a single tensor
    result = tf.stack([h_new, k_new, l_new], axis=1)

    return result


def get_atomic_form_factor(qnorm, atom):
    """
    Function to calculate the atomic form factor for a specific atom. Values for the Gaussian's are from
    International Tables for Crystallography, Vol. C, 2006.
    :param qnorm: Norm of the hkl vector |Q|
    :param atom: Type of atom (atm only Pr, Ni or O is possible)
    :return: The atomic form factor
    """
    # Define values for Pr, Ni, O atoms as TensorFlow constants
    Pr_vals = {
        'a': tf.constant([21.3727, 19.7491, 12.1329, 0.97578], dtype=tf.float32),
        'b': tf.constant([2.64520, 0.214299, 15.323, 36.4065], dtype=tf.float32),
        'c': tf.constant(1.77132, dtype=tf.float32),
    }
    Ni_vals = {
        'a': tf.constant([12.1271, 7.34625, 4.8940, 1.67865], dtype=tf.float32),
        'b': tf.constant([3.77755, 0.25070000000000003, 10.52465, 44.25235], dtype=tf.float32),
        'c': tf.constant(0.94775, dtype=tf.float32),
    }
    O_vals = {
        'a': tf.constant([3.7504, 2.84294, 1.54298, 1.652091], dtype=tf.float32),
        'b': tf.constant([16.5151, 6.59203, 0.319201, 42.3486], dtype=tf.float32),
        'c': tf.constant(0.24206, dtype=tf.float32),
    }

    # Choose atom values based on the input atom
    if atom == "Pr":
        vals_dict = Pr_vals
    elif atom == "Ni":
        vals_dict = Ni_vals
    else:
        vals_dict = O_vals

    # Start with the constant "c" term
    fq = vals_dict["c"]

    # Use element-wise operations instead of a loop
    a_vals = vals_dict["a"]
    b_vals = vals_dict["b"]

    # Compute the exponential terms
    exponential_terms = tf.exp(-b_vals * (qnorm / (4 * tf.constant(np.pi))) ** 2)
    # Multiply the "a" values with the corresponding exponential terms and sum them
    fq += tf.reduce_sum(a_vals * exponential_terms)

    return fq


def get_structure_factors(hkl_batch, structure):
    """
    Vectorized structure factor calculation.

    Parameters
    ----------
    hkl_batch : Tensor [N, 3]
        List of N hkl vectors
    structure : List of (atom, occupancy, position)
        Atomic basis of the crystal

    Returns
    -------
    Tensor [N] (complex64)
        Structure factors for each hkl
    """
    # Get atomic types and positions
    atoms = [a for a, _, _ in structure]
    positions = tf.stack([tf.convert_to_tensor(p, dtype=tf.float32) for _, _, p in structure])  # [A, 3]

    # Compute qnorms for each hkl vector (shape [N])
    qnorms = tf.norm(tf.cast(hkl_batch, tf.float32), axis=1)  # [N]
    # w = tf.constant(0.01, dtype=tf.float32)  # Debye-Waller factor old is 0.00159

    # Get per-atom form factors per hkl
    fq_table = {
        "Ni": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Ni"), tf.complex64), qnorms),
        "Pr": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Pr"), tf.complex64), qnorms),
        "O": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "O"), tf.complex64), qnorms)
    }  # Each: [N]

    # Build full form factor matrix [N, A]
    fq_matrix = tf.stack([fq_table[atom] for atom in atoms], axis=1)  # shape [N, A]

    # Compute phase terms: [N, A]
    phase_arg = tf.tensordot(tf.cast(hkl_batch, tf.float32), tf.transpose(positions), axes=1)  # [N, A]
    phase = tf.exp(tf.complex(0.0, -2.0 * np.pi) * tf.cast(phase_arg, tf.complex64))  # [N, A]

    # Element-wise multiply and sum over atoms
    F_hkl = tf.reduce_sum(fq_matrix * phase, axis=1)  # [N]
    # Apply Debye-Waller factor
    # F_hkl = tf.cast(F_hkl, tf.complex64) * tf.cast(tf.exp(-w * qnorms ** 2), tf.complex64)  # [N]
    return F_hkl

def shift_atoms(a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14):
    res = [
        [+  0.07620*a2],
        [+  0.03188*a3 +  0.03188*a4],
        [+  0.07620*a1],
        [-  0.03188*a3 +  0.03188*a4],
        [-  0.01594*a5 +  0.01594*a6 -  0.01594*a7 -  0.01594*a8 -  0.01594*a9 -  0.01594*a10 -  0.01594*a11 -  0.01594*a12],
        [+  0.01594*a5 -  0.01594*a6 -  0.01594*a7 -  0.01594*a8 -  0.01594*a9 +  0.01594*a10 -  0.01594*a11 +  0.01594*a12],
        [-  0.01594*a5 -  0.01594*a6 -  0.01594*a7 +  0.01594*a8 -  0.01594*a9 -  0.01594*a10 +  0.01594*a11 +  0.01594*a12],
        [+  0.01594*a5 +  0.01594*a6 -  0.01594*a7 +  0.01594*a8 -  0.01594*a9 +  0.01594*a10 +  0.01594*a11 -  0.01594*a12],
        [+  0.01594*a5 +  0.01594*a6 +  0.01594*a7 -  0.01594*a8 -  0.01594*a9 -  0.01594*a10 +  0.01594*a11 +  0.01594*a12],
        [-  0.01594*a5 -  0.01594*a6 +  0.01594*a7 -  0.01594*a8 -  0.01594*a9 +  0.01594*a10 +  0.01594*a11 -  0.01594*a12],
        [+  0.01594*a5 -  0.01594*a6 +  0.01594*a7 +  0.01594*a8 -  0.01594*a9 -  0.01594*a10 -  0.01594*a11 -  0.01594*a12],
        [-  0.01594*a5 +  0.01594*a6 +  0.01594*a7 +  0.01594*a8 -  0.01594*a9 +  0.01594*a10 -  0.01594*a11 +  0.01594*a12],
        [-  0.04509*a13],
        [-  0.04509*a14]

    ]

    return res


def atom_position_list(Pr1_1_dx, Pr1_1_dy, Pr1_2_dx, Pr1_2_dz, O1_1_dy, O1_1_dz, O1_2_dy, O1_2_dz, O1_3_dy, O1_3_dz, O1_4_dy, O1_4_dz, Ni1_3_dy, Ni1_3_dz):
    
    res = [
        ['Pr','59',[0.75 + Pr1_1_dx,0.25 + Pr1_1_dy,0.0]],
        ['Pr','59',[0.25 + Pr1_1_dx,0.75 + Pr1_1_dy,0.5]],
        ['Pr','59',[0.25 + Pr1_1_dx,0.75 + Pr1_1_dy,0.0]],
        ['Pr','59',[0.75 + Pr1_1_dx,0.25 + Pr1_1_dy,0.5]],
        ['Pr','59',[0.25 + Pr1_1_dx,0.25 + Pr1_1_dy,0.0]],
        ['Pr','59',[0.75 + Pr1_1_dx,0.75 + Pr1_1_dy,0.5]],
        ['Pr','59',[0.75 + Pr1_1_dx,0.75 + Pr1_1_dy,0.0]],
        ['Pr','59',[0.25 + Pr1_1_dx,0.25 + Pr1_1_dy,0.5]],
        ['Pr','59',[0.75 + Pr1_2_dx,0.0,0.75 + Pr1_2_dz]],
        ['Pr','59',[0.25 + Pr1_2_dx,0.5,0.25 + Pr1_2_dz]],
        ['Pr','59',[0.25 + Pr1_2_dx,0.0,0.75 + Pr1_2_dz]],
        ['Pr','59',[0.75 + Pr1_2_dx,0.5,0.25 + Pr1_2_dz]],
        ['Pr','59',[0.25 + Pr1_2_dx,0.0,0.25 + Pr1_2_dz]],
        ['Pr','59',[0.75 + Pr1_2_dx,0.5,0.75 + Pr1_2_dz]],
        ['Pr','59',[0.75 + Pr1_2_dx,0.0,0.25 + Pr1_2_dz]],
        ['Pr','59',[0.25 + Pr1_2_dx,0.5,0.75 + Pr1_2_dz]],
        ['O','8',[0.0,0.875 + O1_1_dy,0.875 + O1_1_dz]],
        ['O','8',[0.5,0.375 + O1_1_dy,0.375 + O1_1_dz]],
        ['O','8',[0.0,0.125 + O1_1_dy,0.875 + O1_1_dz]],
        ['O','8',[0.5,0.625 + O1_1_dy,0.375 + O1_1_dz]],
        ['O','8',[0.0,0.875 + O1_1_dy,0.125 + O1_1_dz]],
        ['O','8',[0.5,0.375 + O1_1_dy,0.625 + O1_1_dz]],
        ['O','8',[0.0,0.125 + O1_1_dy,0.125 + O1_1_dz]],
        ['O','8',[0.5,0.625 + O1_1_dy,0.625 + O1_1_dz]],
        ['O','8',[0.0,0.875 + O1_2_dy,0.375 + O1_2_dz]],
        ['O','8',[0.5,0.375 + O1_2_dy,0.875 + O1_2_dz]],
        ['O','8',[0.0,0.125 + O1_2_dy,0.375 + O1_2_dz]],
        ['O','8',[0.5,0.625 + O1_2_dy,0.875 + O1_2_dz]],
        ['O','8',[0.0,0.875 + O1_2_dy,0.625 + O1_2_dz]],
        ['O','8',[0.5,0.375 + O1_2_dy,0.125 + O1_2_dz]],
        ['O','8',[0.0,0.125 + O1_2_dy,0.625 + O1_2_dz]],
        ['O','8',[0.5,0.625 + O1_2_dy,0.125 + O1_2_dz]],
        ['O','8',[0.0,0.375 + O1_3_dy,0.875 + O1_3_dz]],
        ['O','8',[0.5,0.875 + O1_3_dy,0.375 + O1_3_dz]],
        ['O','8',[0.0,0.625 + O1_3_dy,0.875 + O1_3_dz]],
        ['O','8',[0.5,0.125 + O1_3_dy,0.375 + O1_3_dz]],
        ['O','8',[0.0,0.375 + O1_3_dy,0.125 + O1_3_dz]],
        ['O','8',[0.5,0.875 + O1_3_dy,0.625 + O1_3_dz]],
        ['O','8',[0.0,0.625 + O1_3_dy,0.125 + O1_3_dz]],
        ['O','8',[0.5,0.125 + O1_3_dy,0.625 + O1_3_dz]],
        ['O','8',[0.0,0.375 + O1_4_dy,0.375 + O1_4_dz]],
        ['O','8',[0.5,0.875 + O1_4_dy,0.875 + O1_4_dz]],
        ['O','8',[0.0,0.625 + O1_4_dy,0.375 + O1_4_dz]],
        ['O','8',[0.5,0.125 + O1_4_dy,0.875 + O1_4_dz]],
        ['O','8',[0.0,0.375 + O1_4_dy,0.625 + O1_4_dz]],
        ['O','8',[0.5,0.875 + O1_4_dy,0.125 + O1_4_dz]],
        ['O','8',[0.0,0.625 + O1_4_dy,0.625 + O1_4_dz]],
        ['O','8',[0.5,0.125 + O1_4_dy,0.125 + O1_4_dz]],
        ['Ni','28',[0.0,0.0,0.0]],
        ['Ni','28',[0.5,0.5,0.5]],
        ['Ni','28',[0.5,0.5,0.0]],
        ['Ni','28',[0.0,0.0,0.5]],
        ['Ni','28',[0.0,0.75 + Ni1_3_dy,0.75 + Ni1_3_dz]],
        ['Ni','28',[0.5,0.25 + Ni1_3_dy,0.25 + Ni1_3_dz]],
        ['Ni','28',[0.0,0.25 + Ni1_3_dy,0.75 + Ni1_3_dz]],
        ['Ni','28',[0.5,0.75 + Ni1_3_dy,0.25 + Ni1_3_dz]],
        ['Ni','28',[0.0,0.75 + Ni1_3_dy,0.25 + Ni1_3_dz]],
        ['Ni','28',[0.5,0.25 + Ni1_3_dy,0.75 + Ni1_3_dz]],
        ['Ni','28',[0.0,0.25 + Ni1_3_dy,0.25 + Ni1_3_dz]],
        ['Ni','28',[0.5,0.75 + Ni1_3_dy,0.75 + Ni1_3_dz]],
        ['Ni','28',[0.5,0.0,0.5]],
        ['Ni','28',[0.0,0.5,0.0]],
        ['Ni','28',[0.0,0.5,0.5]],
        ['Ni','28',[0.5,0.0,0.0]]



    ]

    return res