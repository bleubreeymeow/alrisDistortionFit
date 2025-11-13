import numpy as np
import tensorflow as tf

def transform_list_hkl_p63_p65(hkl_list):
    """
    Function to transform a list of hkl vectors from original (r.l.u) units back to ang**-1 units
    :param hkl_list: List of hkl vectors
    :return: List of hkl vectors in units of ang**-1

    """
    # Convert hkl_list to a TensorFlow tensor
    # a = 3.82030
    # b = 3.88548
    # c = 11.68350
    # hkl_list = tf.convert_to_tensor(hkl_list, dtype=tf.float32)

    # h_new = hkl_list[:, 0] * 2 * np.pi / a
    # k_new = hkl_list[:, 1] * 2 * np.pi / b
    # l_new = hkl_list[:, 2] * 2 * np.pi / c

    hkl_list = tf.convert_to_tensor(hkl_list, dtype=tf.float32)

    h_new = -3 * hkl_list[:, 0]
    k_new = hkl_list[: , 2]
    l_new = 3 * hkl_list[:, 1]

    return tf.stack([h_new, k_new, l_new], axis=1)


def fractional_coords(positions):
    '''
    Function to change the coordinates from its fractional form w.r.t the superlattice unit
    cell back with units of ang.
    -3a, 3b , c transformation was done for P2
    '''
    a = 3.82030
    b = 3.88548
    c = 11.68350

    x_new = 3 * a * positions[:, 0] 
    y_new = 3 * b * positions[:, 1]
    z_new = c * positions[:, 2]

    fractional_positions = tf.stack([x_new, y_new, z_new], axis=1)

    return fractional_positions


def get_atomic_form_factor(qnorm, atom):
    """
    Function to calculate the atomic form factor for a specific atom. Values for the Gaussian's are from
    International Tables for Crystallography, Vol. C, 2006.

    https://lampz.tugraz.at/~hadley/ss1/crystaldiffraction/atomicformfactors/formfactors.php#:~:text=Each%20diffraction%20peak%20corresponds%20to,intensity%20of%20the%20diffraction%20peak.
    
    :param qnorm: Norm of the hkl vector |Q|
    :param atom: Type of atom (atm only Pr, Ni or O is possible)
    :return: The atomic form factor

    Oxidation states:
    Pr: 4+
    Ba: 2+
    Cu: 2+
    O: 2+

    this one is used

    value of O is used becuase there is no O2-

    or

    Pr: 1+
    Ba: 2+
    Cu: 3+
    O: 2-


    """
    # Define values for Pr, Ni, O atoms as TensorFlow constants
    Pr_vals = {
        'a': tf.constant([20.9413, 20.0539, 12.4668, 0.296689], dtype=tf.float32),
        'b': tf.constant([2.54467, 0.202481, 14.8137, 45.4643], dtype=tf.float32),
        'c': tf.constant(1.24285, dtype=tf.float32),
    }
    Ba_vals = {
        'a': tf.constant([20.1807, 19.1136, 10.9054, 0.77634], dtype=tf.float32),
        'b': tf.constant([3.21367, 0.28331, 20.0558, 51.746], dtype=tf.float32),
        'c': tf.constant(3.02902, dtype=tf.float32),
    }
    Cu_vals = {
        'a': tf.constant([11.8168, 7.11181, 5.78135, 1.14523], dtype=tf.float32),
        'b': tf.constant([3.37484, 0.244078, 7.9876, 19.897], dtype=tf.float32),
        'c': tf.constant(1.14431, dtype=tf.float32),
    }
    O_vals = {
        'a': tf.constant([3.7504, 2.84294, 1.54298, 1.652091], dtype=tf.float32),
        'b': tf.constant([16.5151, 6.59203, 0.319201, 43.3486], dtype=tf.float32),
        'c': tf.constant(0.24206, dtype=tf.float32),
    }

    # Choose atom values based on the input atom
    if atom == "Pr":
        vals_dict = Pr_vals
    elif atom == "Ba":
        vals_dict = Ba_vals
    elif atom == "Cu":
        vals_dict = Cu_vals
    else:
        vals_dict = O_vals

    # Start with the constant "c" term
    fq = vals_dict["c"]

    # Use element-wise operations instead of a loop
    a_vals = vals_dict["a"]
    b_vals = vals_dict["b"]

    # Compute the exponential terms
    exponential_terms = tf.exp(-b_vals * (qnorm / (4 * tf.constant(np.pi, dtype=tf.float32))) ** 2)

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

    https://advanced.onlinelibrary.wiley.com/doi/10.1002/aenm.202300760

    """

    # Get atomic types and positions
    atoms = [a for a, _, _ in structure]
    positions = tf.stack([tf.convert_to_tensor(p, dtype=tf.float32) for _, _, p in structure])  # [A, 3]
    # positions = fractional_coords(positions)  # Convert to angstrom

    # Compute qnorms for each hkl vector (shape [N])
    qnorms = tf.norm(tf.cast(hkl_batch, tf.float32), axis=1)  # [N]
    # w = tf.constant(0.01, dtype=tf.float32)  # Debye-Waller factor old is 0.00159

    # Get per-atom form factors per hkl
    fq_table = {
        "Pr": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Pr"), tf.complex64), qnorms),
        "Ba": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Ba"), tf.complex64), qnorms),
        "Cu": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "Cu"), tf.complex64), qnorms),
        "O": tf.vectorized_map(lambda q: tf.cast(get_atomic_form_factor(q, "O"), tf.complex64), qnorms)
    }  # Each: [N]

    # Build full form factor matrix [N, A]
    fq_matrix = tf.stack([fq_table[atom] for atom in atoms], axis=1)  # shape [N, A]

    # Compute phase terms: [N, A]
    phase_arg = tf.tensordot(tf.cast(hkl_batch, tf.float32), tf.transpose(positions), axes=1)  # [N, A]
    phase_arg = tf.cast(phase_arg, tf.float32)  # Ensure float32 type for complex conversion
    phase = tf.exp(tf.complex(0.0,-2 * np.pi * phase_arg))  # [N, A]

    # Element-wise multiply and sum over atoms
    F_hkl = tf.reduce_sum(fq_matrix * phase, axis=1)  # [N]

    return F_hkl

def shift_atoms(matrix , a):   
    '''
        matrix multiplication of mode amplitudes and its respective weights

        Parameters
        ---------
        matrix : [N , certain value]
            loaded from a txt file, values which gives the linear combination of modes its weights
        a : Tensor [N]
            mode magnitudes 

        Returns
        ---------
        Tensor [N]
    
    ''' 
    return matrix @ tf.reshape(a , [-1 , 1])


def atom_position_list( M1 ,  M2 ,  M27 ,  M40 ,  M3 ,  M4 ,  M5 ,  M22 ,  M28 ,  M29 ,  M41 ,  M42 ,  M6 ,  M7 ,  M30 ,  M43 ,  M8 ,  M9 ,  M10 ,  M23 ,  M31 ,  M32 ,  M44 ,  M45 ,  M11 ,  M12 ,  M33 ,  M46 ,  M13 ,  M14 ,  M15 ,  M24 ,  M34 ,  M35 ,  M47 ,  M48 ,  M16 ,  M17 ,  M18 ,  M25 ,  M36 ,  M37 ,  M49 ,  M50 ,  M19 ,  M20 ,  M21 ,  M26 ,  M38 ,  M39 ,  M51 ,  M52 ):

    res = [
        ['Pr', 59, [0.83333 - 0.5*M1 - M40, 0.5, 0.16667 + 0.5*M2 + M27]],
        ['Pr', 59, [0.83333 - 0.5*M1 - M40, 0.5, 0.83333 - 0.5*M2 - M27]],
        ['Pr', 59, [0.16667 + 0.5*M1 + M40, 0.5, 0.16667 + 0.5*M2 + M27]],
        ['Pr', 59, [0.16667 + 0.5*M1 + M40, 0.5, 0.83333 - 0.5*M2 - M27]],
        ['Pr', 59, [0.83333 + M1 - M40, 0.5, 0.5]],
        ['Pr', 59, [0.16667 - M1 + M40, 0.5, 0.5]],
        ['Pr', 59, [0.5, 0.5, 0.16667 - M2 + M27]],
        ['Pr', 59, [0.5, 0.5, 0.83333 + M2 - M27]],
        ['Pr', 59, [0.5, 0.5, 0.5]],

        ['Ba', 56, [0.83333 - 0.5*M5 - M42, 0.18393 - 0.25*M3 + M22 + 0.5*M28 + 0.5*M41, 0.16667 + 0.5*M4 + M29]],
        ['Ba', 56, [0.83333 - 0.5*M5 - M42, 0.18393 - 0.25*M3 + M22 + 0.5*M28 + 0.5*M41, 0.83333 - 0.5*M4 - M29]],
        ['Ba', 56, [0.16667 + 0.5*M5 + M42, 0.18393 - 0.25*M3 + M22 + 0.5*M28 + 0.5*M41, 0.16667 + 0.5*M4 + M29]],
        ['Ba', 56, [0.16667 + 0.5*M5 + M42, 0.18393 - 0.25*M3 + M22 + 0.5*M28 + 0.5*M41, 0.83333 - 0.5*M4 - M29]],
        ['Ba', 56, [0.83333 - 0.5*M5 - M42, 0.81607 + 0.25*M3 - M22 - 0.5*M28 - 0.5*M41, 0.83333 - 0.5*M4 - M29]],
        ['Ba', 56, [0.83333 - 0.5*M5 - M42, 0.81607 + 0.25*M3 - M22 - 0.5*M28 - 0.5*M41, 0.16667 + 0.5*M4 + M29]],
        ['Ba', 56, [0.16667 + 0.5*M5 + M42, 0.81607 + 0.25*M3 - M22 - 0.5*M28 - 0.5*M41, 0.83333 - 0.5*M4 - M29]],
        ['Ba', 56, [0.16667 + 0.5*M5 + M42, 0.81607 + 0.25*M3 - M22 - 0.5*M28 - 0.5*M41, 0.16667 + 0.5*M4 + M29]],
        ['Ba', 56, [0.83333 + M5 - M42, 0.18393 + 0.5*M3 + M22 - M28 + 0.5*M41, 0.5]],
        ['Ba', 56, [0.16667 - M5 + M42, 0.18393 + 0.5*M3 + M22 - M28 + 0.5*M41, 0.5]],
        ['Ba', 56, [0.83333 + M5 - M42, 0.81607 - 0.5*M3 - M22 + M28 - 0.5*M41, 0.5]],
        ['Ba', 56, [0.16667 - M5 + M42, 0.81607 - 0.5*M3 - M22 + M28 - 0.5*M41, 0.5]],
        ['Ba', 56, [0.5, 0.18393 + 0.5*M3 + M22 + 0.5*M28 - M41, 0.16667 - M4 + M29]],
        ['Ba', 56, [0.5, 0.18393 + 0.5*M3 + M22 + 0.5*M28 - M41, 0.83333 + M4 - M29]],
        ['Ba', 56, [0.5, 0.81607 - 0.5*M3 - M22 - 0.5*M28 + M41, 0.83333 + M4 - M29]],
        ['Ba', 56, [0.5, 0.81607 - 0.5*M3 - M22 - 0.5*M28 + M41, 0.16667 - M4 + M29]],
        ['Ba', 56, [0.5, 0.18393 - M3 + M22 - M28 - M41, 0.5]],
        ['Ba', 56, [0.5, 0.81607 + M3 - M22 + M28 + M41, 0.5]],

        ['Cu', 29, [0, 0, 0]],
        ['Cu', 29, [0, 0, 0.33333 + M7 + M30]],
        ['Cu', 29, [0, 0, 0.66667 - M7 - M30]],
        ['Cu', 29, [0.33333 + M6 + M43, 0, 0]],
        ['Cu', 29, [0.66667 - M6 - M43, 0, 0]],
        ['Cu', 29, [0.33333 - 0.5*M6 + M43, 0, 0.33333 - 0.5*M7 + M30]],
        ['Cu', 29, [0.33333 - 0.5*M6 + M43, 0, 0.66667 + 0.5*M7 - M30]],
        ['Cu', 29, [0.66667 + 0.5*M6 - M43, 0, 0.33333 - 0.5*M7 + M30]],
        ['Cu', 29, [0.66667 + 0.5*M6 - M43, 0, 0.66667 + 0.5*M7 - M30]],
        ['Cu', 29, [0, 0.35501 + M8 + M23 + M31 + M44, 0]],
        ['Cu', 29, [0, 0.64499 - M8 - M23 - M31 - M44, 0]],
        ['Cu', 29, [0, 0.35501 - 0.5*M8 + M23 - 0.5*M31 + M44, 0.33333 + M9 + M32]],
        ['Cu', 29, [0, 0.35501 - 0.5*M8 + M23 - 0.5*M31 + M44, 0.66667 - M9 - M32]],
        ['Cu', 29, [0, 0.64499 + 0.5*M8 - M23 + 0.5*M31 - M44, 0.33333 + M9 + M32]],
        ['Cu', 29, [0, 0.64499 + 0.5*M8 - M23 + 0.5*M31 - M44, 0.66667 - M9 - M32]],
        ['Cu', 29, [0.33333 + M10 + M45, 0.35501 - 0.5*M8 + M23 + M31 - 0.5*M44, 0]],
        ['Cu', 29, [0.66667 - M10 - M45, 0.35501 - 0.5*M8 + M23 + M31 - 0.5*M44, 0]],
        ['Cu', 29, [0.33333 + M10 + M45, 0.64499 + 0.5*M8 - M23 - M31 + 0.5*M44, 0]],
        ['Cu', 29, [0.66667 - M10 - M45, 0.64499 + 0.5*M8 - M23 - M31 + 0.5*M44, 0]],
        ['Cu', 29, [0.33333 - 0.5*M10 + M45, 0.35501 + 0.25*M8 + M23 - 0.5*M31 - 0.5*M44, 0.33333 - 0.5*M9 + M32]],
        ['Cu', 29, [0.33333 - 0.5*M10 + M45, 0.35501 + 0.25*M8 + M23 - 0.5*M31 - 0.5*M44, 0.66667 + 0.5*M9 - M32]],
        ['Cu', 29, [0.66667 + 0.5*M10 - M45, 0.35501 + 0.25*M8 + M23 - 0.5*M31 - 0.5*M44, 0.33333 - 0.5*M9 + M32]],
        ['Cu', 29, [0.66667 + 0.5*M10 - M45, 0.35501 + 0.25*M8 + M23 - 0.5*M31 - 0.5*M44, 0.66667 + 0.5*M9 - M32]],
        ['Cu', 29, [0.33333 - 0.5*M10 + M45, 0.64499 - 0.25*M8 - M23 + 0.5*M31 + 0.5*M44, 0.33333 - 0.5*M9 + M32]],
        ['Cu', 29, [0.33333 - 0.5*M10 + M45, 0.64499 - 0.25*M8 - M23 + 0.5*M31 + 0.5*M44, 0.66667 + 0.5*M9 - M32]],
        ['Cu', 29, [0.66667 + 0.5*M10 - M45, 0.64499 - 0.25*M8 - M23 + 0.5*M31 + 0.5*M44, 0.33333 - 0.5*M9 + M32]],
        ['Cu', 29, [0.66667 + 0.5*M10 - M45, 0.64499 - 0.25*M8 - M23 + 0.5*M31 + 0.5*M44, 0.66667 + 0.5*M9 - M32]],

        ['O', 8, [0, 0, 0.16667 + M12 + M33]],
        ['O', 8, [0, 0, 0.83333 - M12 - M33]],
        ['O', 8, [0, 0, 0.5]],
        ['O', 8, [0.33333 + 0.5*M11 + M46, 0, 0.16667 - 0.5*M12 + M33]],
        ['O', 8, [0.33333 + 0.5*M11 + M46, 0, 0.83333 + 0.5*M12 - M33]],
        ['O', 8, [0.66667 - 0.5*M11 - M46, 0, 0.16667 - 0.5*M12 + M33]],
        ['O', 8, [0.66667 - 0.5*M11 - M46, 0, 0.83333 + 0.5*M12 - M33]],
        ['O', 8, [0.33333 - M11 + M46, 0, 0.5]],
        ['O', 8, [0.66667 + M11 - M46, 0, 0.5]],
        ['O', 8, [0.83333 - M15 - M48, 0.37819 + 0.5*M13 + M24 + M34 + 0.5*M47, 0]],
        ['O', 8, [0.16667 + M15 + M48, 0.37819 + 0.5*M13 + M24 + M34 + 0.5*M47, 0]],
        ['O', 8, [0.83333 - M15 - M48, 0.62181 - 0.5*M13 - M24 - M34 - 0.5*M47, 0]],
        ['O', 8, [0.16667 + M15 + M48, 0.62181 - 0.5*M13 - M24 - M34 - 0.5*M47, 0]],
        ['O', 8, [0.83333 + 0.5*M15 - M48, 0.37819 - 0.25*M13 + M24 - 0.5*M34 + 0.5*M47, 0.33333 + 0.5*M14 + M35]],
        ['O', 8, [0.83333 + 0.5*M15 - M48, 0.37819 - 0.25*M13 + M24 - 0.5*M34 + 0.5*M47, 0.66667 - 0.5*M14 - M35]],
        ['O', 8, [0.16667 - 0.5*M15 + M48, 0.37819 - 0.25*M13 + M24 - 0.5*M34 + 0.5*M47, 0.33333 + 0.5*M14 + M35]],
        ['O', 8, [0.16667 - 0.5*M15 + M48, 0.37819 - 0.25*M13 + M24 - 0.5*M34 + 0.5*M47, 0.66667 - 0.5*M14 - M35]],
        ['O', 8, [0.83333 + 0.5*M15 - M48, 0.62181 + 0.25*M13 - M24 + 0.5*M34 - 0.5*M47, 0.33333 + 0.5*M14 + M35]],
        ['O', 8, [0.83333 + 0.5*M15 - M48, 0.62181 + 0.25*M13 - M24 + 0.5*M34 - 0.5*M47, 0.66667 - 0.5*M14 - M35]],
        ['O', 8, [0.16667 - 0.5*M15 + M48, 0.62181 + 0.25*M13 - M24 + 0.5*M34 - 0.5*M47, 0.33333 + 0.5*M14 + M35]],
        ['O', 8, [0.16667 - 0.5*M15 + M48, 0.62181 + 0.25*M13 - M24 + 0.5*M34 - 0.5*M47, 0.66667 - 0.5*M14 - M35]],
        ['O', 8, [0.5, 0.37819 - M13 + M24 + M34 - M47, 0]],
        ['O', 8, [0.5, 0.62181 + M13 - M24 - M34 + M47, 0]],
        ['O', 8, [0.5, 0.37819 + 0.5*M13 + M24 - 0.5*M34 - M47, 0.33333 - M14 + M35]],
        ['O', 8, [0.5, 0.37819 + 0.5*M13 + M24 - 0.5*M34 - M47, 0.66667 + M14 - M35]],
        ['O', 8, [0.5, 0.62181 - 0.5*M13 - M24 + 0.5*M34 + M47, 0.33333 - M14 + M35]],
        ['O', 8, [0.5, 0.62181 - 0.5*M13 - M24 + 0.5*M34 + M47, 0.66667 + M14 - M35]],
        ['O', 8, [0, 0.37693 + 0.5*M16 + M25 + 0.5*M36 + M49, 0.16667 + M17 + M37]],
        ['O', 8, [0, 0.37693 + 0.5*M16 + M25 + 0.5*M36 + M49, 0.83333 - M17 - M37]],
        ['O', 8, [0, 0.62307 - 0.5*M16 - M25 - 0.5*M36 - M49, 0.83333 - M17 - M37]],
        ['O', 8, [0, 0.62307 - 0.5*M16 - M25 - 0.5*M36 - M49, 0.16667 + M17 + M37]],
        ['O', 8, [0, 0.37693 - M16 + M25 - M36 + M49, 0.5]],
        ['O', 8, [0, 0.62307 + M16 - M25 + M36 - M49, 0.5]],
        ['O', 8, [0.33333 + 0.5*M18 + M50, 0.37693 - 0.25*M16 + M25 + 0.5*M36 - 0.5*M49, 0.16667 - 0.5*M17 + M37]],
        ['O', 8, [0.33333 + 0.5*M18 + M50, 0.37693 - 0.25*M16 + M25 + 0.5*M36 - 0.5*M49, 0.83333 + 0.5*M17 - M37]],
        ['O', 8, [0.66667 - 0.5*M18 - M50, 0.37693 - 0.25*M16 + M25 + 0.5*M36 - 0.5*M49, 0.16667 - 0.5*M17 + M37]],
        ['O', 8, [0.66667 - 0.5*M18 - M50, 0.37693 - 0.25*M16 + M25 + 0.5*M36 - 0.5*M49, 0.83333 + 0.5*M17 - M37]],
        ['O', 8, [0.33333 + 0.5*M18 + M50, 0.62307 + 0.25*M16 - M25 - 0.5*M36 + 0.5*M49, 0.83333 + 0.5*M17 - M37]],
        ['O', 8, [0.33333 + 0.5*M18 + M50, 0.62307 + 0.25*M16 - M25 - 0.5*M36 + 0.5*M49, 0.16667 - 0.5*M17 + M37]],
        ['O', 8, [0.66667 - 0.5*M18 - M50, 0.62307 + 0.25*M16 - M25 - 0.5*M36 + 0.5*M49, 0.83333 + 0.5*M17 - M37]],
        ['O', 8, [0.66667 - 0.5*M18 - M50, 0.62307 + 0.25*M16 - M25 - 0.5*M36 + 0.5*M49, 0.16667 - 0.5*M17 + M37]],
        ['O', 8, [0.33333 - M18 + M50, 0.37693 + 0.5*M16 + M25 - M36 - 0.5*M49, 0.5]],
        ['O', 8, [0.66667 + M18 - M50, 0.37693 + 0.5*M16 + M25 - M36 - 0.5*M49, 0.5]],
        ['O', 8, [0.33333 - M18 + M50, 0.62307 - 0.5*M16 - M25 + M36 + 0.5*M49, 0.5]],
        ['O', 8, [0.66667 + M18 - M50, 0.62307 - 0.5*M16 - M25 + M36 + 0.5*M49, 0.5]],
        ['O', 8, [0, 0.1584 + M19 + M26 + M38 + M51, 0]],
        ['O', 8, [0, 0.8416 - M19 - M26 - M38 - M51, 0]],
        ['O', 8, [0, 0.1584 - 0.5*M19 + M26 - 0.5*M38 + M51, 0.33333 + M20 + M39]],
        ['O', 8, [0, 0.1584 - 0.5*M19 + M26 - 0.5*M38 + M51, 0.66667 - M20 - M39]],
        ['O', 8, [0, 0.8416 + 0.5*M19 - M26 + 0.5*M38 - M51, 0.33333 + M20 + M39]],
        ['O', 8, [0, 0.8416 + 0.5*M19 - M26 + 0.5*M38 - M51, 0.66667 - M20 - M39]],
        ['O', 8, [0.33333 + M21 + M52, 0.1584 - 0.5*M19 + M26 + M38 - 0.5*M51, 0]],
        ['O', 8, [0.66667 - M21 - M52, 0.1584 - 0.5*M19 + M26 + M38 - 0.5*M51, 0]],
        ['O', 8, [0.33333 + M21 + M52, 0.8416 + 0.5*M19 - M26 - M38 + 0.5*M51, 0]],
        ['O', 8, [0.66667 - M21 - M52, 0.8416 + 0.5*M19 - M26 - M38 + 0.5*M51, 0]],
        ['O', 8, [0.33333 - 0.5*M21 + M52, 0.1584 + 0.25*M19 + M26 - 0.5*M38 - 0.5*M51, 0.33333 - 0.5*M20 + M39]],
        ['O', 8, [0.33333 - 0.5*M21 + M52, 0.1584 + 0.25*M19 + M26 - 0.5*M38 - 0.5*M51, 0.66667 + 0.5*M20 - M39]],
        ['O', 8, [0.66667 + 0.5*M21 - M52, 0.1584 + 0.25*M19 + M26 - 0.5*M38 - 0.5*M51, 0.33333 - 0.5*M20 + M39]],
        ['O', 8, [0.66667 + 0.5*M21 - M52, 0.1584 + 0.25*M19 + M26 - 0.5*M38 - 0.5*M51, 0.66667 + 0.5*M20 - M39]],
        ['O', 8, [0.33333 - 0.5*M21 + M52, 0.8416 - 0.25*M19 - M26 + 0.5*M38 + 0.5*M51, 0.33333 - 0.5*M20 + M39]],
        ['O', 8, [0.33333 - 0.5*M21 + M52, 0.8416 - 0.25*M19 - M26 + 0.5*M38 + 0.5*M51, 0.66667 + 0.5*M20 - M39]],
        ['O', 8, [0.66667 + 0.5*M21 - M52, 0.8416 - 0.25*M19 - M26 + 0.5*M38 + 0.5*M51, 0.33333 - 0.5*M20 + M39]],
        ['O', 8, [0.66667 + 0.5*M21 - M52, 0.8416 - 0.25*M19 - M26 + 0.5*M38 + 0.5*M51, 0.66667 + 0.5*M20 - M39]],
    ]
    # # swap the element in res[:, 2, 1] and res[:, 2, 2]
    # # res is a list of lists
    # for i in range(len(res)):
    #     res[i][2][1], res[i][2][2] = res[i][2][2], res[i][2][1]

    return res


