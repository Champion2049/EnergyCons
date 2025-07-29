# wavelet_transform.py

import numpy as np
import pandas as pd
import os
from itertools import product # Used for generating 'L'/'H' combinations for N-D sub-bands

def haar_lwt_nd_decompose(data, level):
    """
    Performs an N-level N-dimensional Haar Lifting Wavelet Transform decomposition.

    Args:
        data (np.ndarray): The N-dimensional input data array.
        level (int): The number of decomposition levels to perform.

    Returns:
        dict: A dictionary containing the decomposition results.
    """
    current_approx = np.array(data, dtype=float)
    coeffs_tree = {'original_shape': data.shape}
    
    ndim = data.ndim

    for l in range(1, level + 1):
        details_this_level = {}
        level_l_input_shapes_before_axis_transform = {}

        current_set_of_arrays = {'': current_approx} 
        
        for dim_idx in range(ndim):
            next_set_of_arrays = {}
            
            for input_prefix, arr_to_process in current_set_of_arrays.items():
                if arr_to_process.size == 0:
                    continue

                level_l_input_shapes_before_axis_transform[input_prefix] = arr_to_process.shape

                approx_part, detail_part, _ = _apply_1d_lwt_along_axis(arr_to_process, dim_idx)
                
                next_set_of_arrays[input_prefix + 'L'] = approx_part
                next_set_of_arrays[input_prefix + 'H'] = detail_part
            
            current_set_of_arrays = next_set_of_arrays
        
        current_approx = current_set_of_arrays.pop('L' * ndim)
        details_this_level = current_set_of_arrays

        coeffs_tree[f'level_{l}_details'] = details_this_level
        coeffs_tree[f'level_{l}_input_shapes_before_axis_transform'] = level_l_input_shapes_before_axis_transform
        
        if any(s < 2 for s in current_approx.shape) or current_approx.size == 0:
            print(f"Warning: Stopped N-D decomposition at level {l} because approximation sub-band became too small: {current_approx.shape}")
            break
    
    coeffs_tree['final_approx'] = current_approx
    return coeffs_tree

def haar_lwt_nd_reconstruct(coeffs_tree):
    """
    Reconstructs an N-dimensional signal from its N-level Haar LWT coefficients.

    Args:
        coeffs_tree (dict): A dictionary from haar_lwt_nd_decompose.

    Returns:
        np.ndarray: The reconstructed N-dimensional signal.
    """
    current_reconstruction = coeffs_tree['final_approx']
    original_full_shape = coeffs_tree['original_shape']
    ndim = len(original_full_shape)

    levels = 0
    for key in coeffs_tree:
        if key.startswith('level_') and key.endswith('_details'):
            levels = max(levels, int(key.split('_')[1]))
    
    for l in range(levels, 0, -1):
        details_this_level = coeffs_tree[f'level_{l}_details']
        input_shapes_before_axis_transform = coeffs_tree[f'level_{l}_input_shapes_before_axis_transform']

        all_sub_bands_at_this_level = details_this_level.copy()
        all_sub_bands_at_this_level['L' * ndim] = current_reconstruction

        for dim_idx in range(ndim - 1, -1, -1):
            next_reconstructed_arrays = {}
            
            prefix_combinations = [''.join(p) for p in product('LH', repeat=dim_idx)]
            reconstruct_suffix_combinations = [''.join(p) for p in product('R', repeat=(ndim - 1 - dim_idx))]

            for p_before in prefix_combinations:
                for r_suffix in reconstruct_suffix_combinations:
                    approx_key = p_before + 'L' + r_suffix
                    detail_key = p_before + 'H' + r_suffix
                    
                    if approx_key in all_sub_bands_at_this_level and detail_key in all_sub_bands_at_this_level:
                        approx_part = all_sub_bands_at_this_level[approx_key]
                        detail_part = all_sub_bands_at_this_level[detail_key]
                        
                        input_prefix_for_lookup = p_before
                        original_input_shape_for_this_dim = input_shapes_before_axis_transform.get(input_prefix_for_lookup)

                        if original_input_shape_for_this_dim is None:
                            raise KeyError(f"Original input shape not found for (dim_idx={dim_idx}, input_prefix='{input_prefix_for_lookup}') at level {l}.")

                        original_len_for_this_axis = original_input_shape_for_this_dim[dim_idx]

                        reconstructed_part = _apply_1d_inverse_lwt_along_axis(
                            approx_part, detail_part, dim_idx, original_len_for_this_axis
                        )
                        
                        next_reconstructed_arrays[p_before + 'R' + r_suffix] = reconstructed_part
            
            all_sub_bands_at_this_level = next_reconstructed_arrays
        
        current_reconstruction = all_sub_bands_at_this_level['R' * ndim]
    
    final_reconstruction_slicer = tuple(slice(0, s) for s in original_full_shape)
    return current_reconstruction[final_reconstruction_slicer]


def get_coefficients_df_and_save(output_dir, coeffs_tree, filename, level):
    """
    Saves Haar LWT coefficients to a CSV and returns them as a pandas DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_coeff_arrays_flat = []
    headers = []

    if 'final_approx' in coeffs_tree:
        all_coeff_arrays_flat.append(coeffs_tree['final_approx'].flatten())
        headers.append('Final_Approximation')

    levels = 0
    ndim = len(coeffs_tree['original_shape'])
    for key in coeffs_tree:
        if key.startswith('level_') and key.endswith('_details'):
            levels = max(levels, int(key.split('_')[1]))

    all_lh_combinations = [''.join(p) for p in product('LH', repeat=ndim)]
    detail_lh_combinations = [c for c in all_lh_combinations if c != 'L' * ndim]

    for l in range(1, levels + 1):
        details_this_level = coeffs_tree.get(f'level_{l}_details', {})
        for combo in sorted(detail_lh_combinations):
            if combo in details_this_level and details_this_level[combo].size > 0:
                all_coeff_arrays_flat.append(details_this_level[combo].flatten())
                headers.append(f'{combo}_L{l}')

    max_len = max(len(arr) for arr in all_coeff_arrays_flat) if all_coeff_arrays_flat else 0

    padded_coeff_arrays = []
    for arr in all_coeff_arrays_flat:
        padded_arr = np.pad(arr, (0, max_len - len(arr)), 'constant', constant_values=np.nan)
        padded_coeff_arrays.append(padded_arr)

    if padded_coeff_arrays:
        combined_coeffs_2d = np.column_stack(padded_coeff_arrays)
    else:
        combined_coeffs_2d = np.array([[]])

    df = pd.DataFrame(combined_coeffs_2d, columns=headers)

    df['Matlab_coeff'] = df['Final_Approximation'] * level
    
    combined_filepath = os.path.join(output_dir, filename)
    df.to_csv(combined_filepath, index=False)
    
    print(f"Saved all Haar LWT coefficients to: {combined_filepath}")
    return df

# --- Helper private functions from original notebook ---

def _apply_1d_lwt_along_axis(data_nd, axis):
    original_len_axis = data_nd.shape[axis]
    pad_width = [(0, 0)] * data_nd.ndim
    if original_len_axis % 2 != 0:
        pad_width[axis] = (0, 1)
        padded_data_nd = np.pad(data_nd, pad_width, 'constant', constant_values=0)
    else:
        padded_data_nd = data_nd

    slicer_even = [slice(None)] * padded_data_nd.ndim
    slicer_even[axis] = np.arange(0, padded_data_nd.shape[axis], 2)
    even_part = padded_data_nd[tuple(slicer_even)]

    slicer_odd = [slice(None)] * padded_data_nd.ndim
    slicer_odd[axis] = np.arange(1, padded_data_nd.shape[axis], 2)
    odd_part = padded_data_nd[tuple(slicer_odd)]
    
    detail_coeffs_nd = odd_part - even_part
    approx_coeffs_nd = even_part + detail_coeffs_nd / 2
    return approx_coeffs_nd, detail_coeffs_nd, original_len_axis

def _apply_1d_inverse_lwt_along_axis(approx_coeffs_nd, detail_coeffs_nd, axis, original_len_axis):
    even_part = approx_coeffs_nd - detail_coeffs_nd / 2
    odd_part = detail_coeffs_nd + even_part

    reconstructed_shape = list(even_part.shape)
    reconstructed_shape[axis] = even_part.shape[axis] + odd_part.shape[axis]
    reconstructed_padded_nd = np.empty(reconstructed_shape, dtype=float)

    slicer_even_out = [slice(None)] * reconstructed_padded_nd.ndim
    slicer_even_out[axis] = np.arange(0, reconstructed_padded_nd.shape[axis], 2)
    reconstructed_padded_nd[tuple(slicer_even_out)] = even_part

    slicer_odd_out = [slice(None)] * reconstructed_padded_nd.ndim
    slicer_odd_out[axis] = np.arange(1, reconstructed_padded_nd.shape[axis], 2)
    reconstructed_padded_nd[tuple(slicer_odd_out)] = odd_part

    trim_slicer = [slice(None)] * reconstructed_padded_nd.ndim
    trim_slicer[axis] = slice(0, original_len_axis)
    return reconstructed_padded_nd[tuple(trim_slicer)]