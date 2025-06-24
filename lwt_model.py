import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# --- Custom N-Dimensional Lifting Wavelet Transform (LWT) Implementation ---
# This implementation recursively applies 1D LWT along each dimension.
# It supports 'haar' and a 'linear_bior' (simplified biorthogonal) wavelet.

def _custom_1d_forward_single_level(data, wavelet_type='haar'):
    """Performs a single level of 1D LWT (forward) based on wavelet_type."""
    n = len(data)
    
    # Pad for even length and boundary handling based on filter size
    pad_len = 0
    if wavelet_type == 'haar':
        if n % 2 != 0:
            pad_len = 1
    elif wavelet_type == 'linear_bior':
        # For predict step using (even[i] + even[i+1])/2, we need one extra element at the end
        if n % 2 != 0: # Ensure even number of samples for odd/even split
            pad_len = 1
        # Also need padding for filter taps at the end for prediction (x_e[i+1])
        # Simple symmetric padding if data is too short for filter.
        # This is a basic pad, for robustness, periodic or symmetric extension would be better.
        if n < 2: # Need at least 2 elements for prediction
            pad_len = max(pad_len, 2 - n) # Ensure at least 2 elements
    
    if pad_len > 0:
        data = np.pad(data, (0, pad_len), 'reflect') # 'reflect' for better signal extension
        n = len(data)
    
    even = data[0::2]
    odd = data[1::2]

    # Handle odd/even lengths for detail/approx calculation if they are different
    min_len = min(len(even), len(odd))
    even = even[:min_len]
    odd = odd[:min_len]

    detail = np.zeros_like(odd, dtype=float)
    approximation = np.zeros_like(even, dtype=float)

    if wavelet_type == 'haar':
        detail = odd - even
        approximation = even + 0.5 * detail
    elif wavelet_type == 'linear_bior':
        # Predict: d_j = x_o[j] - 0.5 * (x_e[j] + x_e[j+1])
        # Need to handle x_e[j+1] for the last element of even.
        # This simple padding inside the 1D function might not be ideal for complex cases.
        # A more robust solution involves global symmetric extension of the full N-D array.
        
        # Ensure even has enough elements for (even[i] + even[i+1])
        even_padded_for_predict = np.pad(even, (0,1), 'reflect')

        detail = odd - 0.5 * (even_padded_for_predict[:-1] + even_padded_for_predict[1:])
        
        # Update: a_j = x_e[j] + 0.5 * d_j
        approximation = even + 0.5 * detail # Simple update
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet_type}")

    return approximation, detail

def _custom_1d_inverse_single_level(approximation, detail, wavelet_type='haar'):
    """Performs a single level of 1D LWT (inverse) reconstruction based on wavelet_type."""
    # Ensure approx and detail have same length for inverse operations
    min_len = min(len(approximation), len(detail))
    approximation = approximation[:min_len]
    detail = detail[:min_len]

    even_reconstructed = np.zeros_like(approximation, dtype=float)
    odd_reconstructed = np.zeros_like(detail, dtype=float)

    if wavelet_type == 'haar':
        even_reconstructed = approximation - 0.5 * detail
        odd_reconstructed = detail + even_reconstructed
    elif wavelet_type == 'linear_bior':
        # Inverse Update: even_reconstructed = approximation - 0.5 * detail
        even_reconstructed = approximation - 0.5 * detail

        # Inverse Predict: odd_reconstructed = detail + 0.5 * (even_reconstructed[i] + even_reconstructed[i+1])
        # Need to handle x_e[j+1] for the last element of even_reconstructed.
        even_reconstructed_padded_for_predict = np.pad(even_reconstructed, (0,1), 'reflect')
        odd_reconstructed = detail + 0.5 * (even_reconstructed_padded_for_predict[:-1] + even_reconstructed_padded_for_predict[1:])
    else:
        raise ValueError(f"Unknown wavelet type: {wavelet_type}")

    reconstructed_full = np.empty(len(even_reconstructed) + len(odd_reconstructed), dtype=float)
    reconstructed_full[0::2] = even_reconstructed
    reconstructed_full[1::2] = odd_reconstructed

    return reconstructed_full

def _generate_coeff_types(ndim):
    """Generates all 2^ndim possible coefficient type strings (e.g., 'AA', 'AD', 'DA', 'DD' for 2D)."""
    if ndim == 0:
        return [""]
    
    prev_types = _generate_coeff_types(ndim - 1)
    new_types = []
    for p_type in prev_types:
        new_types.append(p_type + 'A')
        new_types.append(p_type + 'D')
    return new_types

def custom_n_d_haar_forward(data_nd, level=1, wavelet_type='haar'):
    """
    Performs multi-level N-D separable LWT using specified wavelet type.
    Returns: (cA_n, {details_n}, {details_n-1}, ..., {details_1})
    where details_k is a dictionary of detail coefficients at level k (e.g., {'aad': array}).
    """
    if level < 0:
        raise ValueError("Decomposition level must be non-negative.")
    if level == 0:
        return (data_nd,) # No decomposition, return original data as approximation

    ndim = data_nd.ndim
    all_level_details = []
    current_approx = np.copy(data_nd)

    for l in range(level):
        current_level_coeff_dict = {}
        # The list 'blocks_to_process' holds tuples of (data_block, its_type_string_so_far)
        # Initially, only the full approximation block (type 'A' for all axes)
        blocks_to_process = [(current_approx, 'A' * ndim)]
        
        # Iterate through each dimension (axis)
        for axis_idx in range(ndim):
            next_blocks_for_axis = []
            for block_data, block_type_str_so_far in blocks_to_process:
                # Apply 1D LWT along the current axis for each slice
                approx_slices = []
                detail_slices = []
                
                # Iterate through slices along the current_axis
                iter_slices = [slice(None)] * ndim
                for i in range(block_data.shape[axis_idx]):
                    iter_slices[axis_idx] = i
                    slice_data = block_data[tuple(iter_slices)]
                    approx_slice, detail_slice = _custom_1d_forward_single_level(slice_data, wavelet_type)
                    approx_slices.append(approx_slice)
                    detail_slices.append(detail_slice)
                
                # Stack results back into (N-1)-D arrays, or N-D if this is the first axis
                approx_block_from_axis = np.stack(approx_slices, axis=axis_idx)
                detail_block_from_axis = np.stack(detail_slices, axis=axis_idx)

                # Generate new block types (e.g., if block_type_str_so_far was 'AA', now 'AAA' and 'AAD')
                # This correctly constructs the N-D type string (e.g., 'AADA')
                type_A = list(block_type_str_so_far)
                type_D = list(block_type_str_so_far)
                type_A[axis_idx] = 'A'
                type_D[axis_idx] = 'D'

                next_blocks_for_axis.append((approx_block_from_axis, "".join(type_A)))
                next_blocks_for_axis.append((detail_block_from_axis, "".join(type_D)))
            blocks_to_process = next_blocks_for_axis
        
        # After processing all dimensions for this level, filter for current cA and details
        cA_this_level = None
        for block_data, block_type_str in blocks_to_process:
            if block_type_str == 'A' * ndim:
                cA_this_level = block_data
            else:
                current_level_coeff_dict[block_type_str] = block_data
        
        if cA_this_level is None:
            raise RuntimeError("Approximation coefficients (all 'A' type) not found after decomposition level.")
        
        all_level_details.append(current_level_coeff_dict)
        current_approx = cA_this_level # This becomes the input for the next decomposition level

    # Final approximation is `current_approx` after all levels
    final_cA = current_approx
    
    # Reorder details to match pywt.wavedecn structure: (cA_n, {details_n}, {details_n-1}, ..., {details_1})
    coeffs_output_structure = [final_cA] + list(reversed(all_level_details))
    return tuple(coeffs_output_structure)

def custom_n_d_haar_inverse(coeffs_tuple, wavelet_type='haar'):
    """
    Performs multi-level N-D separable LWT inverse using specified wavelet type.
    coeffs_tuple: (cA_n, {details_n}, {details_n-1}, ..., {details_1})
    """
    if not isinstance(coeffs_tuple, tuple) or len(coeffs_tuple) < 1:
        raise ValueError("Coefficients must be a tuple from custom_n_d_haar_forward_recursive.")

    reconstructed_data_block = coeffs_tuple[0] # Highest level approximation
    detail_levels = list(coeffs_tuple[1:]) # List of detail dictionaries, ordered from high to low level

    # Iterate backwards through levels (from highest level of details to lowest)
    for details_for_this_level_dict in detail_levels:
        ndim = reconstructed_data_block.ndim # Number of dimensions of the current block
        
        # Prepare the blocks for inverse transformation at this level
        # This will contain the 'A' block (reconstructed_data_block) from the previous (higher) level
        # and all 'D' blocks for this current level's details.
        current_level_blocks = {('A' * ndim): reconstructed_data_block}
        current_level_blocks.update(details_for_this_level_dict)
        
        # Inverse transform through each dimension in reverse order
        for axis_idx in reversed(range(ndim)):
            newly_reconstructed_blocks = {} # Stores results after inverse along current axis
            
            # Get all current keys that represent an 'A' component at the current axis_idx.
            # These are the blocks that, when combined with their 'D' counterpart,
            # will reconstruct a block for the *previous* stage of the transform along this axis.
            a_component_keys_at_current_stage = [
                k for k in current_level_blocks.keys() if k[axis_idx] == 'A'
            ]
            
            for a_type_str in a_component_keys_at_current_stage:
                # Construct the corresponding 'D' type string for pairing along this axis
                d_type_str = a_type_str[:axis_idx] + 'D' + a_type_str[axis_idx+1:]
                
                # Retrieve the A-component and D-component blocks for this pair
                # Both block_A and block_D *must* be present in current_level_blocks at this point.
                block_A = current_level_blocks[a_type_str]
                block_D = current_level_blocks[d_type_str]
                
                reconstructed_slices = []
                iter_slices = [slice(None)] * ndim
                # Iterate through slices along the current_axis to apply 1D inverse
                for i in range(block_A.shape[axis_idx]):
                    iter_slices[axis_idx] = i
                    approx_slice = block_A[tuple(iter_slices)]
                    detail_slice = block_D[tuple(iter_slices)]
                    reconstructed_slice = _custom_1d_inverse_single_level(approx_slice, detail_slice, wavelet_type)
                    reconstructed_slices.append(reconstructed_slice)
                
                # Stack the 1D reconstructed slices back into an N-D block
                reconstructed_block_from_axis = np.stack(reconstructed_slices, axis=axis_idx)
                
                # Store this newly reconstructed block. Its key is the same 'A' type string
                # because it represents the state *before* this axis was transformed into A/D components.
                newly_reconstructed_blocks[a_type_str] = reconstructed_block_from_axis
            
            # Update `current_level_blocks` for the next iteration (previous axis).
            # This set of blocks now represents the state after inverse transforming along `axis_idx`.
            current_level_blocks = newly_reconstructed_blocks
            
        # After inverse transforming across all dimensions for this level,
        # there should be only one block left in `current_level_blocks`,
        # which is the fully reconstructed data for this level (its key will be 'A'*ndim).
        # This reconstructed block becomes the `reconstructed_data_block` for the next lower level.
        reconstructed_data_block = list(current_level_blocks.values())[0] 

    return reconstructed_data_block


# --- Rest of the Pipeline ---

# --- 1. Load and Prepare Data ---

try:
    df = pd.read_csv('Data_August_Renewable.csv')
    if 'Time' in df.columns and 'Speed' in df.columns and 'Reduced' in df.columns:
        time_data = df['Time'].values
        speed_data = df['Speed'].values
        raw_target_data = df['Reduced'].values
    else:
        print(" 'Time', 'Speed', or 'Reduced' columns not found. Using the first three columns as input and target.")
        if df.shape[1] >= 3:
            time_data = df.iloc[:, 0].values
            speed_data = df.iloc[:, 1].values
            raw_target_data = df.iloc[:, 2].values
        else:
            raise ValueError("CSV does not contain enough columns for 'Time', 'Speed', and 'Reduced' data.")

    time_data = pd.to_numeric(time_data, errors='coerce')
    speed_data = pd.to_numeric(speed_data, errors='coerce')
    raw_target_data = pd.to_numeric(raw_target_data, errors='coerce')

    valid_indices = ~np.isnan(time_data) & ~np.isnan(speed_data) & ~np.isnan(raw_target_data)
    time_data = time_data[valid_indices]
    speed_data = speed_data[valid_indices]
    raw_target_data = raw_target_data[valid_indices]

except FileNotFoundError:
    print("Error: CSV file not found. Creating dummy data for demonstration.")
    np.random.seed(42)
    total_dummy_samples = 5000
    time_data = np.arange(1, total_dummy_samples + 1).astype(float)
    speed_data = np.sin(np.linspace(0, 100, total_dummy_samples)) * 5 + np.random.randn(total_dummy_samples) * 0.5 + 10
    raw_target_data = np.cos(np.linspace(0, 100, total_dummy_samples)) * 3 + np.random.randn(total_dummy_samples) * 0.3 + 8

print(f"Original time data length: {len(time_data)}")
print(f"Original speed data length: {len(speed_data)}")
print(f"Original raw target data length: {len(raw_target_data)}")

# --- User Defines N-dimensional Shape ---
while True:
    try:
        dim_input = input("\nEnter desired N-dimensional shape as comma-separated integers (e.g., '31,162' for 2D, '4,31,40' for 3D): ")
        TARGET_DIMENSIONS = tuple(map(int, dim_input.split(',')))
        if not TARGET_DIMENSIONS:
            raise ValueError("Dimensions cannot be empty.")
        
        total_samples_for_target_dims = np.prod(TARGET_DIMENSIONS)
        if total_samples_for_target_dims < len(time_data):
            print(f"Warning: Your specified dimensions ({TARGET_DIMENSIONS}) result in {total_samples_for_target_dims} total elements, which is less than the data length ({len(time_data)}). Data will be truncated.")
            confirm = input("Continue with truncation? (yes/no): ").lower()
            if confirm != 'yes':
                continue # Ask again
        elif total_samples_for_target_dims > len(time_data):
            print(f"Info: Your specified dimensions ({TARGET_DIMENSIONS}) result in {total_samples_for_target_dims} total elements, which is more than the data length ({len(time_data)}). Data will be padded.")
        else:
            print(f"Dimensions ({TARGET_DIMENSIONS}) perfectly match data length ({len(time_data)}).")
        break # Exit loop if input is valid
    except ValueError as e:
        print(f"Invalid input: {e}. Please enter comma-separated integers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Please try again.")

print(f"\nUsing TARGET_DIMENSIONS: {TARGET_DIMENSIONS}")

# --- User Defines Wavelet Type ---
while True:
    wavelet_choice = input("Choose wavelet type ('haar' or 'linear_bior'): ").lower()
    if wavelet_choice in ['haar', 'linear_bior']:
        break
    else:
        print("Invalid wavelet choice. Please enter 'haar' or 'linear_bior'.")
print(f"Using wavelet type: {wavelet_choice}")


# --- Function to reshape and pad 1D data for N-D Wavelet Transform ---
def prepare_data_for_nd_lwt(data_1d, target_dims):
    current_len = len(data_1d)
    total_required = np.prod(target_dims)

    if current_len > total_required:
        # If original data is larger, truncate it.
        # This handles cases where user provides smaller target_dims than actual data.
        print(f"Data ({current_len} samples) is larger than target dimensions product ({total_required}). Truncating to fit.")
        padded_1d = data_1d[:total_required]
    elif current_len < total_required:
        # If original data is smaller, pad it.
        padding_needed = total_required - current_len
        padded_1d = np.pad(data_1d, (0, padding_needed), 'constant', constant_values=0)
        print(f"Padding {padding_needed} samples to reach {total_required} total samples for reshaping.")
    else:
        padded_1d = data_1d

    data_nd = padded_1d.reshape(target_dims)
    print(f"Reshaped N-D data shape: {data_nd.shape}")

    # Pad each dimension independently to the nearest power of 2 for optimal wavelet transform
    padding_config = []
    for dim_size in data_nd.shape:
        next_power_of_2 = 2**int(np.ceil(np.log2(dim_size)))
        padding_config.append((0, next_power_of_2 - dim_size))

    padded_for_wavelet_data_nd = np.pad(data_nd,
                                        tuple(padding_config),
                                        'constant', constant_values=0)
    print(f"Padded N-D data for wavelet transform shape (to powers of 2): {padded_for_wavelet_data_nd.shape}")
    return data_nd, padded_for_wavelet_data_nd, data_nd.shape # Return original N-D shape for cropping later


# --- Prepare Input Data (Time and Speed) for LWT ---
print("\n--- Preparing Time Data ---")
original_time_data_nd, padded_time_for_lwt_nd, original_time_shape = \
    prepare_data_for_nd_lwt(time_data, TARGET_DIMENSIONS)

print("\n--- Preparing Speed Data ---")
original_speed_data_nd, padded_speed_for_lwt_nd, original_speed_shape = \
    prepare_data_for_nd_lwt(speed_data, TARGET_DIMENSIONS)

# --- Prepare Target Data (Reduced) for LWT ---
print("\n--- Preparing Target (Reduced) Data ---")
original_target_data_nd, padded_target_for_lwt_nd, original_y_shape = \
    prepare_data_for_nd_lwt(raw_target_data, TARGET_DIMENSIONS)


# --- 2. Multi-dimensional Lifting Wavelet Transform (LWT) - Using Custom LWT ---
decomposition_level = 3 # As specified in the prompt

print(f"\nPerforming custom N-D {wavelet_choice.upper()} LWT on Time data up to level {decomposition_level}...")
coeffs_time_nd = custom_n_d_haar_forward(padded_time_for_lwt_nd, level=decomposition_level, wavelet_type=wavelet_choice)
lwt_features_time = coeffs_time_nd[0].flatten() # Approximation coefficients for Time
print(f"Flattened LWT features for X (Time) length: {len(lwt_features_time)}")

print(f"\nPerforming custom N-D {wavelet_choice.upper()} LWT on Speed data up to level {decomposition_level}...")
coeffs_speed_nd = custom_n_d_haar_forward(padded_speed_for_lwt_nd, level=decomposition_level, wavelet_type=wavelet_choice)
lwt_features_speed = coeffs_speed_nd[0].flatten() # Approximation coefficients for Speed
print(f"Flattened LWT features for X (Speed) length: {len(lwt_features_speed)}")

# Concatenate LWT features from Time and Speed for the Neural Network input
if len(lwt_features_time) != len(lwt_features_speed):
    min_len_features_concat = min(len(lwt_features_time), len(lwt_features_speed))
    lwt_features_x = np.concatenate((lwt_features_time[:min_len_features_concat], lwt_features_speed[:min_len_features_concat]))
    print(f"Warning: LWT features for Time ({len(lwt_features_time)}) and Speed ({len(lwt_features_speed)}) have different lengths. Truncating for concatenation.")
else:
    lwt_features_x = np.concatenate((lwt_features_time, lwt_features_speed))

print(f"Flattened combined LWT features for X (Time + Speed) length: {len(lwt_features_x)}")


print(f"\nPerforming custom N-D {wavelet_choice.upper()} LWT on target data (Reduced) up to level {decomposition_level}...")
coeffs_y_nd = custom_n_d_haar_forward(padded_target_for_lwt_nd, level=decomposition_level, wavelet_type=wavelet_choice)
lwt_features_y = coeffs_y_nd[0].flatten() # Approximation coefficients for Y (target)
print(f"Flattened LWT features for Y (Reduced) length: {len(lwt_features_y)}")


# --- 3. Reconstruction and Error Calculation ---

# Reconstruct Time
reconstructed_data_padded_time_nd = custom_n_d_haar_inverse(coeffs_time_nd, wavelet_type=wavelet_choice)
crop_slices_time = tuple(slice(0, dim) for dim in original_time_shape)
reconstructed_data_cropped_time_nd = reconstructed_data_padded_time_nd[crop_slices_time]
mse_time = np.mean((original_time_data_nd.flatten()[:len(reconstructed_data_cropped_time_nd.flatten())] - reconstructed_data_cropped_time_nd.flatten())**2)
print(f"\nMean Squared Error (MSE) for Time after Level {decomposition_level} Custom N-D LWT reconstruction: {mse_time:.4f}")


# Reconstruct Speed
reconstructed_data_padded_speed_nd = custom_n_d_haar_inverse(coeffs_speed_nd, wavelet_type=wavelet_choice)
crop_slices_speed = tuple(slice(0, dim) for dim in original_speed_shape)
reconstructed_data_cropped_speed_nd = reconstructed_data_padded_speed_nd[crop_slices_speed]
mse_speed = np.mean((original_speed_data_nd.flatten()[:len(reconstructed_data_cropped_speed_nd.flatten())] - reconstructed_data_cropped_speed_nd.flatten())**2)
print(f"Mean Squared Error (MSE) for Speed after Level {decomposition_level} Custom N-D LWT reconstruction: {mse_speed:.4f}")

# Reconstruct Y (Reduced)
reconstructed_data_padded_y_nd = custom_n_d_haar_inverse(coeffs_y_nd, wavelet_type=wavelet_choice)
crop_slices_y = tuple(slice(0, dim) for dim in original_y_shape)
reconstructed_data_cropped_y_nd = reconstructed_data_padded_y_nd[crop_slices_y]
mse_y = np.mean((original_target_data_nd.flatten()[:len(reconstructed_data_cropped_y_nd.flatten())] - reconstructed_data_cropped_y_nd.flatten())**2)
print(f"Mean Squared Error (MSE) for Y (Reduced) after Level {decomposition_level} Custom N-D LWT reconstruction: {mse_y:.4f}")


# --- 4. Neural Network Implementation ---

# Prepare input (X) and target (y) for NN
X_nn = np.array([lwt_features_x])
y_nn = np.array([lwt_features_y])

print(f"\nNeural Network Input (X_nn) shape: {X_nn.shape}")
print(f"Neural Network Target (y_nn) shape: {y_nn.shape}")

# Define the Neural Network model
model_nd = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=(X_nn.shape[1],),
                 kernel_initializer='he_normal', bias_initializer='zeros',
                 name='hidden_layer_1'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros',
                 name='hidden_layer_2'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu', kernel_initializer='he_normal', bias_initializer='zeros',
                 name='hidden_layer_3'),
    layers.Dense(y_nn.shape[1], activation='linear', name='output_layer')
])

# Compile the model
model_nd.compile(optimizer='adam', loss='mse', metrics=['mae'])
model_nd.summary()

print("\nTraining the Neural Network...")
history_nd = model_nd.fit(X_nn, y_nn, epochs=5000, verbose=1)

print(f"NN Training Loss (last epoch): {history_nd.history['loss'][-1]:.4f}")
print(f"NN Training MAE (last epoch): {history_nd.history['mae'][-1]:.4f}")

# Make a prediction with the trained model
predicted_lwt_features_y = model_nd.predict(X_nn)
print(f"\nPredicted LWT features (shape): {predicted_lwt_features_y.shape}")

# --- 5. Inverse LWT to get predicted 'Reduced' data back to original space ---
predicted_cA3_y_nd_shape = coeffs_y_nd[0].shape
predicted_cA3_y_nd = predicted_lwt_features_y.reshape(predicted_cA3_y_nd_shape)

# For reconstruction, we combine the predicted approximation coefficients with the
# original detail coefficients from the target's LWT. In a pure prediction scenario,
# if you only predict approximation coefficients, the details would be lost,
# leading to a smoother (lower frequency) reconstruction.
# We create a new coeffs tuple for the predicted data, replacing the approximation,
# but keeping the original detail coefficients for accurate reconstruction quality display.
reconstruction_coeffs_for_prediction = (predicted_cA3_y_nd,) + coeffs_y_nd[1:]

predicted_raw_target_padded_nd = custom_n_d_haar_inverse(reconstruction_coeffs_for_prediction, wavelet_type=wavelet_choice)

crop_slices_y = tuple(slice(0, dim) for dim in original_y_shape)
predicted_raw_target_cropped_nd = predicted_raw_target_padded_nd[crop_slices_y]

predicted_raw_target_flat = predicted_raw_target_cropped_nd.flatten()
actual_raw_target_flat = raw_target_data

min_len_final_comp = min(len(predicted_raw_target_flat), len(actual_raw_target_flat))
predicted_raw_target_flat = predicted_raw_target_flat[:min_len_final_comp]
actual_raw_target_flat = actual_raw_target_flat[:min_len_final_comp]

final_prediction_mse = np.mean((actual_raw_target_flat - predicted_raw_target_flat)**2)
print(f"\nFinal Prediction MSE (comparing original 'Reduced' vs. NN-predicted & reconstructed 'Reduced'): {final_prediction_mse:.4f}")


# --- 6. Visualization ---
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(time_data, label='Original Time Data', alpha=0.8, color='blue')
# Ensure reconstruction is cropped to original data length for plotting
plt.plot(reconstructed_data_cropped_time_nd.flatten()[:len(time_data)], label=f'Reconstructed Time (from Custom N-D {wavelet_choice.upper()} LWT)', linestyle='--', alpha=0.7, color='lightblue')
plt.title(f'Original vs. Reconstructed Time (N={len(TARGET_DIMENSIONS)}D)')
plt.xlabel('Sample Index')
plt.ylabel('Time Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(speed_data, label='Original Speed Data', alpha=0.8, color='orange')
plt.plot(reconstructed_data_cropped_speed_nd.flatten()[:len(speed_data)], label=f'Reconstructed Speed (from Custom N-D {wavelet_choice.upper()} LWT)', linestyle='--', alpha=0.7, color='lightsalmon')
plt.title(f'Original Speed vs. Reconstructed Speed (N={len(TARGET_DIMENSIONS)}D)')
plt.xlabel('Sample Index')
plt.ylabel('Speed Value')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(actual_raw_target_flat, label='Original Reduced Data', alpha=0.8, color='green')
plt.plot(predicted_raw_target_flat, label='NN Predicted & Reconstructed Reduced Data', linestyle='--', alpha=0.7, color='red')
plt.title(f'Original Reduced vs. NN Predicted & Reconstructed Reduced (N={len(TARGET_DIMENSIONS)}D)')
plt.xlabel('Sample Index')
plt.ylabel('Reduced Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("\n--- Process Complete ---")
print(f"Custom N-D {wavelet_choice.upper()} LWT (Level {decomposition_level}) applied independently to 'Time' and 'Speed' columns.")
print(f"Features from 'Time' and 'Speed' (cA{decomposition_level} from shape {coeffs_time_nd[0].shape} each) were concatenated to form the NN input.")
print(f"NN predicts features for 'Reduced' (cA{decomposition_level} from shape {coeffs_y_nd[0].shape}).")
print(f"Reconstruction MSE for 'Time': {mse_time:.4f}")
print(f"Reconstruction MSE for 'Speed': {mse_speed:.4f}")
print(f"Reconstruction MSE for 'Reduced' (target): {mse_y:.4f}")
print(f"Final NN Prediction MSE (original 'Reduced' vs. NN-predicted & reconstructed 'Reduced'): {final_prediction_mse:.4f}")
print("\nIMPORTANT: This custom LWT is a general N-D Haar/linear_bior implementation. For other wavelets or for highly optimized production environments,")
print("using a dedicated library like `PyWavelets` is strongly recommended due to its performance, robustness, and wider range of supported wavelets.")
print("Consider creating multiple (X,y) samples for robust NN training in a real application.")
