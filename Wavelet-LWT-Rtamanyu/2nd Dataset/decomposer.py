import pandas as pd
import numpy as np

original_dataset_path = 'Wavelet-LWT-Rtamanyu/2nd Dataset/TSLA.csv'

df = pd.read_csv(original_dataset_path)
columns = [col.strip() for col in df.columns]

def haar_lwt_1d_decompose(data, level):
    """
    Performs a 1D Haar Lifting Wavelet Transform decomposition for a given number of levels.

    The lifting scheme for Haar wavelet involves:
    1. Split: Separate the signal into even and odd indexed samples.
    2. Predict: Calculate detail coefficients (high-frequency) by subtracting
       the even samples from the odd samples.
    3. Update: Calculate approximation coefficients (low-frequency) by adding
       half of the detail coefficients to the even samples.

    Args:
        data (np.ndarray): The 1D input data array.
        level (int): The number of decomposition levels to perform.

    Returns:
        tuple: A tuple containing:
            - final_approximation_coeffs (np.ndarray): The approximation coefficients
              at the highest decomposition level. This represents the "reduced values".
            - detail_coeffs_info_list (list): A list of tuples, where each tuple contains:
                (detail_coeffs (np.ndarray), original_length_at_this_level (int))
              The list is ordered from the lowest decomposition level (finest details)
              to the highest decomposition level (coarsest details).
              The `original_length_at_this_level` is crucial for accurate reconstruction
              when dealing with signals that had odd lengths at certain stages.
    """
    # Ensure data is a float numpy array for calculations
    current_coeffs = np.array(data, dtype=float)
    
    # List to store detail coefficients and their original lengths at each level
    detail_coeffs_info_list = [] 

    # Perform decomposition for the specified number of levels
    for i in range(level):
        # Store the original length of the signal at the current level before any padding
        original_len_at_this_level = len(current_coeffs)

        # Pad the current coefficients if their length is odd.
        # This ensures that 'even' and 'odd' parts have compatible lengths.
        if original_len_at_this_level % 2 != 0:
            # Pad with a single zero at the end
            padded_coeffs = np.pad(current_coeffs, (0, 1), 'constant')
        else:
            padded_coeffs = current_coeffs

        # Step 1: Split - Separate into even and odd indexed samples
        even = padded_coeffs[::2]
        odd = padded_coeffs[1::2]

        # Step 2: Predict - Calculate detail coefficients (d_j = odd - even)
        detail = odd - even

        # Step 3: Update - Calculate approximation coefficients (s_j = even + d_j / 2)
        approximation = even + detail / 2

        # Store the detail coefficients along with the original length of the signal
        # at this level (before padding), which is needed for accurate reconstruction.
        detail_coeffs_info_list.append((detail, original_len_at_this_level))
        
        # The approximation coefficients become the input for the next decomposition level
        current_coeffs = approximation

        # Stop decomposition if the approximation coefficients become too small (e.g., single element),
        # as further decomposition is not meaningful.
        if len(current_coeffs) < 2:
            print(f"Warning: Stopped decomposition at level {i+1} because signal length became too small.")
            break

    # The final `current_coeffs` are the approximation coefficients at the highest level
    # detail_coeffs_info_list contains the detail coefficients and their original lengths for each level
    return current_coeffs, detail_coeffs_info_list

a,b = haar_lwt_1d_decompose(df[columns[2]], 2)
print(df[columns[2]].size, a.size,a, b)


