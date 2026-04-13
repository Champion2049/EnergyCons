import pandas as pd
import numpy as np
import re
import os

def haar_lwt_1d_decompose(data, level):
    """
    Performs a 1D Haar Lifting Wavelet Transform decomposition for a given number of levels.
    (Code provided by user)
    """
    current_coeffs = np.array(data, dtype=float)
    detail_coeffs_info_list = [] 

    for i in range(level):
        original_len_at_this_level = len(current_coeffs)

        if original_len_at_this_level % 2 != 0:
            padded_coeffs = np.pad(current_coeffs, (0, 1), 'constant')
        else:
            padded_coeffs = current_coeffs

        even = padded_coeffs[::2]
        odd = padded_coeffs[1::2]

        detail = odd - even
        approximation = even + detail / 2

        detail_coeffs_info_list.append((detail, original_len_at_this_level))
        current_coeffs = approximation

        if len(current_coeffs) < 2:
            print(f"Warning: Stopped decomposition at level {i+1} because signal length became too small.")
            break

    return current_coeffs, detail_coeffs_info_list

def process_dataset_with_lwt(input_csv, output_csv, level=1):
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # 1. Separate columns into metadata, target, and weather features
    metadata_cols = ['Year', 'Yield']
    district_cols = [col for col in df.columns if col.startswith('District_')]
    non_weather_cols = metadata_cols + district_cols
    
    # 2. Automatically group weather columns by their prefix (e.g., 'Tmax', 'Rain')
    weather_cols = [col for col in df.columns if col not in non_weather_cols]
    
    col_groups = {}
    for col in weather_cols:
        # Match the text part (e.g., 'Tmax') and number part (e.g., '31')
        match = re.match(r'([a-zA-Z]+)(\d+)', col)
        if match:
            prefix = match.group(1)
            if prefix not in col_groups:
                col_groups[prefix] = []
            col_groups[prefix].append(col)
            
    print("Found the following feature groups to decompose:")
    for prefix, cols in col_groups.items():
        print(f"  - {prefix}: {len(cols)} weeks")
        
    # 3. Apply LWT row by row
    print(f"\nApplying Level {level} 1D Haar LWT...")
    new_rows = []
    
    for idx, row in df.iterrows():
        new_row = {}
        
        # Copy over Year, Yield, and District columns unaltered
        for col in non_weather_cols:
            new_row[col] = row[col]
            
        # Process each weather group
        for prefix, cols in col_groups.items():
            # Extract the 1D sequence for this specific row and weather parameter
            data_sequence = row[cols].values
            
            # Apply the transform
            cA, cD_info = haar_lwt_1d_decompose(data_sequence, level)
            
            # Since level=1, the detail coefficients are in the first tuple of the list
            cD = cD_info[0][0]
            
            # Save Approximation (cA) features (Low-frequency / Base trends)
            for i, val in enumerate(cA):
                new_row[f'{prefix}_cA_{i+1}'] = val
                
            # Save Detail (cD) features (High-frequency / Weather shocks)
            for i, val in enumerate(cD):
                new_row[f'{prefix}_cD_{i+1}'] = val
                
        new_rows.append(new_row)
        
    # 4. Create new DataFrame and save
    df_lwt = pd.DataFrame(new_rows)
    df_lwt.to_csv(output_csv, index=False)
    
    print(f"\nSuccess! LWT Preprocessed data saved to: {output_csv}")
    print(f"Original shape: {df.shape}")
    print(f"New shape:      {df_lwt.shape}")

if __name__ == "__main__":
    INPUT_FILE = 'Wavelet-LWT-Rtamanyu/maams_preprocessed_yield_weather.csv'
    OUTPUT_FILE = 'Wavelet-LWT-Rtamanyu/lwt_level_2_preprocessed_yield_weather.csv'
    
    if os.path.exists(INPUT_FILE):
        process_dataset_with_lwt(INPUT_FILE, OUTPUT_FILE, level=2)
    else:
        print(f"Error: {INPUT_FILE} not found. Please ensure the file is in the same directory.")