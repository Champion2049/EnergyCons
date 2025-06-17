import torch
import torch.nn as nn
import torch.nn.functional as F

class LWT1D(nn.Module):
    def __init__(self):
        super(LWT1D, self).__init__()
    def forward(self, x):
        # Ensure input is 2D (batch_size, sequence_length)
        if x.dim() == 3 and x.size(1) == 1: # Handle (batch_size, 1, sequence_length) from previous unsqueeze
            x = x.squeeze(1)
        if x.dim() != 2:
            raise ValueError(f"Input to LWT1D must be 2D (batch_size, sequence_length), but got shape {x.shape}")

        N = x.size(1)

        if N % 2 != 0:
            x = x[:, :-1]
            N = x.size(1) # Update N after truncation

        # 1. Split Phase (Lazy Wavelet Transform):
        # x_e[n] = x[2n] (even-indexed samples)
        # x_o[n] = x[2n+1] (odd-indexed samples)
        even_samples = x[:, 0::2] # All samples at even indices
        odd_samples = x[:, 1::2]  # All samples at odd indices

        # 2. Predict Phase (Dual Lifting):
        # For Haar-like: P(x_e[n]) = x_e[n]
        # d[n] = x_o[n] - P(x_e[n])
        detail_coeffs = odd_samples - even_samples

        # 3. Update Phase (Primal Lifting):
        # For Haar-like: U(d[n]) = d[n]/2
        # c[n] = x_e[n] + U(d[n])
        approximation_coeffs = even_samples + 0.5 * detail_coeffs # Using 0.5 for Haar-like update

        return approximation_coeffs, detail_coeffs

class InverseLWT1D(nn.Module):
    def __init__(self):
        super(InverseLWT1D, self).__init__()

    def forward(self, approx_coeffs, detail_coeffs):
        # 1. Inverse Update Step:
        # x_e[n] = c[n] - U(d[n])
        even_samples_reconstructed = approx_coeffs - 0.5 * detail_coeffs

        # 2. Inverse Predict Step:
        # x_o[n] = d[n] + P(x_e[n])
        odd_samples_reconstructed = detail_coeffs + even_samples_reconstructed

        # 3. Merge Phase:
        N_half = even_samples_reconstructed.size(1)
        reconstructed_signal = torch.empty(
            even_samples_reconstructed.size(0),
            N_half * 2,
            dtype=even_samples_reconstructed.dtype,
            device=even_samples_reconstructed.device
        )
        reconstructed_signal[:, 0::2] = even_samples_reconstructed
        reconstructed_signal[:, 1::2] = odd_samples_reconstructed

        return reconstructed_signal

class LWTBasedModel(nn.Module):
    """
    Lifting Wavelet Transform (LWT) based neural network model for regression.
    This model uses LWT layers for feature extraction, replacing the
    convolutional layers found in a CWNN. The extracted LWT coefficients
    are then fed into fully connected layers for the regression task.
    """
    def __init__(self, input_window_size, hidden_size=128, output_size=1, num_lwt_levels=2):
        super(LWTBasedModel, self).__init__()

        self.num_lwt_levels = num_lwt_levels
        # Create a list of LWT1D layers for multi-level decomposition
        self.lwt_layers = nn.ModuleList([LWT1D() for _ in range(num_lwt_levels)])

        # This determines the input size for the first fully connected layer.
        # After each LWT level, the approximation coefficients are passed to the next level, and the detail coefficients are collected. The final approximation coefficients are also part of the collected features.
        current_len = input_window_size
        total_coeffs_len = 0
        for _ in range(self.num_lwt_levels):
            if current_len % 2 != 0:
                current_len -= 1 # Handle potential odd length from previous level
            approx_len = current_len // 2
            detail_len = current_len // 2
            total_coeffs_len += detail_len # Sum up detail coeffs length from each level
            current_len = approx_len # The approximation coeffs become input for the next level
        total_coeffs_len += current_len # Add the length of the final approximation coeffs

        print(f"Calculated total flattened size of LWT coefficients: {total_coeffs_len}")

        # Fully connected layers (mimicking the Back Propagation network)
        self.fc1 = nn.Linear(total_coeffs_len, hidden_size)
        self.dropout = nn.Dropout(0.5) # Dropout for regularization
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer for regression

    def forward(self, x):
        all_coeffs = []
        current_signal = x

        # Apply LWT decomposition level by level
        for i in range(self.num_lwt_levels):
            # Pass the current signal (initially input x, then approximation coeffs) to the next LWT layer.
            approx_coeffs, detail_coeffs = self.lwt_layers[i](current_signal)
            # Collect the detail coefficients from each level.
            all_coeffs.append(detail_coeffs)
            # The approximation coefficients become the input for the next level.
            current_signal = approx_coeffs
            
        all_coeffs.append(current_signal)
        # Concatenate all collected coefficients (approximation and details) to form a single feature vector for the fully connected layers. Ensure each coefficient tensor is flattened before concatenation.
        x_flattened = torch.cat([c.view(c.size(0), -1) for c in all_coeffs], dim=1)
        # Apply fully connected layers with ReLU activation and dropout
        x = self.fc1(x_flattened)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # Squeeze the output to remove the last dimension if output_size is 1, resulting in (batch_size,).
        return x.squeeze(1)
