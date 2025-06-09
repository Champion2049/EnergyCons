import torch
import torch.nn as nn
import torch.nn.functional as F # Import F for functional API like F.relu
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import sys
import numpy as np
import yaml # Import yaml for loading settings.yaml
import time # Import the time module for tracking training duration

# Import the dataset class from the dataset.py file
# Ensure that 'dataset.py' is in the same directory or accessible via PYTHONPATH.
try:
    from dataset import InMemoryKoreaDataset # Now importing InMemoryKoreaDataset
except ImportError:
    print("Error: Could not import InMemoryKoreaDataset from dataset.py.")
    print("Please ensure dataset.py is in the same directory as this script, or configured in PYTHONPATH.")
    sys.exit(1)


class CWNN(nn.Module):
    """
    Continuous Wavelet Neural Network (CWNN) model.
    This implementation approximates the "time-frequency feature extraction"
    aspect of a CWNN using 1D convolutional layers, followed by
    standard fully connected (Back Propagation) layers for regression.
    """
    def __init__(self, input_window_size, hidden_size=128, output_size=1):
        """
        Initializes the CWNN model.

        Args:
            input_window_size (int): The length of the input time series window.
            hidden_size (int): The number of neurons in the hidden fully connected layer.
            output_size (int): The number of output values (1 for single appliance power).
        """
        super(CWNN, self).__init__()

        # Feature extraction layers (approximating wavelet-like filters)
        # Conv1d expects input in shape (batch_size, in_channels, sequence_length)
        # Here, in_channels=1 as we're processing a single time series (mains).
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        # MaxPool1d reduces the sequence length, helping to capture features at different scales.
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # Calculate the size of the features after convolutional and pooling layers
        # This is crucial for defining the first fully connected layer.
        # We pass a dummy tensor to dynamically compute the flattened size.
        with torch.no_grad(): # Ensure no gradients are computed for this dummy pass
            dummy_input = torch.randn(1, 1, input_window_size)
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2(F.relu(self.conv2(x)))
            flattened_size = x.numel() # Total number of elements in the flattened tensor

        # Fully connected layers (mimicking the Back Propagation network)
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.dropout = nn.Dropout(0.5) # Dropout for regularization to prevent overfitting
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer for regression

    def forward(self, x):
        """
        Defines the forward pass of the CWNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_window_size).

        Returns:
            torch.Tensor: Output tensor of predicted power consumption,
                          shape (batch_size,).
        """
        # Add a channel dimension for Conv1d: (batch_size, 1, input_window_size)
        x = x.unsqueeze(1)

        # Apply convolutional and pooling layers
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Apply fully connected layers with ReLU activation and dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)

        # Squeeze the output to remove the last dimension if output_size is 1,
        # resulting in (batch_size,).
        return x.squeeze(1)


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs=10,
    device='cpu'
):
    """
    Trains and validates the CWNN model.

    Args:
        model (nn.Module): The CWNN model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (nn.Module): Loss function (e.g., MSELoss).
        num_epochs (int): Number of training epochs.
        device (str): Device to run training on ('cpu' or 'cuda').
    """
    model.to(device) # Move model to the specified device

    start_time_overall = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        for i, (inputs, targets, _) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}")

        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            if len(val_loader) > 0:
                for batch_idx, (inputs, targets, _) in enumerate(val_loader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)

                    loss = criterion(outputs, targets)
                    if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                        print(f"  [Validation Error] Loss is NaN/Inf in validation batch {batch_idx}. Exiting validation for this epoch.")
                        print(f"    Inputs shape: {inputs.shape}, has NaN: {torch.isnan(inputs).any().item()}, has Inf: {torch.isinf(inputs).any().item()}, Min={inputs.min().item():.4f}, Max={inputs.max().item():.4f}")
                        print(f"    Targets shape: {targets.shape}, has NaN: {torch.isnan(targets).any().item()}, has Inf: {torch.isinf(targets).any().item()}, Min={targets.min().item():.4f}, Max={targets.max().item():.4f}")
                        print(f"    Outputs shape: {outputs.shape}, has NaN: {torch.isnan(outputs).any().item()}, has Inf: {torch.isinf(outputs).any().item()}, Min={outputs.min().item():.4f}, Max={outputs.max().item():.4f}")
                        val_running_loss = float('nan')
                        break
                    val_running_loss += loss.item() * inputs.size(0)

                if not np.isnan(val_running_loss):
                    val_loss = val_running_loss / len(val_loader.dataset)
                    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: nan (due to NaN/Inf in batch loss)")

            else:
                print(f"Epoch {epoch+1}/{num_epochs}, No validation data available.")

        epoch_end_time = time.time()
        time_elapsed_this_epoch = epoch_end_time - epoch_start_time
        time_per_epoch_avg = (epoch_end_time - start_time_overall) / (epoch + 1)
        
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_time_remaining_seconds = remaining_epochs * time_per_epoch_avg
        
        m, s = divmod(int(estimated_time_remaining_seconds), 60)
        h, m = divmod(m, 60)
        
        print(f"Time elapsed for epoch: {time_elapsed_this_epoch:.2f}s | Estimated time remaining: {h:02d}h {m:02d}m {s:02d}s")


    print("Training complete.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python cwnn_model_and_train.py <path_to_settings_yaml> <path_to_dataset_folder> <appliance_key_in_settings>")
        sys.exit(1)

    settings_yaml_path = sys.argv[1]
    data_folder_path = sys.argv[2]
    appliance_key = sys.argv[3]

    try:
        with open(settings_yaml_path, 'r') as f:
            settings = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: settings.yaml not found at {settings_yaml_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing settings.yaml: {e}")
        sys.exit(1)

    hparams_global = settings.get('hparams', {})
    GLOBAL_LR = hparams_global.get('lr', 0.001)
    GLOBAL_BATCH_SIZE = hparams_global.get('batch_size', 64)
    GLOBAL_EPOCHS = hparams_global.get('epochs', 5)

    appliances_config = settings.get('appliances', {})
    if appliance_key not in appliances_config:
        print(f"Error: Appliance '{appliance_key}' not found in settings.yaml under 'appliances'.")
        print(f"Available appliances: {list(appliances_config.keys())}")
        sys.exit(1)

    appliance_settings = appliances_config[appliance_key]

    model_hparams = appliance_settings.get('hparams', {})
    INPUT_WINDOW_SIZE = model_hparams.get('L', 496)
    HIDDEN_SIZE = model_hparams.get('H', 128)
    OUTPUT_SIZE = 1

    train_buildings = appliance_settings['buildings']['train']
    test_buildings = appliance_settings['buildings']['test']

    normalization_enabled = appliance_settings.get('normalization', False)
    active_threshold = appliance_settings.get('active_threshold', None)

    print(f"--- Configuration for {appliance_key} ---")
    print(f"Input Window Size (L): {INPUT_WINDOW_SIZE}")
    print(f"Hidden Size (H): {HIDDEN_SIZE}")
    print(f"Normalization Enabled: {normalization_enabled}")
    print(f"Active Threshold: {active_threshold}")
    print(f"Training Buildings: {train_buildings}")
    print(f"Test Buildings: {test_buildings}")
    print(f"Global Learning Rate: {GLOBAL_LR}")
    print(f"Global Batch Size: {GLOBAL_BATCH_SIZE}")
    print(f"Global Epochs: {GLOBAL_EPOCHS}")
    print("---------------------------------")

    print(f"Loading training data for appliance: {appliance_key} from training buildings...")
    try:
        base_appliance_name = appliance_key.split('-')[0]
        train_full_dataset = InMemoryKoreaDataset(
            path=data_folder_path,
            buildings=train_buildings,
            appliance=base_appliance_name,
            windowsize=INPUT_WINDOW_SIZE,
            normalization_enabled=normalization_enabled,
            active_threshold=active_threshold
        )
    except ValueError as ve:
        print(f"Training dataset loading error: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during training dataset loading: {e}")
        sys.exit(1)

    if len(train_full_dataset) == 0:
        print("No training data loaded. Please check your dataset path and building configurations.")
        sys.exit(1)

    transform_params = train_full_dataset.transform_params

    print(f"Loading test data for appliance: {appliance_key} from test buildings...")
    try:
        val_dataset = InMemoryKoreaDataset(
            path=data_folder_path,
            buildings=test_buildings,
            appliance=base_appliance_name,
            windowsize=INPUT_WINDOW_SIZE,
            normalization_enabled=normalization_enabled,
            transform_params=transform_params,
            active_threshold=active_threshold
        )
    except ValueError as ve:
        print(f"Validation dataset loading error: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during validation dataset loading: {e}")
        sys.exit(1)

    if len(val_dataset) == 0:
        print("No validation data loaded. This may be expected if only one house is used for testing. Training will proceed without validation.")
        val_loader = DataLoader([], batch_size=GLOBAL_BATCH_SIZE)
    else:
        val_loader = DataLoader(val_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=False)

    train_loader = DataLoader(train_full_dataset, batch_size=GLOBAL_BATCH_SIZE, shuffle=True)

    print(f"Total training samples: {len(train_full_dataset)}")
    print(f"Total validation samples: {len(val_dataset)}")

    model = CWNN(input_window_size=INPUT_WINDOW_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=GLOBAL_LR)
    criterion = nn.MSELoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if len(train_loader) > 0:
        train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=GLOBAL_EPOCHS, device=device)
    else:
        print("Skipping training as training dataset is empty.")