import torch
import torch.nn as nn
import torch.nn.functional as F

class CWNN(nn.Module):
    def __init__(self, input_window_size, hidden_size=128, output_size=1):

        super(CWNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        with torch.no_grad(): # Ensure no gradients are computed for this dummy pass
            dummy_input = torch.randn(1, 1, input_window_size)
            x = self.pool1(F.relu(self.conv1(dummy_input)))
            x = self.pool2(F.relu(self.conv2(x)))
            flattened_size = x.numel() # Total number of elements in the flattened tensor
        self.fc1 = nn.Linear(flattened_size, hidden_size)
        self.dropout = nn.Dropout(0.5) # Dropout for regularization to prevent overfitting
        self.fc2 = nn.Linear(hidden_size, output_size) # Output layer for regression

    def forward(self, x):
        # Channel dimension for Conv1d: (batch_size, 1, input_window_size)
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

        # Squeeze the output to remove the last dimension if output_size is 1, resulting in (batch_size,).
        return x.squeeze(1)
