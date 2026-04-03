"""
Neural Network and Baseline Models for Jellyfish Forecasting

All models are implemented with explicit layer definitions in PyTorch.
Models expect input shape: (batch_size, 7, 11) for sequences or (batch_size, engineered_dim) for baseline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineLogisticRegression(nn.Module):
    """Baseline: Logistic Regression for Time Series Forecasting
    
    Simple linear model for binary forecasting using engineered temporal features.
    Uses engineered temporal features from 7-day historical windows:
    - Recent values, trends, volatility, min/max, changes and slopes
    
    This is a strong baseline for temporal data that can learn seasonal patterns
    via feature engineering without requiring RNN architecture.
    
    Input: (batch_size, engineered_features) where engineered_features = 11 * 10 = 110
    Output: (batch_size, 1) - probability of jellyfish presence
    """
    
    def __init__(self, input_dim):
        super(BaselineLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten all features
        x = self.linear(x)
        return torch.sigmoid(x)


class FeedforwardNet(nn.Module):
    """Feedforward Network for Time Series Forecasting
    
    Flattens temporal sequences and uses dense layers.
    
    Input: (batch_size, 7, 11) - 7 days × 11 features = 77 values per sample
    Output: (batch_size, 1) - probability of jellyfish presence
    """
    
    def __init__(self, input_dim, dropout_prob=0.3):
        super(FeedforwardNet, self).__init__()
        
        # Layer 1: 77 → 128 with BatchNorm
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        # Layer 2: 128 → 64 with BatchNorm
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        # Layer 3: 64 → 32 with BatchNorm
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
        # Output: 32 → 1
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten (batch, 7, 11) → (batch, 77)
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc4(x)
        return torch.sigmoid(x)


class LSTMNet(nn.Module):
    """LSTM Network for Temporal Forecasting
    
    Processes sequences naturally while capturing long-range dependencies.
    
    Input: (batch_size, 7 days, 11 features)
    Output: (batch_size, 1) - probability of jellyfish presence
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_prob=0.3):
        super(LSTMNet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, 7, 11)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last LSTM output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        x = self.fc1(last_output)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class GRUNet(nn.Module):
    """GRU Network for Temporal Forecasting (faster LSTM alternative)
    
    Similar to LSTM but with fewer parameters and faster training.
    
    Input: (batch_size, 7 days, 11 features)
    Output: (batch_size, 1) - probability of jellyfish presence
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_prob=0.3):
        super(GRUNet, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, 7, 11)
        gru_out, h_n = self.gru(x)
        
        # Use last GRU output
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_dim)
        
        x = self.fc1(last_output)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class Conv1DNet(nn.Module):
    """1D CNN for Temporal Forecasting
    
    Applies convolutional filters over time to detect temporal patterns.
    Useful for capturing local patterns and trends.
    
    Input: (batch_size, 7 days, 11 features)
    Output: (batch_size, 1) - probability of jellyfish presence
    """
    
    def __init__(self, input_dim, hidden_dim=64, dropout_prob=0.3):
        super(Conv1DNet, self).__init__()
        
        # First conv block: 11 features → 64 filters
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        # Second conv block: 64 filters → 32 filters
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim//2, 32)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, 7, 11)
        x = x.transpose(1, 2)  # (batch_size, 11, 7) - 11 features, 7 time steps
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Flatten
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        return torch.sigmoid(x)


class HybridNet(nn.Module):
    """Hybrid CNN + LSTM for Temporal Forecasting
    
    Combines CNN feature extraction with LSTM sequence modeling:
    - CNN extracts local temporal patterns across features
    - LSTM captures long-range temporal dependencies
    
    Input: (batch_size, 7 days, 11 features)
    Output: (batch_size, 1) - probability of jellyfish presence
    """
    
    def __init__(self, input_dim, hidden_dim=64, dropout_prob=0.3):
        super(HybridNet, self).__init__()
        
        # CNN: Extract temporal patterns from each feature
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn_conv = nn.BatchNorm1d(hidden_dim)
        self.dropout_conv = nn.Dropout(p=dropout_prob)
        
        # LSTM: Model sequence of extracted features
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Dense: Final prediction
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout_fc = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, 7, 11)
        x = x.transpose(1, 2)  # (batch_size, 11, 7)
        
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn_conv(x)
        x = F.relu(x)
        x = self.dropout_conv(x)
        
        # Back to sequence format for LSTM
        x = x.transpose(1, 2)  # (batch_size, 7, hidden_dim)
        
        # LSTM sequence modeling
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        
        # Dense layers
        x = self.fc1(last_output)
        x = F.relu(x)
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return torch.sigmoid(x)
