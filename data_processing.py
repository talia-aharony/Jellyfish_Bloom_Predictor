"""
Jellyfish Presence Prediction - Pure PyTorch Implementation
Using ONLY torch methods - NO TensorFlow, NO sklearn
Explicit layer definitions matching reference style
Includes baseline logistic regression model for benchmarking
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from data_loader import load_jellyfish_data

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print()

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_PROB = 0.3
NUM_EPOCHS = 100


# ============================================================================
# DATA GENERATION
# ============================================================================

class DataGenerator:
    """Generate realistic multi-modal jellyfish data"""
    
    @staticmethod
    def generate_synthetic_data(n_days=1200):
        """
        Generate synthetic multi-modal data from 12 sources:
        
        Real data sources to integrate:
        1. Meduzot Ba'am: https://doi.org/10.25607/lverbw
        2. CMEMS: https://marine.copernicus.eu/
        3. NOAA: https://www.ncei.noaa.gov/
        4. ERA5: https://cds.climate.copernicus.eu/
        5. NASA Ocean Color: https://oceancolor.gsfc.nasa.gov/
        6. Sentinel-3: https://sentinels.copernicus.eu/
        7. Twitter API: https://developer.twitter.com/
        8. Wikipedia API: https://www.mediawiki.org/
        9. Google Trends: https://trends.google.com/
        10. NewsAPI: https://newsapi.org/
        11. Israeli Meteorology: https://ims.gov.il/
        12. World Ocean Database: https://www.ncei.noaa.gov/
        """
        
        # ===== Environmental Features (5) =====
        base_sst = 20 + np.linspace(0, 8, n_days)
        sst = base_sst + np.random.normal(0, 1.2, n_days)
        sst = np.clip(sst, 18, 30)
        
        wind_speed = np.random.gamma(shape=3, scale=2, size=n_days)
        wind_speed = np.clip(wind_speed, 0, 20)
        
        wind_direction = np.random.normal(180, 70, n_days) % 360
        onshore_component = np.cos(np.radians(wind_direction - 180))
        
        pressure = np.random.normal(1013, 5, n_days)
        
        # ===== Satellite Features (3) =====
        chlorophyll = np.abs(np.random.normal(0.3, 0.15, n_days))
        secchi_depth = np.abs(np.random.normal(20, 5, n_days))
        sea_surface_height = np.random.normal(0, 0.1, n_days)
        
        # ===== Social Media Features (4) =====
        twitter_mentions = np.random.poisson(lam=5, size=n_days)
        wikipedia_edits = np.random.poisson(lam=2, size=n_days)
        news_articles = np.random.poisson(lam=1, size=n_days)
        google_trend_volume = np.random.uniform(0, 100, n_days)
        
        # ===== Oceanographic Features (6) =====
        salinity = np.random.normal(39.1, 0.2, n_days)
        current_speed = np.random.gamma(shape=2, scale=0.05, size=n_days)
        current_direction = np.random.uniform(0, 360, n_days)
        nitrate = np.random.gamma(shape=2, scale=0.5, size=n_days)
        phosphate = np.random.gamma(shape=2, scale=0.1, size=n_days)
        oxygen = np.random.normal(5.0, 0.5, n_days)
        
        # ===== Create Labels =====
        temp_effect = 1 / (1 + np.exp(-(sst - 23) / 2))
        wind_effect = np.exp(-((wind_speed - 7.5) ** 2) / 10)
        onshore_effect = np.maximum(onshore_component, 0)
        
        month = np.arange(n_days) % 365 // 30
        seasonal_effect = np.ones(n_days, dtype=float)
        seasonal_effect[month <= 5] *= 0.7
        seasonal_effect[month >= 8] *= 0.6
        
        jellyfish_prob = (0.3 * temp_effect + 0.3 * wind_effect + 
                         0.25 * onshore_effect + 0.15 * seasonal_effect)
        jellyfish_prob = np.clip(jellyfish_prob, 0, 1)
        labels = (np.random.rand(n_days) < jellyfish_prob).astype(np.float32)
        
        # ===== Combine Features =====
        features = np.column_stack([
            sst, wind_speed, wind_direction, onshore_component, pressure,
            chlorophyll, secchi_depth, sea_surface_height,
            twitter_mentions, wikipedia_edits, news_articles, google_trend_volume,
            salinity, current_speed, current_direction, nitrate, phosphate, oxygen
        ])
        
        return features.astype(np.float32), labels


# ============================================================================
# FEATURE ENGINEERING FOR BASELINE MODEL
# ============================================================================

def create_engineered_features(X, lookback=7):
    """Create engineered features for baseline logistic regression
    
    Fixed-length temporal windows with:
    - Lagged variables: current, t-1, t-2, etc.
    - Rolling averages: 7-day and 14-day windows
    - Rolling standard deviations
    
    This allows the baseline logistic regression model to capture
    temporal patterns without using RNN architecture.
    
    Args:
        X: Shape (n_samples, n_features)
        lookback: Number of days to look back
    
    Returns:
        X_engineered: Shape (n_samples - lookback + 1, engineered_dim)
    """
    
    n_samples, n_features = X.shape
    engineered_features = []
    
    for i in range(n_samples - lookback + 1):
        window = X[i:i+lookback]  # (lookback, n_features)
        
        # 1. Current values (t=0, most recent)
        current = window[-1, :]  # (n_features,)
        
        # 2. Lagged values (t-1, t-2, t-3)
        lag1 = window[-2, :] if lookback >= 2 else window[0, :]
        lag2 = window[-3, :] if lookback >= 3 else window[0, :]
        lag3 = window[-4, :] if lookback >= 4 else window[0, :]
        
        # 3. Rolling mean (7-day window)
        rolling_mean_7 = np.mean(window, axis=0)
        
        # 4. Rolling standard deviation (7-day window)
        rolling_std_7 = np.std(window, axis=0)
        
        # 5. Rolling mean (shorter 3-day if available)
        rolling_mean_3 = np.mean(window[-3:, :], axis=0) if lookback >= 3 else rolling_mean_7
        
        # 6. Change from previous day (current - lag1)
        change_1day = current - lag1
        
        # 7. Change from 3 days ago (current - lag3)
        change_3day = current - lag3
        
        # Concatenate all engineered features
        engineered = np.concatenate([
            current,           # Current values (n_features,)
            lag1,             # Lag 1 (n_features,)
            lag2,             # Lag 2 (n_features,)
            lag3,             # Lag 3 (n_features,)
            rolling_mean_7,   # 7-day rolling mean (n_features,)
            rolling_std_7,    # 7-day rolling std (n_features,)
            rolling_mean_3,   # 3-day rolling mean (n_features,)
            change_1day,      # 1-day change (n_features,)
            change_3day       # 3-day change (n_features,)
        ])
        
        engineered_features.append(engineered)
    
    return np.array(engineered_features, dtype=np.float32)


# ============================================================================
# BASELINE MODEL - MULTINOMIAL LOGISTIC REGRESSION
# ============================================================================

class BaselineLogisticRegression(nn.Module):
    """Baseline: Multinomial Logistic Regression
    
    Simple linear model for binary classification of jellyfish presence.
    Uses fixed-length temporal windows of meteorological data with:
    - Lagged variables (t, t-1, t-2)
    - Rolling averages (7-day, 14-day windows)
    
    This serves as a benchmark to evaluate neural network performance.
    A linear model can capture long-range dependencies through feature
    engineering alone, without requiring complex architectures.
    """
    
    def __init__(self, input_dim):
        super(BaselineLogisticRegression, self).__init__()
        
        # Single linear layer: logistic regression
        # Maps flattened input directly to probability of jellyfish presence
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # Flatten input to (batch_size, input_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Linear transformation followed by sigmoid
        # This is equivalent to multinomial logistic regression
        x = self.linear(x)
        return torch.sigmoid(x)


# ============================================================================
# PYTORCH MODELS - EXPLICIT LAYER DEFINITIONS
# ============================================================================

class FeedforwardNet(nn.Module):
    """Feedforward neural network for jellyfish prediction
    
    Explicit layer definition in __init__, explicit operations in forward.
    Architecture: Input → FC(128) → FC(64) → FC(32) → Output(1)
    """
    
    def __init__(self, input_dim, dropout_prob=DROPOUT_PROB):
        super(FeedforwardNet, self).__init__()
        
        # Layer 1: 128 hidden units with batch norm
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        # Layer 2: 64 hidden units with batch norm
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        # Layer 3: 32 hidden units with batch norm
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
        # Output layer
        self.fc4 = nn.Linear(32, 1)
    
    def forward(self, x):
        # Flatten input to (batch_size, input_dim)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Layer 1: FC + BatchNorm + ReLU + Dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2: FC + BatchNorm + ReLU + Dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3: FC + BatchNorm + ReLU + Dropout
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer with sigmoid for binary classification
        x = self.fc4(x)
        return torch.sigmoid(x)


class LSTMNet(nn.Module):
    """LSTM network for sequence modeling
    
    Explicit layer definition in __init__, explicit operations in forward.
    Architecture: LSTM(64) → FC(32) → Output(1)
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_prob=DROPOUT_PROB):
        super(LSTMNet, self).__init__()
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # LSTM returns (output, (h_n, c_n))
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last output from LSTM sequence
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # FC layer with ReLU activation
        x = self.fc1(last_output)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer with sigmoid
        x = self.fc2(x)
        return torch.sigmoid(x)


class GRUNet(nn.Module):
    """GRU network (faster LSTM alternative)
    
    Explicit layer definition in __init__, explicit operations in forward.
    Architecture: GRU(64) → FC(32) → Output(1)
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_prob=DROPOUT_PROB):
        super(GRUNet, self).__init__()
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # GRU returns (output, h_n)
        gru_out, h_n = self.gru(x)
        
        # Take last output from GRU sequence
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # FC layer with ReLU activation
        x = self.fc1(last_output)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer with sigmoid
        x = self.fc2(x)
        return torch.sigmoid(x)


class Conv1DNet(nn.Module):
    """1D CNN for temporal feature extraction
    
    Explicit layer definition in __init__, explicit operations in forward.
    Architecture: Conv1D(64) → Conv1D(32) → GlobalPool → FC(32) → Output(1)
    """
    
    def __init__(self, input_dim, hidden_dim=64, dropout_prob=DROPOUT_PROB):
        super(Conv1DNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        # Second convolutional block
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        # Adaptive global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim//2, 32)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Convert to (batch_size, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # First conv block: Conv → BatchNorm → ReLU → MaxPool → Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second conv block: Conv → BatchNorm → ReLU → MaxPool → Dropout
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Global average pooling to reduce to (batch_size, hidden_dim//2, 1)
        x = self.global_pool(x)
        
        # Flatten to (batch_size, hidden_dim//2)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # FC layer with ReLU activation
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer with sigmoid
        x = self.fc2(x)
        return torch.sigmoid(x)


class HybridNet(nn.Module):
    """Hybrid CNN + LSTM network
    
    Explicit layer definition in __init__, explicit operations in forward.
    Architecture: Conv1D(64) → LSTM(64) → FC(32) → Output(1)
    """
    
    def __init__(self, input_dim, hidden_dim=64, dropout_prob=DROPOUT_PROB):
        super(HybridNet, self).__init__()
        
        # CNN branch for feature extraction
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn_conv = nn.BatchNorm1d(hidden_dim)
        self.dropout_conv = nn.Dropout(p=dropout_prob)
        
        # LSTM branch for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.dropout_fc = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        # Transpose to (batch_size, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)
        
        # CNN feature extraction: Conv → BatchNorm → ReLU → Dropout
        x = self.conv1(x)
        x = self.bn_conv(x)
        x = F.relu(x)
        x = self.dropout_conv(x)
        
        # Transpose back to (batch_size, seq_len, hidden_dim) for LSTM
        x = x.transpose(1, 2)
        
        # LSTM sequence modeling: returns (output, (h_n, c_n))
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Take last LSTM output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)
        
        # FC layer with ReLU activation
        x = self.fc1(last_output)
        x = F.relu(x)
        x = self.dropout_fc(x)
        
        # Output layer with sigmoid
        x = self.fc2(x)
        return torch.sigmoid(x)


# ============================================================================
# TRAINER CLASS
# ============================================================================

class Trainer:
    """PyTorch training pipeline"""
    
    def __init__(self, model, device='cpu', learning_rate=LEARNING_RATE):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).unsqueeze(1)
            
            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Binary accuracy
            predictions = (outputs > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).unsqueeze(1)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                
                predictions = (outputs > 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def fit(self, train_loader, val_loader, epochs=NUM_EPOCHS, patience=15):
        """Train the model"""
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            self.scheduler.step(val_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(torch.load('best_model.pth'))
                    break
    
    def test(self, test_loader):
        """Test on test set and compute metrics"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                
                all_preds.extend(outputs.detach().cpu().flatten().tolist())
                all_labels.extend(y_batch.detach().cpu().flatten().tolist())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_preds_binary = (all_preds > 0.5).astype(int)
        
        # Compute metrics manually (pure torch/numpy)
        accuracy = np.mean(all_preds_binary == all_labels)
        
        # Precision
        tp = np.sum((all_preds_binary == 1) & (all_labels == 1))
        fp = np.sum((all_preds_binary == 1) & (all_labels == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Recall
        fn = np.sum((all_preds_binary == 0) & (all_labels == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # AUC-ROC
        auc = self._compute_auc(all_labels, all_preds)
        
        # Confusion matrix
        tn = np.sum((all_preds_binary == 0) & (all_labels == 0))
        cm = np.array([[tn, fp], [fn, tp]])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels
        }
    
    @staticmethod
    def _compute_auc(y_true, y_pred):
        """Compute AUC-ROC using numpy"""
        # Sort by prediction
        sorted_indices = np.argsort(-y_pred)
        y_sorted = y_true[sorted_indices]
        
        # Compute TPR and FPR
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr_list = []
        fpr_list = []
        
        for threshold in np.linspace(1, 0, 101):
            y_pred_thresholded = (y_pred >= threshold).astype(int)
            tp = np.sum((y_pred_thresholded == 1) & (y_true == 1))
            fp = np.sum((y_pred_thresholded == 1) & (y_true == 0))
            
            tpr = tp / n_pos if n_pos > 0 else 0
            fpr = fp / n_neg if n_neg > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        # Compute AUC using trapezoidal rule
        fpr_list = np.array(fpr_list)
        tpr_list = np.array(tpr_list)
        
        auc = np.trapz(tpr_list, fpr_list)
        return auc


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(trainer, model_name='Model'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    axes[0].plot(trainer.train_losses, label='Train', linewidth=2, color='steelblue')
    axes[0].plot(trainer.val_losses, label='Val', linewidth=2, color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(trainer.train_accs, label='Train', linewidth=2, color='steelblue')
    axes[1].plot(trainer.val_accs, label='Val', linewidth=2, color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{model_name} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved {model_name} training history")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("=" * 100)
    print("JELLYFISH PRESENCE PREDICTION - PURE PYTORCH IMPLEMENTATION")
    print("=" * 100)
    print()
    
    # =====================================================================
    # DATA GENERATION
    # =====================================================================
    print("1. DATA GENERATION")
    print("-" * 100)
    
    # generator = DataGenerator()
    # X, y = generator.generate_synthetic_data(n_days=1200)

    X, y = load_jellyfish_data()
    
    print(f"✓ Generated dataset shape: {X.shape}")
    print(f"✓ Jellyfish presence: {int(y.sum())} days ({y.mean()*100:.1f}%)")
    print(f"✓ Features: 18 (Environmental: 5, Satellite: 3, Social: 4, Oceanographic: 6)")
    print()
    
    # =====================================================================
    # DATA NORMALIZATION (Pure PyTorch)
    # =====================================================================
    print("2. DATA NORMALIZATION")
    print("-" * 100)
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    # Normalize using torch
    mean = X_tensor.mean(dim=0)
    std = X_tensor.std(dim=0)
    X_normalized = (X_tensor - mean) / (std + 1e-8)
    
    print(f"✓ Normalized using torch.mean() and torch.std()")
    print(f"✓ Feature mean: {mean.mean():.4f}, std: {std.mean():.4f}")
    print()
    
    # =====================================================================
    # CREATE SEQUENCES
    # =====================================================================
    print("3. CREATE SEQUENCES")
    print("-" * 100)
    
    lookback = 7
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X_normalized) - lookback + 1):
        X_sequences.append(X_normalized[i:i+lookback])
        y_sequences.append(y_tensor[i+lookback-1])
    
    X_sequences = torch.stack(X_sequences)
    y_sequences = torch.stack(y_sequences)
    
    print(f"✓ Created sequences with lookback={lookback}")
    print(f"✓ Sequence shape: {X_sequences.shape}")
    print()
    
    # =====================================================================
    # CREATE DATASET AND DATALOADERS
    # =====================================================================
    print("4. CREATE DATALOADERS")
    print("-" * 100)
    
    # Create TensorDataset
    dataset = TensorDataset(X_sequences, y_sequences)
    
    # Split dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Train set: {len(train_dataset)} samples")
    print(f"✓ Val set: {len(val_dataset)} samples")
    print(f"✓ Test set: {len(test_dataset)} samples")
    print()
    
    # =====================================================================
    # TRAIN MODELS
    # =====================================================================
    print("5. TRAINING MODELS")
    print("-" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Store all model results
    results = {}
    
    # =====================================================================
    # BASELINE MODEL - LOGISTIC REGRESSION
    # =====================================================================
    print("TRAINING BASELINE MODEL (LOGISTIC REGRESSION)")
    print("-" * 50)
    
    # Create engineered features for baseline
    X_engineered = create_engineered_features(X, lookback=7)
    
    # Adjust y to match engineered features length
    y_baseline = y[6:]  # Skip first 6 samples due to feature engineering
    
    # Normalize engineered features
    X_eng_tensor = torch.FloatTensor(X_engineered)
    mean_eng = X_eng_tensor.mean(dim=0)
    std_eng = X_eng_tensor.std(dim=0)
    X_eng_normalized = (X_eng_tensor - mean_eng) / (std_eng + 1e-8)
    y_eng_tensor = torch.FloatTensor(y_baseline)
    
    # Create dataset
    baseline_dataset = TensorDataset(X_eng_normalized, y_eng_tensor)
    
    # Split dataset
    train_size_baseline = int(0.7 * len(baseline_dataset))
    val_size_baseline = int(0.15 * len(baseline_dataset))
    test_size_baseline = len(baseline_dataset) - train_size_baseline - val_size_baseline
    
    train_dataset_bl, val_dataset_bl, test_dataset_bl = random_split(
        baseline_dataset,
        [train_size_baseline, val_size_baseline, test_size_baseline],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader_bl = DataLoader(train_dataset_bl, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_bl = DataLoader(val_dataset_bl, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_bl = DataLoader(test_dataset_bl, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train baseline
    baseline_model = BaselineLogisticRegression(input_dim=X_eng_normalized.shape[1])
    baseline_trainer = Trainer(baseline_model, device=device, learning_rate=LEARNING_RATE)
    
    start_time = time.time()
    baseline_trainer.fit(train_loader_bl, val_loader_bl, epochs=NUM_EPOCHS, patience=15)
    baseline_time = time.time() - start_time
    
    # Test baseline
    baseline_metrics = baseline_trainer.test(test_loader_bl)
    
    print(f"\nBaseline (Logistic Regression) Results:")
    print(f"  Accuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall: {baseline_metrics['recall']:.4f}")
    print(f"  F1-Score: {baseline_metrics['f1']:.4f}")
    print(f"  AUC-ROC: {baseline_metrics['auc']:.4f}")
    print(f"  Train time: {baseline_time:.1f}s")
    print(f"  Engineered features: {X_eng_normalized.shape[1]}")
    print(f"  Model parameters: {sum(p.numel() for p in baseline_model.parameters()):,}")
    
    # Plot baseline
    plot_training_history(baseline_trainer, 'Baseline_LogisticRegression')
    
    # Store baseline results
    results['Baseline (Logistic Regression)'] = {
        'accuracy': baseline_metrics['accuracy'],
        'precision': baseline_metrics['precision'],
        'recall': baseline_metrics['recall'],
        'f1': baseline_metrics['f1'],
        'auc': baseline_metrics['auc'],
        'params': sum(p.numel() for p in baseline_model.parameters()),
        'time': baseline_time
    }
    
    print()
    
    # =====================================================================
    # NEURAL NETWORK MODELS
    # =====================================================================
    sequence_length = X_sequences.shape[1]
    feature_dim = X_sequences.shape[2]

    models = {
        'Feedforward': FeedforwardNet(input_dim=feature_dim * sequence_length, dropout_prob=DROPOUT_PROB),
        'LSTM': LSTMNet(input_dim=feature_dim, hidden_dim=64, dropout_prob=DROPOUT_PROB),
        'GRU': GRUNet(input_dim=feature_dim, hidden_dim=64, dropout_prob=DROPOUT_PROB),
        'Conv1D': Conv1DNet(input_dim=feature_dim, hidden_dim=64, dropout_prob=DROPOUT_PROB),
        'Hybrid': HybridNet(input_dim=feature_dim, hidden_dim=64, dropout_prob=DROPOUT_PROB)
    }
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        print("-" * 50)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        # Train
        trainer = Trainer(model, device=device, learning_rate=LEARNING_RATE)
        start_time = time.time()
        trainer.fit(train_loader, val_loader, epochs=NUM_EPOCHS, patience=15)
        train_time = time.time() - start_time
        
        # Test
        test_metrics = trainer.test(test_loader)
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")
        print(f"  F1-Score: {test_metrics['f1']:.4f}")
        print(f"  AUC-ROC: {test_metrics['auc']:.4f}")
        print(f"  Train time: {train_time:.1f}s")
        
        # Save results
        results[model_name] = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'params': num_params,
            'time': train_time
        }
        
        # Plot
        plot_training_history(trainer, model_name)
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY - BASELINE vs NEURAL NETWORKS")
    print("=" * 100)
    print()
    
    print(f"{'Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12} {'Params':<12}")
    print("-" * 115)
    
    # Sort results: baseline first, then neural networks
    sorted_results = sorted(results.items(), key=lambda x: (x[0] != 'Baseline (Logistic Regression)', x[0]))
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<35} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f} {metrics['params']:<12,}")
    
    # Calculate improvements
    baseline_acc = results['Baseline (Logistic Regression)']['accuracy']
    baseline_f1 = results['Baseline (Logistic Regression)']['f1']
    baseline_auc = results['Baseline (Logistic Regression)']['auc']
    
    print("\n" + "=" * 100)
    print("NEURAL NETWORK IMPROVEMENTS OVER BASELINE")
    print("=" * 100)
    print()
    
    for model_name, metrics in sorted_results:
        if model_name != 'Baseline (Logistic Regression)':
            acc_improvement = (metrics['accuracy'] - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
            f1_improvement = (metrics['f1'] - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
            auc_improvement = (metrics['auc'] - baseline_auc) / baseline_auc * 100 if baseline_auc > 0 else 0
            
            print(f"{model_name:<35}")
            print(f"  Accuracy improvement: {acc_improvement:+.2f}%")
            print(f"  F1-Score improvement: {f1_improvement:+.2f}%")
            print(f"  AUC-ROC improvement: {auc_improvement:+.2f}%")
            print()
    
    print("\n✓ Pure PyTorch implementation with baseline model complete!")
    print("=" * 100)