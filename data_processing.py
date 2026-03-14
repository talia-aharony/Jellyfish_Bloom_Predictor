"""
Jellyfish Presence Prediction - Pure PyTorch Forecasting Implementation
FORECASTING TASK: Predict jellyfish per beach per day based on 7-day history
Using ONLY torch methods - NO TensorFlow, NO sklearn
Explicit layer definitions with baseline logistic regression
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
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
# FEATURE ENGINEERING FOR BASELINE MODEL - TIME SERIES FORECASTING
# ============================================================================

def create_engineered_features_forecasting(X, lookback=7):
    """Create engineered features for baseline logistic regression
    
    Takes historical sequences and extracts temporal features:
    - Latest values (most recent day)
    - Temporal trends (linear slope over lookback period)
    - Volatility (standard deviation over period)
    - Lagged differences (day-to-day changes)
    
    This allows logistic regression to capture temporal patterns without RNNs.
    
    Args:
        X: Shape (n_samples, lookback, n_features)
        lookback: Number of days in lookback window (typically 7)
    
    Returns:
        X_engineered: Shape (n_samples, engineered_dim)
    """
    
    n_samples, lookback, n_features = X.shape
    engineered_features = []
    
    for sample_idx in range(n_samples):
        seq = X[sample_idx]  # (lookback, n_features) - 7 days of data
        
        features_for_sample = []
        
        for feat_idx in range(n_features):
            time_series = seq[:, feat_idx]  # (lookback,) - values over 7 days
            
            # 1. Current value (most recent day)
            current = time_series[-1]
            features_for_sample.append(current)
            
            # 2. Previous day (t-1)
            prev = time_series[-2] if lookback >= 2 else time_series[0]
            features_for_sample.append(prev)
            
            # 3. 3 days ago (t-3)
            prev_3 = time_series[-4] if lookback >= 4 else time_series[0]
            features_for_sample.append(prev_3)
            
            # 4. Trend (linear slope)
            x_vals = np.arange(lookback)
            coeffs = np.polyfit(x_vals, time_series, 1)
            trend = coeffs[0]  # Slope
            features_for_sample.append(trend)
            
            # 5. Mean over lookback period
            mean_val = np.mean(time_series)
            features_for_sample.append(mean_val)
            
            # 6. Std over lookback period (volatility)
            std_val = np.std(time_series)
            features_for_sample.append(std_val)
            
            # 7. Min/Max over period
            min_val = np.min(time_series)
            max_val = np.max(time_series)
            features_for_sample.append(min_val)
            features_for_sample.append(max_val)
            
            # 8. 1-day change (current - prev)
            change_1day = current - prev
            features_for_sample.append(change_1day)
            
            # 9. 3-day change (current - 3-day-ago)
            change_3day = current - prev_3
            features_for_sample.append(change_3day)
        
        engineered_features.append(features_for_sample)
    
    return np.array(engineered_features, dtype=np.float32)


# ============================================================================
# BASELINE MODEL - MULTINOMIAL LOGISTIC REGRESSION
# ============================================================================

class BaselineLogisticRegression(nn.Module):
    """Baseline: Logistic Regression for Time Series Forecasting
    
    Simple linear model for binary forecasting.
    Uses engineered temporal features from 7-day historical windows:
    - Recent values, trends, volatility, min/max
    - Changes and slopes
    
    This is a strong baseline for temporal data:
    - Can learn seasonal patterns (via engineered features)
    - Captures trend information
    - No temporal architecture needed
    
    Prediction: Given 7 days of beach observations → predict tomorrow's jellyfish
    """
    
    def __init__(self, input_dim):
        super(BaselineLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten all features
        x = self.linear(x)
        return torch.sigmoid(x)


# ============================================================================
# PYTORCH MODELS - EXPLICIT LAYER DEFINITIONS FOR TIME SERIES
# ============================================================================

class FeedforwardNet(nn.Module):
    """Feedforward network for time series forecasting
    
    Flattens temporal sequences and uses dense layers.
    Input: 7 days × 11 features = 77 values per sample
    """
    
    def __init__(self, input_dim, dropout_prob=DROPOUT_PROB):
        super(FeedforwardNet, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        
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
    """LSTM for temporal forecasting
    
    Processes sequence naturally while capturing long-range dependencies.
    Input: (batch, 7 days, 11 features)
    Output: Binary prediction for next day
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_prob=DROPOUT_PROB):
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
    """GRU for temporal forecasting (faster LSTM alternative)
    
    Similar to LSTM but with fewer parameters and faster training.
    """
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout_prob=DROPOUT_PROB):
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
    """1D CNN for temporal forecasting
    
    Applies convolutional filters over time to detect temporal patterns.
    Useful for capturing local patterns and trends.
    """
    
    def __init__(self, input_dim, hidden_dim=64, dropout_prob=DROPOUT_PROB):
        super(Conv1DNet, self).__init__()
        
        # Input: (batch, 11, 7) after transpose
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(hidden_dim//2, 32)
        self.dropout3 = nn.Dropout(p=dropout_prob)
        self.fc2 = nn.Linear(32, 1)
    
    def forward(self, x):
        # x shape: (batch_size, 7, 11)
        x = x.transpose(1, 2)  # (batch_size, 11, 7) - 11 features, 7 time steps
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        x = self.global_pool(x)
        
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        return torch.sigmoid(x)


class HybridNet(nn.Module):
    """Hybrid CNN + LSTM for temporal forecasting
    
    Combines CNN feature extraction with LSTM sequence modeling:
    - CNN extracts local temporal patterns across features
    - LSTM captures long-range temporal dependencies
    """
    
    def __init__(self, input_dim, hidden_dim=64, dropout_prob=DROPOUT_PROB):
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
        
        # Back to sequence format
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


# ============================================================================
# TRAINER CLASS
# ============================================================================

class Trainer:
    """PyTorch training pipeline for time series forecasting"""
    
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
            
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
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
        """Test on test set"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)
                
                all_preds.extend(outputs.detach().cpu().view(-1).tolist())
                all_labels.extend(y_batch.detach().cpu().view(-1).tolist())
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_preds_binary = (all_preds > 0.5).astype(int)
        
        accuracy = np.mean(all_preds_binary == all_labels)
        
        tp = np.sum((all_preds_binary == 1) & (all_labels == 1))
        fp = np.sum((all_preds_binary == 1) & (all_labels == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        fn = np.sum((all_preds_binary == 0) & (all_labels == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        auc = self._compute_auc(all_labels, all_preds)
        
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
        """Compute AUC-ROC"""
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr_list = []
        fpr_list = []
        
        for threshold in np.linspace(1, 0, 101):
            y_pred_thresh = (y_pred >= threshold).astype(int)
            tp = np.sum((y_pred_thresh == 1) & (y_true == 1))
            fp = np.sum((y_pred_thresh == 1) & (y_true == 0))
            
            tpr = tp / n_pos
            fpr = fp / n_neg
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        return np.trapz(tpr_list, fpr_list)


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(trainer, model_name='Model'):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(trainer.train_losses, label='Train', linewidth=2, color='steelblue')
    axes[0].plot(trainer.val_losses, label='Val', linewidth=2, color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_name} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
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
    print("JELLYFISH FORECASTING - PER BEACH PER DAY PREDICTIONS")
    print("Pure PyTorch Implementation with Baseline Model")
    print("=" * 100)
    print()
    
    # =====================================================================
    # LOAD DATA
    # =====================================================================
    print("1. LOADING DATA")
    print("-" * 100)
    
    X, y, metadata = load_jellyfish_data(lookback_days=7, forecast_days=1)
    
    print(f"✓ Loaded forecasting data")
    print(f"✓ Input shape: {X.shape} (samples, days, features_per_day)")
    print(f"✓ Output shape: {y.shape}")
    print()
    
    # =====================================================================
    # DATA NORMALIZATION
    # =====================================================================
    print("2. DATA NORMALIZATION")
    print("-" * 100)
    
    # Normalize per day per feature (across all samples)
    X_tensor = torch.FloatTensor(X)  # (n_samples, 7, 11)
    
    # Compute mean/std over all samples
    mean = X_tensor.mean(dim=0)  # (7, 11)
    std = X_tensor.std(dim=0)    # (7, 11)
    X_normalized = (X_tensor - mean) / (std + 1e-8)
    
    y_tensor = torch.FloatTensor(y)
    
    print(f"✓ Normalized using torch.mean() and torch.std()")
    print(f"✓ Feature mean: {mean.mean():.4f}, std: {std.mean():.4f}")
    print()
    
    # =====================================================================
    # CREATE DATASET AND DATALOADERS
    # =====================================================================
    print("3. CREATE DATALOADERS")
    print("-" * 100)
    
    dataset = TensorDataset(X_normalized, y_tensor)
    
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"✓ Train set: {len(train_dataset)} samples")
    print(f"✓ Val set: {len(val_dataset)} samples")
    print(f"✓ Test set: {len(test_dataset)} samples")
    print()
    
    # =====================================================================
    # TRAINING
    # =====================================================================
    print("4. TRAINING MODELS")
    print("-" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    results = {}
    
    # =====================================================================
    # BASELINE MODEL
    # =====================================================================
    print("BASELINE: Logistic Regression with Engineered Features")
    print("-" * 50)
    
    X_engineered = create_engineered_features_forecasting(X, lookback=7)
    
    X_eng_tensor = torch.FloatTensor(X_engineered)
    mean_eng = X_eng_tensor.mean(dim=0)
    std_eng = X_eng_tensor.std(dim=0)
    X_eng_normalized = (X_eng_tensor - mean_eng) / (std_eng + 1e-8)
    
    baseline_dataset = TensorDataset(X_eng_normalized, y_tensor)
    
    train_dataset_bl, val_dataset_bl, test_dataset_bl = random_split(
        baseline_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader_bl = DataLoader(train_dataset_bl, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_bl = DataLoader(val_dataset_bl, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_bl = DataLoader(test_dataset_bl, batch_size=BATCH_SIZE, shuffle=False)
    
    baseline_model = BaselineLogisticRegression(input_dim=X_eng_normalized.shape[1])
    baseline_trainer = Trainer(baseline_model, device=device, learning_rate=LEARNING_RATE)
    
    start_time = time.time()
    baseline_trainer.fit(train_loader_bl, val_loader_bl, epochs=NUM_EPOCHS, patience=15)
    baseline_time = time.time() - start_time
    
    baseline_metrics = baseline_trainer.test(test_loader_bl)
    
    print(f"\nBaseline Results:")
    print(f"  Accuracy:  {baseline_metrics['accuracy']:.4f}")
    print(f"  Precision: {baseline_metrics['precision']:.4f}")
    print(f"  Recall:    {baseline_metrics['recall']:.4f}")
    print(f"  F1-Score:  {baseline_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {baseline_metrics['auc']:.4f}")
    print(f"  Params:    {sum(p.numel() for p in baseline_model.parameters()):,}")
    
    plot_training_history(baseline_trainer, 'Baseline_LogisticRegression')
    
    results['Baseline'] = {
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
    models = {
        'Feedforward': FeedforwardNet(input_dim=7*11, dropout_prob=DROPOUT_PROB),
        'LSTM': LSTMNet(input_dim=11, hidden_dim=64, dropout_prob=DROPOUT_PROB),
        'GRU': GRUNet(input_dim=11, hidden_dim=64, dropout_prob=DROPOUT_PROB),
        'Conv1D': Conv1DNet(input_dim=11, hidden_dim=64, dropout_prob=DROPOUT_PROB),
        'Hybrid': HybridNet(input_dim=11, hidden_dim=64, dropout_prob=DROPOUT_PROB)
    }
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        print("-" * 50)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {num_params:,}")
        
        trainer = Trainer(model, device=device, learning_rate=LEARNING_RATE)
        start_time = time.time()
        trainer.fit(train_loader, val_loader, epochs=NUM_EPOCHS, patience=15)
        train_time = time.time() - start_time
        
        test_metrics = trainer.test(test_loader)
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall:    {test_metrics['recall']:.4f}")
        print(f"  F1-Score:  {test_metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
        
        results[model_name] = {
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'auc': test_metrics['auc'],
            'params': num_params,
            'time': train_time
        }
        
        plot_training_history(trainer, model_name)
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    print("\n" + "=" * 100)
    print("FORECASTING RESULTS - BASELINE VS NEURAL NETWORKS")
    print("=" * 100)
    print()
    
    print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 80)
    
    sorted_results = sorted(results.items(), key=lambda x: (x[0] != 'Baseline', x[0]))
    
    for model_name, metrics in sorted_results:
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")
    
    baseline_acc = results['Baseline']['accuracy']
    baseline_f1 = results['Baseline']['f1']
    
    print("\n" + "=" * 100)
    print("NEURAL NETWORK IMPROVEMENTS OVER BASELINE")
    print("=" * 100)
    print()
    
    for model_name, metrics in sorted_results:
        if model_name != 'Baseline':
            acc_imp = (metrics['accuracy'] - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
            f1_imp = (metrics['f1'] - baseline_f1) / baseline_f1 * 100 if baseline_f1 > 0 else 0
            
            print(f"{model_name:<20}: {acc_imp:+.2f}% accuracy, {f1_imp:+.2f}% F1-score")
    
    print("\n✓ Forecasting implementation complete!")
    print("=" * 100)
