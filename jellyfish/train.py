"""
Jellyfish Forecasting - Training Script

Trains baseline and neural network models for jellyfish presence forecasting.
Saves trained model weights for later use in prediction.
"""

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import warnings
import os
import sys

warnings.filterwarnings('ignore')

if __package__ in (None, ""):
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from jellyfish.data_loader import load_jellyfish_data
    from jellyfish.models import (
        BaselineLogisticRegression,
        FeedforwardNet,
        LSTMNet,
        GRUNet,
        Conv1DNet,
        HybridNet
    )
else:
    from .data_loader import load_jellyfish_data
    from .models import (
        BaselineLogisticRegression,
        # FeedforwardNet,
        # LSTMNet,
        # GRUNet,
        # Conv1DNet,
        HybridNet
    )

# Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_PROB = 0.3
NUM_EPOCHS = 100


def create_engineered_features_forecasting(X, lookback=7):
    """Create engineered features for baseline model"""
    n_samples, lookback, n_features = X.shape
    engineered_features = []
    
    for sample_idx in range(n_samples):
        seq = X[sample_idx]
        features_for_sample = []
        
        for feat_idx in range(n_features):
            time_series = seq[:, feat_idx]
            
            current = time_series[-1]
            features_for_sample.append(current)
            
            prev = time_series[-2] if lookback >= 2 else time_series[0]
            features_for_sample.append(prev)
            
            prev_3 = time_series[-4] if lookback >= 4 else time_series[0]
            features_for_sample.append(prev_3)
            
            x_vals = np.arange(lookback)
            coeffs = np.polyfit(x_vals, time_series, 1)
            trend = coeffs[0]
            features_for_sample.append(trend)
            
            mean_val = np.mean(time_series)
            features_for_sample.append(mean_val)
            
            std_val = np.std(time_series)
            features_for_sample.append(std_val)
            
            min_val = np.min(time_series)
            max_val = np.max(time_series)
            features_for_sample.append(min_val)
            features_for_sample.append(max_val)
            
            change_1day = current - prev
            features_for_sample.append(change_1day)
            
            change_3day = current - prev_3
            features_for_sample.append(change_3day)
        
        engineered_features.append(features_for_sample)
    
    return np.array(engineered_features, dtype=np.float32)


class Trainer:
    """Training pipeline for all models"""
    
    def __init__(self, model, device='cpu', learning_rate=LEARNING_RATE, model_id='default'):
        self.model = model.to(device)
        self.device = device
        self.model_id = model_id
        self.best_state_dict = None  # Store best weights in memory
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
                # Store best weights in memory (avoid file conflicts)
                self.best_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    # Restore best weights from memory
                    if self.best_state_dict is not None:
                        self.model.load_state_dict(self.best_state_dict)
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


def train_all_models():
    """Main training function"""
    print("=" * 100)
    print("JELLYFISH FORECASTING - TRAINING")
    print("=" * 100)
    print()
    
    # Load data
    print("1. LOADING DATA")
    print("-" * 100)
    
    X, y, metadata = load_jellyfish_data(lookback_days=7, forecast_days=1)
    print()
    
    # Normalize
    print("2. DATA NORMALIZATION")
    print("-" * 100)
    
    X_tensor = torch.FloatTensor(X)
    mean = X_tensor.mean(dim=0)
    std = X_tensor.std(dim=0)
    X_normalized = (X_tensor - mean) / (std + 1e-8)
    y_tensor = torch.FloatTensor(y)
    
    print(f"✓ Normalized using torch.mean() and torch.std()")
    print()
    
    # Create dataloaders
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
    
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print()
    
    # Train models
    print("4. TRAINING MODELS")
    print("-" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    results = {}
    
    # Baseline
    print("BASELINE: Logistic Regression")
    print("-" * 90)
    
    X_engineered = create_engineered_features_forecasting(X, lookback=7)
    X_eng_tensor = torch.FloatTensor(X_engineered)
    mean_eng = X_eng_tensor.mean(dim=0)
    std_eng = X_eng_tensor.std(dim=0)
    X_eng_normalized = (X_eng_tensor - mean_eng) / (std_eng + 1e-8)
    
    baseline_dataset = TensorDataset(X_eng_normalized, y_tensor)
    train_dataset_bl, val_dataset_bl, test_dataset_bl = random_split(
        baseline_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader_bl = DataLoader(train_dataset_bl, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_bl = DataLoader(val_dataset_bl, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_bl = DataLoader(test_dataset_bl, batch_size=BATCH_SIZE, shuffle=False)
    
    baseline_model = BaselineLogisticRegression(input_dim=X_eng_normalized.shape[1])
    baseline_trainer = Trainer(baseline_model, device=device, learning_rate=LEARNING_RATE, model_id='baseline')
    
    start_time = time.time()
    baseline_trainer.fit(train_loader_bl, val_loader_bl, epochs=NUM_EPOCHS, patience=15)
    baseline_time = time.time() - start_time
    
    baseline_metrics = baseline_trainer.test(test_loader_bl)
    
    print(f"\nResults: Acc={baseline_metrics['accuracy']:.4f}, F1={baseline_metrics['f1']:.4f}")
    print(f"Saving model...")
    torch.save(baseline_model.state_dict(), 'baseline_model.pth')
    
    plot_training_history(baseline_trainer, 'Baseline')
    results['Baseline'] = baseline_metrics
    print()
    
    # Neural networks
    models = {
        'Feedforward': FeedforwardNet(input_dim=7*11),
        'LSTM': LSTMNet(input_dim=11),
        'GRU': GRUNet(input_dim=11),
        'Conv1D': Conv1DNet(input_dim=11),
        'Hybrid': HybridNet(input_dim=11)
    }
    
    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("-" * 90)
        
        trainer = Trainer(model, device=device, learning_rate=LEARNING_RATE, model_id=model_name.lower())
        start_time = time.time()
        trainer.fit(train_loader, val_loader, epochs=NUM_EPOCHS, patience=15)
        train_time = time.time() - start_time
        
        test_metrics = trainer.test(test_loader)
        
        print(f"\nResults: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}")
        print(f"Saving model...")
        torch.save(model.state_dict(), f'{model_name.lower()}_model.pth')
        
        plot_training_history(trainer, model_name)
        results[model_name] = test_metrics
    
    # Summary
    print("\n" + "=" * 100)
    print("TRAINING SUMMARY")
    print("=" * 100)
    print()
    
    print(f"{'Model':<20} {'Accuracy':<12} {'F1':<12} {'AUC':<12}")
    print("-" * 56)
    
    for model_name in sorted(results.keys()):
        metrics = results[model_name]
        print(f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} {metrics['auc']:<12.4f}")
    
    print("\n✓ Training complete! Model weights saved:")
    print("  - baseline_model.pth")
    print("  - feedforward_model.pth")
    print("  - lstm_model.pth")
    print("  - gru_model.pth")
    print("  - conv1d_model.pth")
    print("  - hybrid_model.pth")
    print("=" * 100)


if __name__ == '__main__':
    train_all_models()
