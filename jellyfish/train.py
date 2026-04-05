"""
Jellyfish Forecasting - Training Script

Trains baseline and neural network models for jellyfish presence forecasting.
Saves trained model weights for later use in prediction.
"""

import numpy as np
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import warnings
import os
import sys
import argparse
from datetime import datetime

warnings.filterwarnings('ignore')

if __package__ in (None, ""):
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from jellyfish.data_loader import load_jellyfish_data
    from jellyfish.data_loader_forecasting import load_integrated_data
    from jellyfish.settings import (
        DEFAULT_LOOKBACK_DAYS,
        DEFAULT_WEATHER_CSV_PATH,
        DEFAULT_USE_INTEGRATED_DATA,
        DEFAULT_INCLUDE_LIVE_XML,
        DEFAULT_BATCH_SIZE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_DROPOUT_PROB,
        DEFAULT_NUM_EPOCHS,
        DEFAULT_PATIENCE,
        DEFAULT_HYBRID_HIDDEN_DIM,
        DEFAULT_REPORT_PATH,
    )
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
    from .data_loader_forecasting import load_integrated_data
    from .settings import (
        DEFAULT_LOOKBACK_DAYS,
        DEFAULT_WEATHER_CSV_PATH,
        DEFAULT_USE_INTEGRATED_DATA,
        DEFAULT_INCLUDE_LIVE_XML,
        DEFAULT_BATCH_SIZE,
        DEFAULT_LEARNING_RATE,
        DEFAULT_DROPOUT_PROB,
        DEFAULT_NUM_EPOCHS,
        DEFAULT_PATIENCE,
        DEFAULT_HYBRID_HIDDEN_DIM,
        DEFAULT_REPORT_PATH,
    )
    from .models import (
        BaselineLogisticRegression,
        FeedforwardNet,
        LSTMNet,
        GRUNet,
        Conv1DNet,
        HybridNet
    )

# Hyperparameters
BATCH_SIZE = DEFAULT_BATCH_SIZE
LEARNING_RATE = DEFAULT_LEARNING_RATE
DROPOUT_PROB = DEFAULT_DROPOUT_PROB
NUM_EPOCHS = DEFAULT_NUM_EPOCHS


def save_training_report(results, config, output_path):
    """Save training configuration and metrics to JSON for experiment tracking."""
    serializable_results = {}
    for model_name, metrics in results.items():
        serializable_results[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['auc']),
            'threshold': float(metrics.get('threshold', 0.5)),
            'val_best_f1': float(metrics.get('val_best_f1', 0.0)),
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
        }

    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'config': config,
        'results': serializable_results,
    }

    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f"✓ Saved training report: {output_path}")


def create_engineered_features_forecasting(X, lookback=DEFAULT_LOOKBACK_DAYS):
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
    
    def _collect_predictions(self, data_loader):
        """Collect probabilities and labels from a data loader."""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch = X_batch.to(self.device)
                outputs = self.model(X_batch)

                all_preds.extend(outputs.detach().cpu().view(-1).tolist())
                all_labels.extend(y_batch.detach().cpu().view(-1).tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        return all_preds, all_labels

    @staticmethod
    def _compute_classification_metrics(all_labels, all_preds, threshold=0.5):
        """Compute thresholded classification metrics from probabilities."""
        all_preds_binary = (all_preds >= threshold).astype(int)

        accuracy = np.mean(all_preds_binary == all_labels)

        tp = np.sum((all_preds_binary == 1) & (all_labels == 1))
        fp = np.sum((all_preds_binary == 1) & (all_labels == 0))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        fn = np.sum((all_preds_binary == 0) & (all_labels == 1))
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        tn = np.sum((all_preds_binary == 0) & (all_labels == 0))
        cm = np.array([[tn, fp], [fn, tp]])

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'labels': all_labels,
            'threshold': threshold,
        }

    def find_best_threshold(self, val_loader):
        """Find probability threshold that maximizes validation F1."""
        all_preds, all_labels = self._collect_predictions(val_loader)

        best_threshold = 0.5
        best_f1 = -1.0

        for threshold in np.linspace(0.1, 0.9, 81):
            metrics = self._compute_classification_metrics(all_labels, all_preds, threshold=threshold)
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_threshold = float(threshold)

        return best_threshold, best_f1

    def test(self, test_loader, threshold=0.5):
        """Test on test set with configurable decision threshold."""
        all_preds, all_labels = self._collect_predictions(test_loader)

        metrics = self._compute_classification_metrics(all_labels, all_preds, threshold=threshold)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        auc = self._compute_auc(all_labels, all_preds)
        metrics['auc'] = auc

        return metrics
    
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


def train_all_models(
    lookback_days=DEFAULT_LOOKBACK_DAYS,
    use_integrated_data=DEFAULT_USE_INTEGRATED_DATA,
    weather_csv_path=DEFAULT_WEATHER_CSV_PATH,
    include_live_xml=DEFAULT_INCLUDE_LIVE_XML,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    dropout_prob=DROPOUT_PROB,
    num_epochs=NUM_EPOCHS,
    patience=DEFAULT_PATIENCE,
    hybrid_hidden_dim=DEFAULT_HYBRID_HIDDEN_DIM,
    model_names=None,
    report_path=DEFAULT_REPORT_PATH,
):
    """Main training function"""
    print("=" * 100)
    print("JELLYFISH FORECASTING - TRAINING")
    print("=" * 100)
    print()
    
    config = {
        'lookback_days': int(lookback_days),
        'use_integrated_data': bool(use_integrated_data),
        'weather_csv_path': str(weather_csv_path),
        'include_live_xml': bool(include_live_xml),
        'batch_size': int(batch_size),
        'learning_rate': float(learning_rate),
        'dropout_prob': float(dropout_prob),
        'num_epochs': int(num_epochs),
        'patience': int(patience),
        'hybrid_hidden_dim': int(hybrid_hidden_dim),
        'model_names': list(model_names) if model_names is not None else ['GRU', 'Hybrid'],
        'threshold_selection_metric': 'f1_on_validation',
    }

    print("Training configuration:")
    for key, value in config.items():
        print(f"  - {key}: {value}")
    print()

    # Load data
    print("1. LOADING DATA")
    print("-" * 100)

    if use_integrated_data:
        integrated = load_integrated_data(
            weather_csv_path=weather_csv_path,
            lookback_days=lookback_days,
            forecast_days=1,
            include_live_xml=include_live_xml,
        )
        if integrated is None:
            raise RuntimeError("Failed to load integrated data. Check weather_csv_path and input files.")
        X, y, metadata, feature_cols, daily_citizen, daily_weather, merged = integrated
    else:
        X, y, metadata = load_jellyfish_data(lookback_days=lookback_days, forecast_days=1)

    n_features = int(X.shape[2])
    config['n_features_per_day'] = n_features
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"✓ Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print()
    
    # Train models
    print("4. TRAINING MODELS")
    print("-" * 100)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    results = {}
    
    # # Baseline
    # print("BASELINE: Logistic Regression")
    # print("-" * 90)
    
    # X_engineered = create_engineered_features_forecasting(X, lookback=lookback_days)
    # X_eng_tensor = torch.FloatTensor(X_engineered)
    # mean_eng = X_eng_tensor.mean(dim=0)
    # std_eng = X_eng_tensor.std(dim=0)
    # X_eng_normalized = (X_eng_tensor - mean_eng) / (std_eng + 1e-8)
    
    # baseline_dataset = TensorDataset(X_eng_normalized, y_tensor)
    # train_dataset_bl, val_dataset_bl, test_dataset_bl = random_split(
    #     baseline_dataset, [train_size, val_size, test_size],
    #     generator=torch.Generator().manual_seed(42)
    # )
    
    # train_loader_bl = DataLoader(train_dataset_bl, batch_size=batch_size, shuffle=True)
    # val_loader_bl = DataLoader(val_dataset_bl, batch_size=batch_size, shuffle=False)
    # test_loader_bl = DataLoader(test_dataset_bl, batch_size=batch_size, shuffle=False)
    
    # baseline_model = BaselineLogisticRegression(input_dim=X_eng_normalized.shape[1])
    # baseline_trainer = Trainer(baseline_model, device=device, learning_rate=learning_rate, model_id='baseline')
    
    # start_time = time.time()
    # baseline_trainer.fit(train_loader_bl, val_loader_bl, epochs=num_epochs, patience=patience)
    # baseline_time = time.time() - start_time
    
    # baseline_best_threshold, baseline_val_best_f1 = baseline_trainer.find_best_threshold(val_loader_bl)
    # baseline_metrics = baseline_trainer.test(test_loader_bl, threshold=baseline_best_threshold)
    # baseline_metrics['val_best_f1'] = baseline_val_best_f1
    
    # print(
    #     f"\nResults: Acc={baseline_metrics['accuracy']:.4f}, "
    #     f"F1={baseline_metrics['f1']:.4f}, "
    #     f"AUC={baseline_metrics['auc']:.4f}, "
    #     f"Threshold={baseline_metrics['threshold']:.2f}"
    # )
    # print(f"Saving model...")
    # torch.save(baseline_model.state_dict(), 'baseline_model.pth')
    
    # plot_training_history(baseline_trainer, 'Baseline')
    # results['Baseline'] = baseline_metrics
    # print()
    
    # Neural networks
    requested_models = list(model_names) if model_names is not None else ['GRU', 'Hybrid']
    allowed_models = {
        'GRU': GRUNet(input_dim=n_features, dropout_prob=dropout_prob),
        'Hybrid': HybridNet(input_dim=n_features, hidden_dim=hybrid_hidden_dim, dropout_prob=dropout_prob),
    }
    models = {}
    for name in requested_models:
        if name not in allowed_models:
            raise ValueError(f"Unsupported model '{name}'. Supported models: {sorted(allowed_models.keys())}")
        models[name] = allowed_models[name]
    
    for model_name, model in models.items():
        print(f"\n{model_name}")
        print("-" * 90)
        
        trainer = Trainer(model, device=device, learning_rate=learning_rate, model_id=model_name.lower())
        start_time = time.time()
        trainer.fit(train_loader, val_loader, epochs=num_epochs, patience=patience)
        train_time = time.time() - start_time
        
        best_threshold, val_best_f1 = trainer.find_best_threshold(val_loader)
        test_metrics = trainer.test(test_loader, threshold=best_threshold)
        test_metrics['val_best_f1'] = val_best_f1
        
        print(
            f"\nResults: Acc={test_metrics['accuracy']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"AUC={test_metrics['auc']:.4f}, "
            f"Threshold={test_metrics['threshold']:.2f}"
        )
        print(f"Saving model...")
        torch.save(model.state_dict(), f'{model_name.lower()}_model.pth')
        
        plot_training_history(trainer, model_name)
        results[model_name] = test_metrics
    
    # Summary
    print("\n" + "=" * 100)
    print("TRAINING SUMMARY")
    print("=" * 100)
    print()
    
    print(f"{'Model':<20} {'Accuracy':<12} {'F1':<12} {'AUC':<12} {'Thresh':<8}")
    print("-" * 66)
    
    for model_name in sorted(results.keys()):
        metrics = results[model_name]
        print(
            f"{model_name:<20} {metrics['accuracy']:<12.4f} {metrics['f1']:<12.4f} "
            f"{metrics['auc']:<12.4f} {metrics['threshold']:<8.2f}"
        )
    
    print("\n✓ Training complete! Model weights saved:")
    for model_name in sorted(results.keys()):
        print(f"  - {model_name.lower()}_model.pth")
    print("=" * 100)

    save_training_report(results, config, report_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train jellyfish forecasting models with tunable hyperparameters')
    parser.add_argument('--lookback-days', type=int, default=DEFAULT_LOOKBACK_DAYS, help=f'Historical input window length in days (default: {DEFAULT_LOOKBACK_DAYS})')
    parser.add_argument('--use-integrated-data', action='store_true', help='Train using integrated citizen + IMS weather + live RSS features')
    parser.add_argument('--weather-csv-path', type=str, default=DEFAULT_WEATHER_CSV_PATH, help='Path to IMS weather CSV for integrated mode')
    parser.add_argument('--disable-live-xml', action='store_true', help='Disable live RSS XML enrichment when using integrated mode')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help=f'Learning rate (default: {LEARNING_RATE})')
    parser.add_argument('--dropout-prob', type=float, default=DROPOUT_PROB, help=f'Dropout probability (default: {DROPOUT_PROB})')
    parser.add_argument('--num-epochs', type=int, default=NUM_EPOCHS, help=f'Number of epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    parser.add_argument('--hybrid-hidden-dim', type=int, default=96, help='Hybrid model hidden dimension (default: 96)')
    parser.add_argument('--models', type=str, default='GRU,Hybrid', help='Comma-separated models to train (default: GRU,Hybrid)')
    parser.add_argument('--report-path', type=str, default='training_report_latest.json', help='Output JSON path for training report')

    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(',') if m.strip()]

    train_all_models(
        lookback_days=args.lookback_days,
        use_integrated_data=args.use_integrated_data,
        weather_csv_path=args.weather_csv_path,
        include_live_xml=not args.disable_live_xml,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_prob=args.dropout_prob,
        num_epochs=args.num_epochs,
        patience=args.patience,
        hybrid_hidden_dim=args.hybrid_hidden_dim,
        model_names=model_names,
        report_path=args.report_path,
    )
