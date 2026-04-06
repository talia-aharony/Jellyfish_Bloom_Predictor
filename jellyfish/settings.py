"""Centralized default settings for training and prediction.

Edit values here to change project-wide defaults in one place.
"""

DEFAULT_LOOKBACK_DAYS = 14
# Weather CSV path options:
#   - None: auto-discover all IMS CSV files in data/IMS/ and consolidate them
#   - "path/to/file.csv": use single IMS CSV file
#   - ["file1.csv", "file2.csv"]: consolidate these specific CSV files
DEFAULT_WEATHER_CSV_PATH = None  # Auto-discover all IMS CSV files
DEFAULT_USE_INTEGRATED_DATA = True
DEFAULT_INCLUDE_LIVE_XML = True

DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_PROB = 0.3
DEFAULT_NUM_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_HYBRID_HIDDEN_DIM = 64
DEFAULT_REPORT_PATH = 'training_report_latest.json'

DEFAULT_MODEL_NAMES = 'GRU,JellyfishNet'
DEFAULT_OUTPUT_DIR = 'models'

DEFAULT_FINETUNE_EPOCHS = 20
DEFAULT_FINETUNE_LR = 1e-4
DEFAULT_MIN_SAMPLES_PER_BEACH = 10

DEFAULT_PER_BEACH_PATIENCE = 8
DEFAULT_LR_SCHEDULER_FACTOR = 0.5
DEFAULT_LR_SCHEDULER_PATIENCE = 5
DEFAULT_GRAD_CLIP_NORM = 1.0

DEFAULT_THRESHOLD_MIN = 0.1
DEFAULT_THRESHOLD_MAX = 0.9
DEFAULT_THRESHOLD_STEPS = 81
DEFAULT_THRESHOLD_MIN_PRECISION = 0.4

DEFAULT_SWEEP_PRESET = 'focused'
DEFAULT_SWEEP_MODELS = DEFAULT_MODEL_NAMES

SWEEP_PRESET_FOCUSED = {
	'lookback_days': [14, 21],
	'learning_rates': [0.001, 0.0005],
	'dropouts': [0.2],
	'hybrid_hidden_dims': [96],
	'batch_sizes': [32],
	'epoch_options': [80],
	'patiences': [10],
}

SWEEP_PRESET_JELLYNET_STRONG = {
	'lookback_days': [14, 21, 28],
	'learning_rates': [0.0007, 0.0005, 0.0003],
	'dropouts': [0.2, 0.3, 0.35],
	'hybrid_hidden_dims': [96, 128],
	'batch_sizes': [16, 32],
	'epoch_options': [120],
	'patiences': [12, 18],
}

SWEEP_PRESET_QUICK = {
	'lookback_days': [DEFAULT_LOOKBACK_DAYS],
	'learning_rates': [0.001, 0.0005],
	'dropouts': [0.2, 0.3],
	'hybrid_hidden_dims': [48, 64],
	'batch_sizes': [DEFAULT_BATCH_SIZE],
	'epoch_options': [DEFAULT_NUM_EPOCHS],
	'patiences': [DEFAULT_PATIENCE],
}

SWEEP_PRESET_STANDARD = {
	'lookback_days': [7, 14, 21],
	'learning_rates': [0.001, 0.0007, 0.0005, 0.0003],
	'dropouts': [0.15, 0.2, 0.25, 0.3],
	'hybrid_hidden_dims': [32, 48, 64, 96],
	'batch_sizes': [16, 32],
	'epoch_options': [80, 100],
	'patiences': [10, 15],
}

SWEEP_PRESET_LARGE = {
	'lookback_days': [7, 10, 14, 21, 28],
	'learning_rates': [0.001, 0.0008, 0.0006, 0.0005, 0.0003, 0.0002],
	'dropouts': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
	'hybrid_hidden_dims': [24, 32, 48, 64, 80, 96],
	'batch_sizes': [16, 24, 32, 48],
	'epoch_options': [60, 80, 100, 120],
	'patiences': [8, 10, 12, 15],
}
