"""Centralized default settings for training and prediction.

Edit values here to change project-wide defaults in one place.
"""

DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_WEATHER_CSV_PATH = 'data/IMS/data_202603142120.csv'
DEFAULT_USE_INTEGRATED_DATA = True
DEFAULT_INCLUDE_LIVE_XML = True

DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_PROB = 0.3
DEFAULT_NUM_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_HYBRID_HIDDEN_DIM = 64
DEFAULT_REPORT_PATH = 'training_report_latest.json'
