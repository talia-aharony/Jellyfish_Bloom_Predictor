"""Compatibility wrappers for data-loading functions."""

from data_loader import load_all_data, load_jellyfish_data

try:
    from data_loader_forecasting import load_integrated_data
except ImportError:
    def load_integrated_data(*args, **kwargs):
        raise RuntimeError(
            "Integrated loader is unavailable in this workspace. "
            "Use load_jellyfish_data() or add data_loader_forecasting.py."
        )

__all__ = ["load_all_data", "load_jellyfish_data", "load_integrated_data"]
