"""Project paths and logger configuration."""

import logging
import os
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
tmp_dir = os.path.join(root_dir, "tmp")
results_dir = os.path.join(root_dir, "results")
dual_cache_dir = os.path.join(results_dir, "dual_cache")
analysis_dir = os.path.join(results_dir, "analysis")
fusion_dir = os.path.join(results_dir, "fusion")
plot_dir = os.path.join(results_dir, "plots")
dataset_dir = os.path.join(root_dir, "dataset", "clinc_oos")


def configure_logger(log_file_path: str, mode: str = "w", reset: bool = True, logging_level=logging.INFO):
    """Set up logger with a rotating file handler and a stream handler."""
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if reset:
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(log_file_path, mode=mode, maxBytes=10 * 1024 * 1024, backupCount=2, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    _extra_setup()


def _extra_setup():
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


logging.getLogger(__name__).addHandler(logging.NullHandler())
