"""Configuration loader: reads config.json with environment variable overrides."""
import os
import json

from .logger import get_logger

_log = get_logger()

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(THIS_DIR, "..", "config.json")


def load_config() -> dict:
    """Load API configuration from config.json, with env var overrides."""
    config: dict = {}
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        pass  # config.json is optional
    except PermissionError:
        _log.warning("config.json exists but is not readable, using environment defaults")
    except json.JSONDecodeError:
        _log.warning("config.json is corrupted, using environment defaults")
    config.setdefault("api_url", os.environ.get("DEEPSEEK_API_URL", ""))
    config.setdefault("api_key", os.environ.get("DEEPSEEK_API_KEY", ""))
    return config
