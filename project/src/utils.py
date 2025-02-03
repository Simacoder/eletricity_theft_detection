# src/utils.py
import logging
from pathlib import Path
import json
from typing import Dict, Any

def setup_logging() -> None:
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_directories() -> None:
    """Create necessary project directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs'
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)