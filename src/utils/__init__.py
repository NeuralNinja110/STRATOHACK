"""
Utility functions for the airline price forecasting system
"""
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

def add_src_to_path():
    """Add src directory to Python path"""
    src_path = Path(__file__).parent.parent / 'src'
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/processed',
        'models/saved',
        'logs',
        'temp'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def setup_environment():
    """Setup the environment for the application"""
    add_src_to_path()
    create_directories()

if __name__ == "__main__":
    setup_environment()
    print("Environment setup complete!")