import sys
import os
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from core.base_system import RoboconSystem

if __name__ == "__main__":
    # Get config path relative to this script
    config_path = Path(__file__).resolve().parent / "config.yaml"
    
    app = RoboconSystem(config_path=str(config_path))
    app.run()
