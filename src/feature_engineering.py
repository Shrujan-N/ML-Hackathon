import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features.feature_engineering import *  # noqa: F401, F403

if __name__ == "__main__":
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_feature_datasets(repo_dir)
