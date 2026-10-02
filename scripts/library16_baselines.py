"""Command-line entry point for library16 baselines."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dicm.experiments.library16_baselines import main

if __name__ == "__main__":
    main()
