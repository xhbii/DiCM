"""Command-line entry point for evaluate subsets."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dicm.experiments.evaluate_subsets import main

if __name__ == "__main__":
    main()
