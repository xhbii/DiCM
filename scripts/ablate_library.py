"""Command-line entry point for ablate library."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dicm.experiments.ablate_library import main

if __name__ == "__main__":
    main()
