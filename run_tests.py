"""Run deterministic tests without starting servers or contacting live news services."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q"], cwd=Path(__file__).parent))
