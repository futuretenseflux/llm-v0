#!/usr/bin/env python3
"""
Run all data preparation scripts in order.
"""

import subprocess
import sys
from pathlib import Path

# List of data prep scripts to run, in order
DATA_PREP_SCRIPTS = [
    "src/data/pretraining/books_prepare.py",
    "src/data/pretraining/code_prepare.py",
    "src/data/pretraining/conv_forum_prepare.py",
    "src/data/pretraining/math_prepare.py",
    "src/data/pretraining/papers_prepare.py",
    "src/data/pretraining/primer_prepare.py",
]

def run_script(script_path: str) -> bool:
    """Run a single script and return True if successful."""
    print(f"\n=== Running {script_path} ===")
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=Path(__file__).parent.parent,
            capture_output=False,
        )
        print(f"✓ Completed {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed {script_path} with exit code {e.returncode}")
        return False

def main():
    """Run all data preparation scripts."""
    repo_root = Path(__file__).parent.parent
    print(f"Repository root: {repo_root}")
    
    failed_scripts = []
    
    for script in DATA_PREP_SCRIPTS:
        script_path = repo_root / script
        if not script_path.exists():
            print(f"✗ Script not found: {script_path}")
            failed_scripts.append(script)
            continue
            
        if not run_script(str(script_path)):
            failed_scripts.append(script)
    
    print("\n=== Summary ===")
    if failed_scripts:
        print(f"Failed scripts: {failed_scripts}")
        sys.exit(1)
    else:
        print("All data preparation scripts completed successfully!")

if __name__ == "__main__":
    main()
