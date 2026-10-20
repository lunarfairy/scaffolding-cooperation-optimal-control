"""Run the manuscript's optimisation settings sequentially."""
import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    tasks = json.loads((root / "optimisation_tasks.json").read_text())
    for args in tasks:
        print("oc", *args, flush=True)
        subprocess.run([sys.executable, "run_rev_ipa.py", "oc", *map(str, args)],
                       cwd=root / "src", check=True)


if __name__ == "__main__":
    main()
