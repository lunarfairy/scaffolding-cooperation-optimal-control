"""Generate Figs. 1–6 from figure_data/."""
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
for number in range(1, 7):
    subprocess.run([sys.executable, str(here / f"make_fig{number}.py")],
                   cwd=here, check=True)
