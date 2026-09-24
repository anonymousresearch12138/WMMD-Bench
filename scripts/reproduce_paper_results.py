"""CPU-only rendering of bundled measurements. Does not train or evaluate models."""
import runpy,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'paper'))
for file in ['generate_tables.py','appendix_tables.py','generate_figures.py','scale7b.py']:
    runpy.run_path(str(ROOT/'paper'/file),run_name='__main__')
print('Tables and figures written under paper/; experimental values unchanged.')
