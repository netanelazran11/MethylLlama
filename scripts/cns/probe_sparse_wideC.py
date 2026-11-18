"""
probe_sparse_wideC.py  (CNS task, stage 3 - Scenario B with a wider C grid)
===========================================================================
In Scenario B the inner CV picked the top of the C grid (100) for most folds, for MethylLlama and the raw baselines
alike, so the head may be under-fitted at the edge. This runs probe_sparse.py unchanged except for the grid, extended
by two values (1e3, 1e4), and writes to a separate root so the original results stay as they are.

Outputs  outputs/cns/sparse/probe_wideC/<A|B>/<rep>/<tag>/  (same files as probe_sparse.py)
For compare_sparse.py the root needs Scenario A beside it:  ln -s ../probe/A outputs/cns/sparse/probe_wideC/A
  python scripts/cns/probe_sparse_wideC.py --scenario B --levels 20,10,5,2,1 --seeds 0,1,2 --reps ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_cns  # noqa: E402

probe_cns.C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)  # fit_predict reads the module global
import probe_sparse  # noqa: E402

probe_sparse.C_GRID = probe_cns.C_GRID  # recorded in metrics.json

if __name__ == "__main__":
    if "--outdir" not in sys.argv:
        sys.argv += ["--outdir", str(probe_sparse.REPO / "outputs/cns/sparse/probe_wideC")]
    probe_sparse.main()
