"""
Write VCC 2026 submission files from the predictions of the last `train.py` run.

For every controls-only context (role = "submit" in datasets.toml, e.g. the official
validation/test NTC files) train.py saves a log2 fold change per (target, gene). Each target
becomes SUBMIT_CELLS_PER_PERT cells: resampled control cells of that context, scaled per gene
by 2**lfc and stochastically rounded to integer counts -- the same emission the scorer in
prepare.py assumes, so the local val_score is about what these files will be scored as.

Usage:
    uv run submit.py                          # one .h5ad per context in ~/.cache/autoresearch-vcc/submission/
    uv run submit.py --combined               # plus a single file with all contexts
    uv run submit.py --pert-col target_gene --context-col context

Check the official submission instructions (obs column names, whether control cells are
included, file layout) before uploading; adjust the flags rather than this file's logic.
"""

import os
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from prepare import (CACHE_DIR, CONTROL_LABEL, SUBMIT_CELLS_PER_PERT, gene_universe, load_submit_tasks,
                     predictions_to_scales, emit_cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(CACHE_DIR, "submission"))
    parser.add_argument("--cells-per-pert", type=int, default=SUBMIT_CELLS_PER_PERT)
    parser.add_argument("--pert-col", default="target_gene", help="obs column holding the target gene")
    parser.add_argument("--context-col", default="context", help="obs column holding the context name")
    parser.add_argument("--include-controls", action="store_true", help="also write the stored control cells")
    parser.add_argument("--combined", action="store_true", help="also write all contexts into one file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import anndata as ad
    tasks = load_submit_tasks()
    if not tasks:
        raise SystemExit("No submit contexts. Add the official control files to datasets.toml with role = \"submit\".")
    universe = gene_universe()
    pred_dir = os.path.join(CACHE_DIR, "predictions", "submit")
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    written = []
    for task in tasks:
        name = task.context.name
        path = os.path.join(pred_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise SystemExit(f"No predictions for {name} at {path}: run train.py first.")
        scales = predictions_to_scales(task, np.load(path))
        ctrl = task.context.ctrl_cells
        blocks, labels = [], []
        for i, target in enumerate(task.targets):
            blocks.append(emit_cells(ctrl, scales[i], args.cells_per_pert, rng))
            labels += [target] * args.cells_per_pert
        if args.include_controls:
            blocks.append(sp.csr_matrix(ctrl, dtype=np.float32))
            labels += [CONTROL_LABEL] * ctrl.shape[0]
        X = sp.vstack(blocks).tocsr()
        # context genes -> full gene universe (genes not measured in this context stay 0)
        expand = sp.csr_matrix((np.ones(len(task.context.genes)), (np.arange(len(task.context.genes)), task.context.genes)),
                               shape=(len(task.context.genes), len(universe)))
        X = (X @ expand).tocsr().astype(np.float32)
        obs = pd.DataFrame({args.pert_col: labels, args.context_col: name},
                           index=[f"{name}_{i}" for i in range(len(labels))])
        a = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=universe))
        out = os.path.join(args.out, f"{name}.h5ad")
        a.write_h5ad(out, compression="gzip")
        written.append(a)
        print(f"wrote {out}: {a.n_obs} cells x {a.n_vars} genes, {len(task.targets)} targets")
    if args.combined:
        allc = ad.concat(written, merge="same")
        out = os.path.join(args.out, "vcc_submission.h5ad")
        allc.write_h5ad(out, compression="gzip")
        print(f"wrote {out}: {allc.n_obs} cells x {allc.n_vars} genes")


if __name__ == "__main__":
    main()
