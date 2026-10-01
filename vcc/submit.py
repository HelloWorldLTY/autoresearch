"""
Write a VCC 2026 submission from the predictions of the last `train.py` run.

For every controls-only context (role = "submit" in datasets.toml: the official control
cells, contexts A/B/C in the validation phase, D/E/F in the final phase) train.py saves a
log2 fold change per (target, gene). Each target becomes SUBMIT_CELLS_PER_PERT cells:
resampled control cells of that context, scaled per gene by 2**lfc and stochastically
rounded to integer counts. This is the same emission the scorer in prepare.py assumes, so
the local val_score is about what this file will be scored as.

The output follows the official rules: ONE .h5ad with all contexts, raw integer counts,
obs `target_gene` and `context` (the label from the control files), no control cells,
genes in the order of the official gene_names.csv (set it as [genes] reference). Then
package and submit with Arc's CLI (`uv tool install vcc-cli`):

    uv run submit.py
    vcc prep ~/.cache/autoresearch-vcc/submission/vcc_submission.h5ad \\
        -g gene_names.csv --perts pert_counts.csv -o prediction.vcc
    vcc submit prediction.vcc -m "autoresearch <commit>" --wait
"""

import os
import argparse

import numpy as np
import pandas as pd
import scipy.sparse as sp

from prepare import (CACHE_DIR, SUBMIT_CELLS_PER_PERT, gene_universe, load_submit_tasks,
                     predictions_to_scales, emit_cells)

OFFICIAL_CONTEXTS = ({"A", "B", "C"}, {"D", "E", "F"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(CACHE_DIR, "submission", "vcc_submission.h5ad"))
    parser.add_argument("--cells-per-pert", type=int, default=SUBMIT_CELLS_PER_PERT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import anndata as ad
    tasks = load_submit_tasks()
    if not tasks:
        raise SystemExit("No submit contexts. Add the official control file(s) to datasets.toml with role = \"submit\".")
    names = {t.context.name for t in tasks}
    if names not in OFFICIAL_CONTEXTS:
        print(f"WARNING: submit contexts are {sorted(names)}; the official labels are A,B,C (validation) or "
              f"D,E,F (final). Name contexts after the control files' `context` labels, or vcc prep will refuse.")
    universe = gene_universe()
    pred_dir = os.path.join(CACHE_DIR, "predictions", "submit")
    rng = np.random.default_rng(args.seed)
    blocks, targets, contexts = [], [], []
    for task in sorted(tasks, key=lambda t: t.context.name):
        name = task.context.name
        path = os.path.join(pred_dir, f"{name}.npy")
        if not os.path.exists(path):
            raise SystemExit(f"No predictions for {name} at {path}: run train.py first.")
        scales = predictions_to_scales(task, np.load(path))
        X = sp.vstack([emit_cells(task.context.ctrl_cells, scales[i], args.cells_per_pert, rng)
                       for i in range(len(task.targets))]).tocsr()
        # context genes -> full gene universe (genes not measured in this context stay 0)
        g = task.context.genes
        expand = sp.csr_matrix((np.ones(len(g)), (np.arange(len(g)), g)), shape=(len(g), len(universe)))
        blocks.append((X @ expand).tocsr())
        targets += np.repeat(task.targets, args.cells_per_pert).tolist()
        contexts += [name] * X.shape[0]
        print(f"[{name}] {len(task.targets)} targets x {args.cells_per_pert} cells")
    X = sp.vstack(blocks).tocsr().astype(np.float32)
    totals = np.asarray(X.sum(1)).ravel()
    obs = pd.DataFrame({"target_gene": targets, "context": contexts}, index=[f"cell_{i}" for i in range(X.shape[0])])
    a = ad.AnnData(X, obs=obs, var=pd.DataFrame(index=universe))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    a.write_h5ad(args.out)
    print(f"wrote {args.out}: {a.n_obs} cells x {a.n_vars} genes, max counts per cell {totals.max():.0f}")
    print("next: vcc prep <that file> -g gene_names.csv --perts pert_counts.csv -o prediction.vcc && "
          "vcc submit prediction.vcc -m \"<model name>\" --wait")


if __name__ == "__main__":
    main()
