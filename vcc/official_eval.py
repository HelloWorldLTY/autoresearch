"""
Score the latest held-out predictions with the official cell-eval2 `vcc2026` pipeline.

train.py saves its held-out predictions; this script turns them into cells exactly as a
submission would (resampled control cells, scaled per gene, integer counts), rebuilds each
held-out context's real cells from the source h5ad, and runs:

    cell-eval2 baseline          (generic-response arm, the 0 end)      -- cached
    cell-eval2 prep-real-bundle  (+ split-half replicate, the 1 end)    -- cached
    cell-eval2 run               (the submission)
    cell-eval2 score --real-bundle

It is slow (cell-level Wilcoxon DE), so it is NOT part of the 5-minute loop. Use it to check
that the fast proxy in prepare.py tracks the real metric, e.g. every few kept improvements.

Usage: uv run official_eval.py [--contexts K562 ...]
"""

import os
import sys
import json
import argparse
import subprocess

import numpy as np
import pandas as pd
import scipy.sparse as sp

from prepare import (CACHE_DIR, CONTROL_LABEL, METRICS, gene_universe, load_val_tasks, dataset_for_context,
                     predictions_to_scales, emit_cells, evaluate_predictions)

OFFICIAL_NAMES = {
    "pds_cosine": "pds", "expr_mse_unbiased_capped_norm": "mse", "de_wilcoxon_lfc_nmae": "nmae",
    "de_wilcoxon_direction_fidelity_yield_raw": "fid", "de_wilcoxon_direction_reach_raw": "reach",
    "de_wilcoxon_sig_jaccard": "jac",
}


def cell_eval2(*args):
    exe = os.path.join(os.path.dirname(sys.executable), "cell-eval2")
    backend = "auto"
    try:
        import gpudge  # noqa: F401  (GPU DE engine; without it, auto raises on a CUDA host)
    except ImportError:
        backend = "pdex"
    cmd = [exe, *args, "--preset", "vcc2026", "--set", f"de.backend={backend}"]
    print("$", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit(f"cell-eval2 failed:\n{res.stdout[-3000:]}\n{res.stderr[-5000:]}")
    return res


def real_cells(task, out_path):
    """Panel + control cells of one held-out context, from the source h5ad, as raw counts."""
    import anndata as ad
    ds = dataset_for_context(task.context.name)
    a = ad.read_h5ad(os.path.expanduser(ds["path"]), backed="r")
    labels = a.obs[ds["pert_col"]].astype(str).values
    labels = np.where(labels == ds.get("control", CONTROL_LABEL), CONTROL_LABEL, labels)
    sel = np.isin(labels, np.append(task.targets, CONTROL_LABEL))
    if ds.get("context_col"):
        sel &= a.obs[ds["context_col"]].astype(str).values == task.context.name
    rows = np.flatnonzero(sel)
    symbols = np.asarray(a.var[ds["gene_col"]].astype(str).values) if ds.get("gene_col") else np.asarray(a.var_names.astype(str))
    universe = gene_universe()
    want = universe[task.context.genes]
    col = {g: j for j, g in reversed(list(enumerate(symbols)))}   # first occurrence wins, as in prepare.py
    cols = np.array([col[g] for g in want])
    X = a.X[rows]
    X = sp.csr_matrix(X.toarray() if hasattr(X, "toarray") and not sp.issparse(X) else X)[:, cols].astype(np.float32)
    a.file.close()
    real = ad.AnnData(X, obs=pd.DataFrame({"target": labels[rows]}, index=[f"c{i}" for i in range(len(rows))]),
                      var=pd.DataFrame(index=want))
    real.write_h5ad(out_path)
    return real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", nargs="*", help="subset of held-out contexts (default: all)")
    args = parser.parse_args()
    import anndata as ad
    tasks = [t for t in load_val_tasks() if not args.contexts or t.context.name in args.contexts]
    pred_dir = os.path.join(CACHE_DIR, "predictions", "val")
    preds = {t.context.name: np.load(os.path.join(pred_dir, f"{t.context.name}.npy")) for t in tasks}
    all_preds = {t.context.name: np.load(os.path.join(pred_dir, f"{t.context.name}.npy")) for t in load_val_tasks()}
    proxy = evaluate_predictions(all_preds, verbose=False)
    summary = {}
    for task in tasks:
        name = task.context.name
        d = os.path.join(CACHE_DIR, "official", name)
        os.makedirs(d, exist_ok=True)
        real_path, base_path, bundle = (os.path.join(d, f) for f in ("real.h5ad", "baseline_pred.h5ad", "bundle"))
        real = ad.read_h5ad(real_path) if os.path.exists(real_path) else real_cells(task, real_path)
        if not os.path.exists(os.path.join(bundle, "manifest.json")):
            cell_eval2("baseline", "-ar", real_path, "-o", os.path.join(d, "baseline"), "--save-pred", base_path)
            cell_eval2("prep-real-bundle", "--real", real_path, "--baseline", base_path, "-o", bundle, "--force")
        # the submission: cells emitted from the real control cells, n per target as in the real data
        labels = real.obs["target"].values
        ctrl = real.X[labels == CONTROL_LABEL]
        scales = predictions_to_scales(task, preds[name])
        rng = np.random.default_rng(0)
        blocks, obs = [], []
        for i, t in enumerate(task.targets):
            n = int((labels == t).sum())
            blocks.append(emit_cells(ctrl, scales[i], n, rng)); obs += [t] * n
        blocks.append(sp.csr_matrix(ctrl, dtype=np.float32)); obs += [CONTROL_LABEL] * ctrl.shape[0]
        pred = ad.AnnData(sp.vstack(blocks).tocsr(), obs=pd.DataFrame({"target": obs}, index=[f"p{i}" for i in range(len(obs))]),
                          var=real.var.copy())
        pred_path = os.path.join(d, "pred.h5ad")
        pred.write_h5ad(pred_path)
        run_dir = os.path.join(d, "run")
        cell_eval2("run", "-ap", pred_path, "-ar", real_path, "-o", run_dir)
        out_csv = os.path.join(d, "score.csv")
        exe = os.path.join(os.path.dirname(sys.executable), "cell-eval2")
        res = subprocess.run([exe, "score", "--user-agg", os.path.join(run_dir, "agg_results.csv"),
                              "--real-bundle", bundle, "-o", out_csv], capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(f"cell-eval2 score failed:\n{res.stderr[-5000:]}")
        df = pd.read_csv(out_csv).set_index("metric")["from_replicate"]
        scores = {OFFICIAL_NAMES[k]: float(v) for k, v in df.items() if k in OFFICIAL_NAMES}
        summary[name] = dict(official=float(df["avg_score"]), metrics=scores)
        line = f"[{name}] official {df['avg_score']:+.4f} | " + " ".join(f"{m} {scores[m]:+.3f}" for m in METRICS)
        if proxy:  # the fast scorer in prepare.py, for comparison
            line += f"\n[{name}] proxy    {proxy[name]['score']:+.4f} | " + " ".join(f"{m} {proxy[name]['scaled'][m]:+.3f}" for m in METRICS)
        print(line)
    official = float(np.mean([v["official"] for v in summary.values()]))
    print("---")
    print(f"official_score:   {official:.6f}")
    print(f"proxy_score:      {np.mean([proxy[n]['score'] for n in summary]):.6f}")
    with open(os.path.join(CACHE_DIR, "official", "last_score.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
