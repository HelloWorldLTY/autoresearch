"""
One-time data preparation + fixed evaluation for autoresearch on the
Arc Virtual Cell Challenge 2026 (zero-shot CRISPRi response prediction).

The 2026 task: for a cell line that has never been seen perturbed, you are given
its non-targeting control cells and a panel of genes to knock down; predict the
single-cell expression of each knockdown. This file turns public Perturb-seq
datasets into that exact task by holding out whole cell lines ("contexts").

Usage:
    python prepare.py                      # ingest everything listed in datasets.toml
    python prepare.py --config my.toml     # use a different config
    python prepare.py --synthetic          # build a small synthetic dataset (smoke tests, no downloads)

Everything is cached in ~/.cache/autoresearch-vcc/ (override with VCC_CACHE_DIR).

This file is READ-ONLY for the agent: it holds the time budget, the zero-shot
split, and `evaluate_predictions`, the ground-truth metric of the research loop.
"""

import os
import sys
import json
import time
import hashlib
import argparse
import tomllib
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy import stats as sstats

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = int(os.environ.get("VCC_TIME_BUDGET", 300))  # training budget in seconds (env override is for smoke tests only)
PANEL_SIZE = 300            # perturbations scored per held-out context (as in VCC 2026)
MIN_CELLS_PER_PERT = 40     # perturbations with fewer cells are dropped
MAX_CTRL_CELLS = 2000       # control cells kept per context as model input
DE_TARGET_SUM = 1e6         # per-cell normalisation for DE (cell-eval2 vcc2026: target_sum)
BULK_TARGET_SUM = 5e4       # pseudobulk normalisation for expression metrics (bulk_target_sum)
P_ADJ = 0.05                # BH-adjusted significance threshold for DE calls
SUBMIT_CELLS_PER_PERT = 400 # cells per perturbation in a VCC 2026 submission
LFC_CLIP = 10.0             # predicted log2 fold changes are clipped to [-LFC_CLIP, LFC_CLIP]
CONTROL_LABEL = "non-targeting"

CACHE_DIR = os.environ.get("VCC_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-vcc"))
CONTEXT_DIR = os.path.join(CACHE_DIR, "contexts")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "datasets.toml")

# ---------------------------------------------------------------------------
# Ingestion: raw Perturb-seq h5ad -> per-context sufficient statistics
# ---------------------------------------------------------------------------
#
# For every context (cell line) we store, gene-wise over the context's measured genes:
#   train_*   : one row per perturbation, summed raw counts + cell counts (model training data)
#   panel_*   : PANEL_SIZE scored perturbations + the control (last row): n, summed counts,
#               summed per-cell CPM.
#   ctrl_cells: up to MAX_CTRL_CELLS raw control cells (what VCC 2026 hands you as input).
#   cells     : held-out contexts only: the raw cells of the panel perturbations (the scorer's
#               Wilcoxon tests and split-half replicate anchor need cells, not sums).

def _stable_seed(*parts):
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def _read_lines(path):
    with open(os.path.expanduser(path)) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def _gene_symbols(adata, gene_col):
    if gene_col:
        return np.asarray(adata.var[gene_col].astype(str).values)
    return np.asarray(adata.var_names.astype(str))


def build_gene_universe(cfg, sources):
    """Gene universe: the reference list if given (e.g. the 18,533 VCC genes), else the union."""
    ref = cfg.get("genes", {}).get("reference")
    if ref:
        ref = os.path.expanduser(ref)
        if ref.endswith(".h5ad"):
            import anndata as ad
            a = ad.read_h5ad(ref, backed="r")
            genes = _gene_symbols(a, cfg["genes"].get("gene_col", ""))
            a.file.close()
        else:
            genes = np.array(_read_lines(ref))
        _, first = np.unique(genes, return_index=True)
        return genes[np.sort(first)]
    allg = set()
    for _, symbols in sources:
        allg.update(symbols.tolist())
    return np.array(sorted(allg))


def _row_chunks(X, n_rows, chunk):
    for start in range(0, n_rows, chunk):
        stop = min(start + chunk, n_rows)
        block = X[start:stop]
        if not sp.issparse(block):
            block = sp.csr_matrix(np.asarray(block))
        yield start, stop, sp.csr_matrix(block, dtype=np.float32)


def ingest_dataset(ds, universe, held_out=(), chunk=20000):
    """Stream one h5ad (possibly huge, read backed) into per-context statistics.
    For held-out contexts the raw panel cells are kept too (the scorer's DE tests need ranks)."""
    import anndata as ad
    path = os.path.expanduser(ds["path"])
    print(f"[{ds['name']}] reading {path}")
    adata = ad.read_h5ad(path, backed="r")
    symbols = _gene_symbols(adata, ds.get("gene_col", ""))
    uni_index = {g: i for i, g in enumerate(universe)}
    # file column -> universe column (first occurrence of a symbol wins)
    col_map = np.full(len(symbols), -1, dtype=np.int64)
    seen = set()
    for j, g in enumerate(symbols):
        if g in uni_index and g not in seen:
            col_map[j] = uni_index[g]
            seen.add(g)
    keep_cols = np.flatnonzero(col_map >= 0)
    ctx_genes = col_map[keep_cols]                     # universe indices measured in this file
    order = np.argsort(ctx_genes)
    keep_cols, ctx_genes = keep_cols[order], ctx_genes[order]
    G = len(keep_cols)

    obs = adata.obs
    labels = obs[ds["pert_col"]].astype(str).values
    control = ds.get("control", CONTROL_LABEL)
    labels = np.where(labels == control, CONTROL_LABEL, labels)
    if ds.get("context_col"):
        ctx_of_cell = obs[ds["context_col"]].astype(str).values
    else:
        ctx_of_cell = np.full(len(labels), ds.get("context", ds["name"]))
    role = ds.get("role", "train")  # "train" (perturbations available) or "submit" (controls only)

    results = {}
    for ctx in sorted(set(ctx_of_cell)):
        in_ctx = ctx_of_cell == ctx
        lab = labels[in_ctx]
        rows = np.flatnonzero(in_ctx)
        rng = np.random.default_rng(_stable_seed(ds["name"], ctx))
        perts, counts = np.unique(lab[lab != CONTROL_LABEL], return_counts=True)
        perts = perts[counts >= MIN_CELLS_PER_PERT]
        if role == "submit":
            perts = perts[:0]
        n_ctrl = int((lab == CONTROL_LABEL).sum())
        if n_ctrl < 2 * MIN_CELLS_PER_PERT:
            print(f"  [{ctx}] skipped: only {n_ctrl} control cells")
            continue
        # scoring panel: perturbations whose target is a measured gene, chosen reproducibly
        measured = set(universe[ctx_genes].tolist())
        eligible = np.array([p for p in perts if p in measured])
        panel = np.sort(rng.choice(eligible, size=min(PANEL_SIZE, len(eligible)), replace=False)) if len(eligible) else eligible
        pert_idx = {p: i for i, p in enumerate(perts)}
        panel_idx = {p: i for i, p in enumerate(panel)}
        # per-cell group assignments
        train_group = np.array([pert_idx.get(l, -1) for l in lab])
        panel_group = np.array([panel_idx.get(l, -1) for l in lab])
        panel_group = np.where(lab == CONTROL_LABEL, len(panel), panel_group)   # control is the last row
        ctrl_rows_local = np.flatnonzero(lab == CONTROL_LABEL)
        keep_ctrl = np.sort(rng.choice(ctrl_rows_local, size=min(MAX_CTRL_CELLS, len(ctrl_rows_local)), replace=False))
        results[ctx] = dict(rows=rows, perts=perts, panel=panel, train_group=train_group,
                            panel_group=panel_group, keep_ctrl=keep_ctrl,
                            n_panel_rows=len(panel) + 1, role=role, keep_cells=ctx in held_out)
        print(f"  [{ctx}] {len(rows)} cells, {len(perts)} perturbations (>= {MIN_CELLS_PER_PERT} cells), "
              f"panel {len(panel)}, {n_ctrl} controls, {G} genes")

    # accumulators
    acc = {}
    for ctx, r in results.items():
        P, R = len(r["perts"]), r["n_panel_rows"]
        acc[ctx] = dict(train_sum=np.zeros((P, G), np.float64), train_n=np.zeros(P, np.int64),
                        p_n=np.zeros(R, np.int64), p_sum=np.zeros((R, G)), p_cpm=np.zeros((R, G)),
                        ctrl=[], ctrl_lib=[], cells=[], cells_lib=[], cells_group=[])
    # map global row -> (context, local position)
    row_ctx = np.full(adata.n_obs, -1, dtype=np.int64)
    row_loc = np.full(adata.n_obs, -1, dtype=np.int64)
    ctx_names = list(results)
    for ci, ctx in enumerate(ctx_names):
        row_ctx[results[ctx]["rows"]] = ci
        row_loc[results[ctx]["rows"]] = np.arange(len(results[ctx]["rows"]))

    t0 = time.time()
    for start, stop, block in _row_chunks(adata.X, adata.n_obs, chunk):
        lib = np.asarray(block.sum(axis=1)).ravel()                 # library size over ALL genes in the file
        block = block[:, keep_cols]
        for ci, ctx in enumerate(ctx_names):
            sel = np.flatnonzero(row_ctx[start:stop] == ci)
            if not len(sel):
                continue
            r, a = results[ctx], acc[ctx]
            loc = row_loc[start:stop][sel]
            X = block[sel]
            libs = np.maximum(lib[sel], 1.0)
            # training sums
            tg = r["train_group"][loc]
            m = tg >= 0
            if m.any():
                M = sp.csr_matrix((np.ones(m.sum()), (tg[m], np.flatnonzero(m))), shape=(len(r["perts"]), len(sel)))
                a["train_sum"] += np.asarray((M @ X).todense())
                np.add.at(a["train_n"], tg[m], 1)
            # panel / control statistics
            pg = r["panel_group"][loc]
            m = pg >= 0
            if m.any():
                idx = np.flatnonzero(m)
                M = sp.csr_matrix((np.ones(len(idx)), (pg[m], np.arange(len(idx)))), shape=(r["n_panel_rows"], len(idx)))
                Xs = X[idx]
                cpm = sp.diags(DE_TARGET_SUM / libs[idx]) @ Xs
                a["p_sum"] += np.asarray((M @ Xs).todense())
                a["p_cpm"] += np.asarray((M @ cpm).todense())
                np.add.at(a["p_n"], pg[m], 1)
                if r["keep_cells"]:
                    pert_cells = idx[pg[m] < len(r["panel"])]
                    a["cells"].append(X[pert_cells])
                    a["cells_lib"].append(libs[pert_cells])
                    a["cells_group"].append(r["panel_group"][loc[pert_cells]])
            # raw control cells (model input)
            keep = np.flatnonzero(np.isin(loc, r["keep_ctrl"]))
            if len(keep):
                a["ctrl"].append(X[keep])
                a["ctrl_lib"].extend(libs[keep].tolist())
        print(f"  rows {stop}/{adata.n_obs} ({time.time() - t0:.0f}s)", end="\r")
    print()
    adata.file.close()

    out = []
    for ctx in ctx_names:
        r, a = results[ctx], acc[ctx]
        stack = lambda xs: sp.vstack(xs).tocsr().astype(np.float32) if xs else sp.csr_matrix((0, G), dtype=np.float32)
        cat = lambda xs, dt: np.concatenate(xs).astype(dt) if xs else np.zeros(0, dt)
        out.append(dict(
            context=ctx, dataset=ds["name"], role=r["role"], genes=ctx_genes,
            perts=r["perts"], train_sum=a["train_sum"].astype(np.float32), train_n=a["train_n"],
            panel=r["panel"], p_n=a["p_n"], p_sum=a["p_sum"].astype(np.float32),
            p_cpm=a["p_cpm"].astype(np.float32),
            ctrl_cells=stack(a["ctrl"]), ctrl_lib=np.array(a["ctrl_lib"], dtype=np.float64),
            cells=stack(a["cells"]), cells_lib=cat(a["cells_lib"], np.float64),
            cells_group=cat(a["cells_group"], np.int32),
        ))
    return out


def _save_csr(prefix, m):
    return {f"{prefix}_data": m.data, f"{prefix}_indices": m.indices, f"{prefix}_indptr": m.indptr,
            f"{prefix}_shape": np.array(m.shape)}


def _load_csr(z, prefix):
    return sp.csr_matrix((z[f"{prefix}_data"], z[f"{prefix}_indices"], z[f"{prefix}_indptr"]),
                         shape=tuple(z[f"{prefix}_shape"]))


def save_context(c):
    os.makedirs(CONTEXT_DIR, exist_ok=True)
    path = os.path.join(CONTEXT_DIR, f"{c['context']}.npz")
    mats = {**_save_csr("ctrl", c.pop("ctrl_cells")), **_save_csr("cells", c.pop("cells"))}
    np.savez(path, **mats, **{k: v for k, v in c.items() if isinstance(v, np.ndarray)},
             meta=json.dumps({k: v for k, v in c.items() if not isinstance(v, np.ndarray)}))
    print(f"  saved {path}")


def prepare(config_path):
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    datasets = [d for d in cfg.get("dataset", []) if d.get("enabled", True)]
    if not datasets:
        sys.exit(f"No enabled [[dataset]] entries in {config_path}. See README.md.")
    import anndata as ad
    sources = []
    for ds in datasets:
        a = ad.read_h5ad(os.path.expanduser(ds["path"]), backed="r")
        sources.append((ds, _gene_symbols(a, ds.get("gene_col", ""))))
        a.file.close()
    universe = build_gene_universe(cfg, sources)
    print(f"Gene universe: {len(universe)} genes")
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.save(os.path.join(CACHE_DIR, "genes.npy"), universe)
    split = cfg.get("split", {})
    made = []
    for ds in datasets:
        for c in ingest_dataset(ds, universe, held_out=set(split.get("val", []))):
            made.append(c["context"])
            save_context(c)
    missing = set(split.get("val", [])) - set(made)
    if missing:
        sys.exit(f"[split] val names contexts that were not produced: {sorted(missing)} (have {made})")
    targets = cfg.get("submit", {}).get("targets")
    with open(os.path.join(CACHE_DIR, "split.json"), "w") as f:
        json.dump({"val": split.get("val", []), "config": os.path.abspath(config_path),
                   "submit_targets": os.path.abspath(os.path.expanduser(targets)) if targets else None}, f)
    print(f"Done. Validation (held-out) contexts: {split.get('val', [])}")

# ---------------------------------------------------------------------------
# Runtime utilities (imported from train.py)
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """One cell line. For held-out contexts only the control-side fields are populated."""
    name: str
    dataset: str
    genes: np.ndarray          # universe indices of the genes measured in this context
    ctrl_cells: sp.csr_matrix  # raw control counts [cells, len(genes)]
    ctrl_sum: np.ndarray       # summed raw control counts [len(genes)]
    ctrl_n: int
    ctrl_lib: np.ndarray       # library size of each control cell over all genes in its source file
    perts: np.ndarray = None       # perturbed target genes (training contexts only)
    pert_sum: np.ndarray = None    # summed raw counts per perturbation [P, len(genes)]
    pert_n: np.ndarray = None      # cells per perturbation [P]


@dataclass
class Task:
    """A zero-shot prediction task: predict `targets` in `context` from its controls alone."""
    context: Context
    targets: np.ndarray


def gene_universe():
    return np.load(os.path.join(CACHE_DIR, "genes.npy"), allow_pickle=True)


def _load_npz(name):
    z = np.load(os.path.join(CONTEXT_DIR, f"{name}.npz"), allow_pickle=True)
    return z, json.loads(str(z["meta"])), _load_csr(z, "ctrl")


def _control_sum(z):
    return z["p_sum"][-1], int(z["p_n"][-1])          # the control is the last panel row


def list_contexts():
    if not os.path.isdir(CONTEXT_DIR):
        sys.exit(f"No prepared data in {CONTEXT_DIR}. Run `uv run prepare.py` first.")
    return sorted(f[:-4] for f in os.listdir(CONTEXT_DIR) if f.endswith(".npz"))


def _split():
    with open(os.path.join(CACHE_DIR, "split.json")) as f:
        return json.load(f)


def val_contexts():
    return _split()["val"]


def dataset_for_context(name):
    """The [[dataset]] config entry a prepared context came from (used by official_eval.py)."""
    _, meta, _ = _load_npz(name)
    with open(_split()["config"], "rb") as f:
        cfg = tomllib.load(f)
    return next(d for d in cfg["dataset"] if d["name"] == meta["dataset"])


def _context(name, with_perts):
    z, meta, ctrl = _load_npz(name)
    s, n = _control_sum(z)
    c = Context(name=name, dataset=meta["dataset"], genes=z["genes"], ctrl_cells=ctrl, ctrl_sum=s, ctrl_n=n,
                ctrl_lib=z["ctrl_lib"])
    if with_perts:
        c.perts, c.pert_sum, c.pert_n = z["perts"], z["train_sum"], z["train_n"]
    return c, meta, z


def load_training_contexts():
    """All perturbation data the model may learn from: every context that is not held out."""
    held = set(val_contexts())
    out = []
    for name in list_contexts():
        if name in held:
            continue
        c, meta, _ = _context(name, with_perts=True)
        if meta["role"] == "train":
            out.append(c)
    return out


def load_val_tasks():
    """Held-out contexts: controls + the panel of targets to predict. No answers."""
    tasks = []
    for name in val_contexts():
        c, _, z = _context(name, with_perts=False)
        tasks.append(Task(context=c, targets=z["panel"]))
    return tasks


def load_submit_tasks():
    """Controls-only contexts (role = "submit", e.g. the official VCC 2026 NTC files), each with
    the target panel listed in the config's [submit] targets file."""
    tasks = []
    targets_file = _split().get("submit_targets")
    for name in list_contexts():
        c, meta, _ = _context(name, with_perts=False)
        if meta["role"] == "submit":
            if not targets_file or not os.path.exists(targets_file):
                raise ValueError("submit contexts need `[submit] targets = <file>` in the config (one gene per line)")
            tasks.append(Task(context=c, targets=np.array(_read_lines(targets_file))))
    return tasks

# ---------------------------------------------------------------------------
# Evaluation: a fast re-implementation of the cell-eval2 `vcc2026` scoring
# ---------------------------------------------------------------------------
#
# Predictions are log2 fold changes vs the context's own control, per target and gene:
#     preds[context_name] = float array [len(task.targets), len(gene_universe())]
# (columns of genes not measured in the context are ignored). A predicted fold change
# r = 2**lfc means "this target's cells are the control cells with gene g scaled by r",
# which is exactly how submit.py / official_eval.py emit cells for the official scorer.
#
# Six metrics, as in cell-eval2's vcc2026 preset: pds (cosine discrimination), mse
# (sampling-corrected expression error), nmae (log2FC error on real DE genes), fid
# (direction fidelity x yield), reach (direction reach), jac (DE-set Jaccard). DE uses a
# Wilcoxon rank-sum test vs the control cells, BH per target, genes with control CPM > 5.
# The real side is computed from cells once and cached. The predicted side is computed from
# the cells the prediction would be submitted as: real-sized groups of resampled control cells,
# scaled per gene and rounded to integer counts (fixed seed). Fast on a GPU, slower on CPU.
# Each metric is then put on the VCC 2026 scale, per context:
#     0 = generic-response baseline (mean perturbation response; a transductive oracle)
#     1 = split-half replicate (half of the real cells predicting the other half)
# and val_score is the unweighted mean over the six metrics and the held-out contexts.

METRICS = ("pds", "mse", "nmae", "fid", "reach", "jac")
SCORER_VERSION = "proxy-v6"
N_SPLITS = 5                # split-half replicates averaged for the 1-anchor (as cell-eval2)
EMIT_SEED = 0               # fixed seed for the cells a prediction is turned into (scoring is deterministic)


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bulk(sums):
    """cell-eval2 `bulk_lognorm`: normalise the summed counts of a group, then log1p once."""
    sums = np.asarray(sums, dtype=np.float64)
    tot = sums.sum(-1, keepdims=True)
    return np.log1p(BULK_TARGET_SUM * sums / np.where(tot > 0, tot, 1.0))


def _stochastic_round(x, rng):
    fl = np.floor(x)
    return fl + (rng.random(x.shape) < (x - fl))


def emit_cells(ctrl_cells, scale, n, rng):
    """Cells for one prediction: resampled control cells, gene g scaled by scale[g], stochastically
    rounded to integer counts. This is how a prediction becomes a submission."""
    pick = rng.integers(0, ctrl_cells.shape[0], size=n)
    X = sp.csr_matrix(sp.csr_matrix(ctrl_cells[pick], dtype=np.float64) @ sp.diags(np.asarray(scale, dtype=np.float64)))
    X.data = _stochastic_round(X.data, rng)
    X.eliminate_zeros()
    return X.astype(np.float32)


class _RankReference:
    """Control cells sorted per gene: Mann-Whitney AUC of any cell group by binary search."""

    def __init__(self, cpm):
        import torch
        self.n = cpm.shape[0]
        self.sorted = torch.sort(cpm.T.contiguous(), dim=1).values      # [G, n0]
        self.zero_frac = (cpm == 0).float().mean(0)                       # [G]

    def auc(self, cpm):
        """cpm [n1, G] -> (AUC [G] = P(x > ref) + P(x = ref)/2, zero fraction of the group [G])"""
        import torch
        v = cpm.T.contiguous()
        lo = torch.searchsorted(self.sorted, v, right=False)
        hi = torch.searchsorted(self.sorted, v, right=True)
        u = (lo + hi).double().sum(1) / 2.0
        return (u / (cpm.shape[0] * self.n)).cpu().numpy(), (cpm == 0).double().mean(0).cpu().numpy()


def _dense_cpm(X, lib, cols, dev):
    import torch
    m = sp.csr_matrix(X)[:, cols].toarray().astype(np.float64)
    m *= (DE_TARGET_SUM / np.maximum(np.asarray(lib, dtype=np.float64), 1.0))[:, None]
    return torch.as_tensor(m, dtype=torch.float32, device=dev)


def _jackknife(counts):
    """cell-eval2's sampling correction for a group's bulk_lognorm profile: the delete-1 jackknife
    variance, summed over genes. counts: dense raw counts [n, G] (torch)."""
    import torch
    n = counts.shape[0]
    if n < 2:
        return 0.0
    y = counts.double()
    P = y.sum(0)
    rest = (P.sum() - y.sum(1)).clamp_min(1e-12)[:, None]
    v = torch.log1p(BULK_TARGET_SUM * (P[None, :] - y) / rest)
    return float(max(0.0, (n - 1) / n * ((v - v.mean(0)) ** 2).sum().item()))


def _bh(p):
    """Benjamini-Hochberg per row; NaN p-values are ignored and get NaN."""
    p = np.asarray(p, dtype=np.float64)
    out = np.full_like(p, np.nan)
    for i in range(p.shape[0]):
        ok = np.flatnonzero(np.isfinite(p[i]))
        if not len(ok):
            continue
        order = ok[np.argsort(p[i, ok], kind="stable")]
        m = len(order)
        q = p[i, order] * m / np.arange(1, m + 1)
        out[i, order] = np.minimum(np.minimum.accumulate(q[::-1])[::-1], 1.0)
    return out


def _mwu_pvalues(auc, zf1, n1, zf0, n0):
    """Two-sided Wilcoxon rank-sum p-values (normal approximation, tie correction for the zero
    block) from AUC and group sizes. n1 [P, 1] or scalar; n0 scalar."""
    n1 = np.asarray(n1, dtype=np.float64)
    N = n1 + n0
    t = zf1 * n1 + zf0 * n0
    tie = np.clip(1.0 - (t ** 3 - t) / (N ** 3 - N), 0.0, 1.0)
    sigma = np.sqrt(n1 * n0 * (N + 1) / 12.0 * tie)
    z = np.where(sigma > 0, (auc - 0.5) * n1 * n0 / np.where(sigma > 0, sigma, 1.0), 0.0)
    return 2.0 * sstats.norm.sf(np.abs(z))


def _de_table(lfc, pval):
    return lfc, pval, _bh(pval)


def _pds(e_pred, e_real, keep_cols):
    """pds_cosine: 1 - midrank(true match) / (n - 1), cosine distance, panel targets removed."""
    a, b = e_pred[:, keep_cols], e_real[:, keep_cols]
    nn = np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))
    D = np.clip(1.0 - (a @ b.T) / np.where(nn > 0, nn, np.inf), 0.0, 2.0)
    d_true = np.diag(D)[:, None]
    rank = (D < d_true).sum(1) + ((D == d_true).sum(1) - 1) / 2.0
    return float(np.mean(1.0 - rank / (D.shape[0] - 1)))


def _mse_norm(m_pred, c_pred, m_real, c_real, m_ctrl, c_ctrl, tcol, budget=False):
    """expr_mse_unbiased_capped_norm: sampling-corrected squared error over the real effect size,
    ratio of sums over perturbations; each perturbation's own target gene is left out.
    The predicted side's correction is capped at the real side's and (budget=True, the scored
    leg) shrunk so that it never exceeds the predicted profiles' own across-perturbation spread."""
    P, G = m_real.shape
    mask = np.ones((P, G), dtype=bool)
    has_t = tcol >= 0
    mask[np.flatnonzero(has_t), tcol[has_t]] = False
    gp = mask.sum(1)
    c_tilde = np.minimum(c_pred, c_real)
    if budget:
        W = mask / gp[:, None]
        wsum = W.sum(0)
        wmean = (W * m_pred).sum(0) / np.where(wsum > 0, wsum, 1.0)
        spread = (W * (m_pred - wmean) ** 2).sum(0)
        B = np.maximum(spread[mask.sum(0) >= 2], 0.0).sum()
        claim = (c_tilde / gp).sum()
        if claim > 0 and B / claim < 1:
            c_tilde = c_tilde * B / claim
    num = (((m_pred - m_real) ** 2) * mask).sum(1) - c_tilde - c_real
    den = (((m_real - m_ctrl[None, :]) ** 2) * mask).sum(1) - c_real - c_ctrl
    return float(np.sum(num / gp) / np.sum(den / gp))


def _de_metrics(real, pred, tcol, nmae_gate=None):
    """lfc_nmae, direction fidelity x yield, direction reach, sig jaccard. real/pred = (lfc, p, padj).
    nmae_gate = (numerator lfc, denominator lfc, gate) overrides lfc_nmae's pieces (anchor only)."""
    rl, _, rq = real
    pl_, pp, pq = pred
    P, G = rl.shape
    nmae_lfc, nmae_ref = (rl, rl) if nmae_gate is None else nmae_gate[:2]
    nmae, fyr, reach, jac = [], [], [], []
    gene_idx = np.arange(G)
    for i in range(P):
        keep = np.ones(G, dtype=bool)
        if tcol[i] >= 0:
            keep[tcol[i]] = False
        rsig = (rq[i] < P_ADJ) & keep
        psig = (pq[i] < P_ADJ) & keep
        gate = (nmae_gate[2][i] if nmae_gate is not None else rsig) & np.isfinite(nmae_ref[i])
        if gate.sum() >= 10:
            den = np.mean(np.abs(nmae_ref[i, gate]))
            if np.isfinite(den) and den > 0:
                lh = np.where(np.isfinite(pl_[i]), pl_[i], 0.0)
                nmae.append(np.mean(np.abs(lh[gate] - nmae_lfc[i, gate])) / den)
        in_denom = np.isfinite(rl[i]) & (rl[i] != 0) & keep
        match = in_denom & np.isfinite(pl_[i]) & (pl_[i] != 0) & (np.sign(pl_[i]) == np.sign(rl[i]))
        n_conf = int(rsig.sum())
        n_pred = int((psig & in_denom).sum())
        if max(n_pred, n_conf) > 0:
            fyr.append(int((psig & match).sum()) / max(n_pred, n_conf))
        if n_conf > 0:
            pool = np.flatnonzero(rsig)
            qv = np.where(np.isfinite(pq[i, pool]), pq[i, pool], np.inf)
            pv = np.where(np.isfinite(pp[i, pool]), pp[i, pool], np.inf)
            al = np.where(np.isfinite(pl_[i, pool]), np.abs(pl_[i, pool]), 0.0)
            order = np.lexsort((gene_idx[pool], -al, pv, qv, -psig[pool].astype(int)))
            cd = np.cumsum(in_denom[pool][order])
            cm = np.cumsum(match[pool][order])
            ok = (cd > 0) & (cm >= 0.9 * cd)
            reach.append((cd[ok].max() if ok.any() else 0) / n_conf)
        union = int((rsig | psig).sum())
        jac.append(int((rsig & psig).sum()) / union if union else 1.0)
    f = lambda v: float(np.mean(v)) if len(v) else float("nan")
    return dict(nmae=f(nmae), fid=f(fyr), reach=f(reach), jac=f(jac))


class _Reference:
    """Everything real-side for one held-out context. Independent of the submission, so it is
    computed once and cached: the real DE table and the baseline / replicate metric values."""

    def __init__(self, name):
        z, meta, ctrl = _load_npz(name)
        self.name = name
        self.panel, self.genes = z["panel"], z["genes"]
        universe = gene_universe()
        col_of = {universe[g]: j for j, g in enumerate(self.genes)}
        self.tcol = np.array([col_of.get(t, -1) for t in self.panel])
        # panel rows 0..P-1, control last
        self.n = z["p_n"].astype(np.float64)
        self.sums = z["p_sum"].astype(np.float64)
        self.bulk = _bulk(self.sums)
        self.mean_cpm = z["p_cpm"].astype(np.float64) / np.maximum(self.n, 1.0)[:, None]
        self.ctrl_sum = self.sums[-1]
        self.gate = np.flatnonzero(self.mean_cpm[-1] > 5.0)        # DE genes: control CPM > 5
        gpos = {g: k for k, g in enumerate(self.gate)}
        self.tcol_gate = np.array([gpos.get(t, -1) for t in self.tcol])
        self.pds_cols = np.setdiff1d(np.arange(len(self.genes)), self.tcol[self.tcol >= 0])
        cache = os.path.join(CACHE_DIR, "anchors", f"{name}.npz")
        if os.path.exists(cache) and str(np.load(cache, allow_pickle=True)["version"]) == SCORER_VERSION:
            d = np.load(cache, allow_pickle=True)
            self.tables = {"full": (d["full_lfc"], d["full_p"], d["full_q"])}
            self.c_full = d["c_full"]
            self.b, self.r = json.loads(str(d["baseline"])), json.loads(str(d["replicate"]))
            return
        t0 = time.time()
        self._build_tables(z, ctrl)
        # the official baseline arm is emitted WITHOUT rounding (fractional counts); submissions are integers
        self.b = self.score_scales(self.baseline_scales(), rounding=False)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, version=SCORER_VERSION, baseline=json.dumps(self.b), replicate=json.dumps(self.r),
                 c_full=self.c_full, **{f"full_{x}": v for x, v in zip(("lfc", "p", "q"), self.tables["full"])})
        print(f"[{name}] built scorer reference in {time.time() - t0:.0f}s")

    def _build_tables(self, z, ctrl):
        """Real DE table (all cells) and the split-half replicate anchor (N_SPLITS random splits of
        every group into two halves, each half tested against its own half of the controls)."""
        import torch
        dev, g = _device(), self.gate
        P, S = len(self.panel), ctrl.shape[0]
        cells = _load_csr(z, "cells")
        group, lib = z["cells_group"], z["cells_lib"]
        ctrl_cpm = _dense_cpm(ctrl, z["ctrl_lib"], g, dev)
        full_ref = _RankReference(ctrl_cpm)
        n0 = self.n[-1]
        dense = lambda m: torch.as_tensor(sp.csr_matrix(m).toarray(), dtype=torch.float32, device=dev)
        ctrl_counts = dense(ctrl)
        # jackknife corrections: real groups from their cells; the control from its stored sample,
        # rescaled to the full control count (the variance of a mean scales as 1/n)
        self.c_full = np.zeros(P + 1)
        self.c_full[-1] = _jackknife(ctrl_counts) * S / n0
        splits = []
        for seed in np.random.SeedSequence(0).generate_state(N_SPLITS):
            rng = np.random.default_rng(seed)
            perm = rng.permutation(S)
            halves = (perm[:S // 2], perm[S // 2: 2 * (S // 2)])
            splits.append(dict(
                rng=rng, halves=halves,
                refs=[_RankReference(ctrl_cpm[torch.as_tensor(h, device=dev)]) for h in halves],
                ctrl_mean=[ctrl_cpm[torch.as_tensor(h, device=dev)].double().mean(0).cpu().numpy() for h in halves],
                ctrl_sum=[np.asarray(ctrl[h].sum(0)).ravel() for h in halves],
                c=np.zeros((2, P + 1)),
                auc=np.full((2, P, len(g)), 0.5), zf=np.zeros((2, P, len(g))), mean=np.zeros((2, P, len(g))),
                sums=np.zeros((2, P, len(self.genes))), n1=np.zeros((2, P, 1))))
        auc, zf, n1 = np.full((P, len(g)), 0.5), np.zeros((P, len(g))), np.zeros((P, 1))
        order = np.argsort(group, kind="stable")
        bounds = np.searchsorted(group[order], np.arange(P + 1))
        for i in range(P):
            idx = order[bounds[i]:bounds[i + 1]]
            if len(idx) < 2:
                continue
            x = _dense_cpm(cells[idx], lib[idx], g, dev)
            counts = dense(cells[idx])
            auc[i], zf[i] = full_ref.auc(x)
            n1[i] = len(idx)
            self.c_full[i] = _jackknife(counts)
            for sp_ in splits:
                perm = sp_["rng"].permutation(len(idx))
                h = len(idx) // 2
                for k, part in enumerate((perm[:h], perm[h:2 * h])):
                    xs = x[torch.as_tensor(part, device=dev)]
                    sp_["auc"][k, i], sp_["zf"][k, i] = sp_["refs"][k].auc(xs)
                    sp_["mean"][k, i] = xs.double().mean(0).cpu().numpy()
                    sp_["sums"][k, i] = np.asarray(cells[idx[part]].sum(0)).ravel()
                    sp_["n1"][k, i] = h
                    sp_["c"][k, i] = _jackknife(counts[torch.as_tensor(part, device=dev)])
        lfc = np.log2((self.mean_cpm[:-1][:, g] + 1e-9) / (self.mean_cpm[-1][g] + 1e-9))
        self.tables = {"full": _de_table(lfc, _mwu_pvalues(auc, zf, n1, full_ref.zero_frac.double().cpu().numpy()[None, :], n0))}
        # replicate anchor: half B predicts half A, averaged over the splits
        rl, _, rq = self.tables["full"]
        gate = rq < P_ADJ
        has_t = self.tcol_gate >= 0
        gate[np.flatnonzero(has_t), self.tcol_gate[has_t]] = False
        per_split = []
        for sp_ in splits:
            for k, h in enumerate(sp_["halves"]):
                sp_["c"][k, -1] = _jackknife(ctrl_counts[torch.as_tensor(h, device=dev)]) * S / n0
            tabs, bulks = [], []
            for k in range(2):
                l = np.log2((sp_["mean"][k] + 1e-9) / (sp_["ctrl_mean"][k] + 1e-9)[None, :])
                zf0 = sp_["refs"][k].zero_frac.double().cpu().numpy()[None, :]
                tabs.append(_de_table(l, _mwu_pvalues(sp_["auc"][k], sp_["zf"][k], sp_["n1"][k], zf0, n0 // 2)))
                bulks.append(_bulk(np.vstack([sp_["sums"][k], sp_["ctrl_sum"][k][None, :]])))
            A, B = bulks
            cA, cB = sp_["c"]
            m = dict(pds=_pds(B[:-1] - B[-1], A[:-1] - A[-1], self.pds_cols),
                     mse=_mse_norm(B[:-1], cB[:-1], A[:-1], cA[:-1], A[-1], cA[-1], self.tcol))
            m.update(_de_metrics(tabs[0], tabs[1], self.tcol_gate))
            # lfc_nmae's anchor ("full_gate_raw"): gate + denominator from the full table, numerator |l_A - l_B|
            m["nmae"] = _de_metrics(self.tables["full"], tabs[1], self.tcol_gate, nmae_gate=(tabs[0][0], rl, gate))["nmae"]
            per_split.append(m)
        self.r = {k: float(np.mean([m[k] for m in per_split])) for k in METRICS}
        self._init_emission(z, ctrl, full_ref)

    def _init_emission(self, z, ctrl, ref):
        """State for turning predictions into cells (lazy: also needed when tables come from cache)."""
        import torch
        dev = _device()
        self._dev, self._ref = dev, ref
        self._ctrl_counts = torch.as_tensor(sp.csr_matrix(ctrl).toarray(), dtype=torch.float32, device=dev)
        # counts of each control cell on genes outside the context's measured set (never rescaled)
        self._ctrl_lib_out = torch.as_tensor(np.maximum(z["ctrl_lib"] - np.asarray(ctrl.sum(1)).ravel(), 0.0),
                                             dtype=torch.float32, device=dev)
        self._ctrl_zf = ref.zero_frac.double().cpu().numpy()

    def _emit(self, scales, rounding=True):
        """Turn a prediction into the cells it would be submitted as (real-sized groups of resampled
        control cells, scaled per gene, rounded to integers) and summarise them like cell-eval2 does:
        DE table vs the real controls, summed counts, and the jackknife sampling correction."""
        import torch
        if not hasattr(self, "_ref"):
            z, _, ctrl = _load_npz(self.name)
            self._init_emission(z, ctrl, _RankReference(_dense_cpm(ctrl, z["ctrl_lib"], self.gate, _device())))
        dev, g = self._dev, torch.as_tensor(self.gate, device=self._dev)
        P, S = scales.shape[0], self._ctrl_counts.shape[0]
        n1 = self.n[:-1].astype(int)
        gen = torch.Generator(device=dev).manual_seed(EMIT_SEED)
        G = len(self.gate)
        auc, zf, mean = np.zeros((P, G)), np.zeros((P, G)), np.zeros((P, G))
        sums, c_pred = np.zeros_like(scales), np.zeros(P)
        for i in range(P):
            idx = torch.randint(0, S, (max(n1[i], 1),), generator=gen, device=dev)
            x = self._ctrl_counts[idx] * torch.as_tensor(scales[i], dtype=torch.float32, device=dev)
            if rounding:
                fl = torch.floor(x)
                x = fl + (torch.rand(x.shape, generator=gen, device=dev) < (x - fl)).float()
            lib = (x.sum(1) + self._ctrl_lib_out[idx]).clamp_min(1.0)
            cpm = x[:, g] * (DE_TARGET_SUM / lib)[:, None]
            auc[i], zf[i] = self._ref.auc(cpm)
            mean[i] = cpm.double().mean(0).cpu().numpy()
            sums[i] = x.double().sum(0).cpu().numpy()
            c_pred[i] = _jackknife(x)
        lfc = np.log2((mean + 1e-9) / (self.mean_cpm[-1][self.gate] + 1e-9))
        pval = _mwu_pvalues(auc, zf, n1[:, None].astype(np.float64), self._ctrl_zf[None, :], self.n[-1])
        return _de_table(lfc, pval), sums, c_pred

    def score_scales(self, scales, rounding=True):
        """Raw metric values for a prediction given as per-gene scale factors [P, G_context]."""
        table, sums, c_pred = self._emit(scales, rounding)
        m_ctrl, m_pred = self.bulk[-1], _bulk(sums)
        out = dict(
            pds=_pds(m_pred - m_ctrl, self.bulk[:-1] - m_ctrl, self.pds_cols),
            mse=_mse_norm(m_pred, c_pred, self.bulk[:-1], self.c_full[:-1], m_ctrl, self.c_full[-1], self.tcol,
                          budget=True),
        )
        out.update(_de_metrics(self.tables["full"], table, self.tcol_gate))
        return out

    def baseline_scales(self):
        """Generic-response baseline: mean raw-count pseudobulk over panel perturbations
        (omitting, per gene, the perturbation that targets it), divided by the control mean."""
        n = self.n
        pb = self.sums[:-1] / np.maximum(n[:-1], 1.0)[:, None]
        tot, cnt = pb.sum(0), np.full(pb.shape[1], float(pb.shape[0]))
        for i, t in enumerate(self.tcol):
            if t >= 0:
                tot[t] -= pb[i, t]
                cnt[t] -= 1
        ctrl_mean = self.ctrl_sum / n[-1]
        r = np.where(ctrl_mean > 0, (tot / np.maximum(cnt, 1)) / np.where(ctrl_mean > 0, ctrl_mean, 1.0), 0.0)
        return np.tile(r, (len(self.panel), 1))


def _scale_metric(name, u, b, r):
    """Raw value -> VCC 2026 scale (0 = baseline, 1 = replicate), with cell-eval2's clamps."""
    lower_better = name in ("nmae", "mse")
    if not np.isfinite(r - b) or (b - r if lower_better else r - b) <= 0:
        return float("nan")  # degenerate scale: metric skipped for this context
    if name == "nmae":
        return max(-6.0, 1.0 - (u - r) / (b - r)) if np.isfinite(u) else -6.0
    if name == "mse":
        return float(np.clip((b - u) / (b - r), 0.0, 1.0)) if np.isfinite(u) else 0.0
    return (u - b) / (r - b) if np.isfinite(u) else (0.0 - b) / (r - b)


def predictions_to_scales(task, lfc):
    """[len(targets), n_universe] log2 fold changes -> per-gene scale factors on the context's genes."""
    lfc = np.asarray(lfc, dtype=np.float64)
    shape = (len(task.targets), len(gene_universe()))
    if lfc.shape != shape:
        raise ValueError(f"{task.context.name}: predictions must be shaped {shape}, got {lfc.shape}")
    if not np.all(np.isfinite(lfc)):
        raise ValueError(f"{task.context.name}: predictions contain NaN/inf")
    return np.exp2(np.clip(lfc[:, task.context.genes], -LFC_CLIP, LFC_CLIP))


def evaluate_predictions(preds, verbose=True):
    """Score {context_name: lfc [n_targets, n_universe]} on every held-out context.

    Returns a dict with `val_score` (mean over contexts of the mean of the six scaled
    metrics: 0 = generic baseline, 1 = split-half replicate) plus per-context details."""
    tasks = {t.context.name: t for t in load_val_tasks()}
    missing = set(tasks) - set(preds)
    if missing:
        raise ValueError(f"missing predictions for held-out contexts: {sorted(missing)}")
    report, ctx_scores = {}, []
    for name, task in tasks.items():
        ref = _Reference(name)
        u = ref.score_scales(predictions_to_scales(task, preds[name]))
        scaled = {m: _scale_metric(m, u[m], ref.b[m], ref.r[m]) for m in METRICS}
        ok = [v for v in scaled.values() if np.isfinite(v)]
        ctx_score = float(np.mean(ok)) if ok else float("nan")
        ctx_scores.append(ctx_score)
        report[name] = dict(score=ctx_score, scaled=scaled, raw=u, baseline=ref.b, replicate=ref.r)
        if verbose:
            print(f"[{name}] score {ctx_score:+.4f} | " + " ".join(f"{m} {scaled[m]:+.3f}" for m in METRICS))
    report["val_score"] = float(np.nanmean(ctx_scores))
    return report


def save_predictions(preds, kind):
    """Persist predictions (kind = "val" or "submit") for submit.py / official_eval.py."""
    d = os.path.join(CACHE_DIR, "predictions", kind)
    os.makedirs(d, exist_ok=True)
    for name, lfc in preds.items():
        np.save(os.path.join(d, f"{name}.npy"), np.asarray(lfc, dtype=np.float32))

# ---------------------------------------------------------------------------
# Synthetic data (smoke tests without downloads)
# ---------------------------------------------------------------------------

def make_synthetic(out_dir, n_genes=2000, n_targets=500, seed=0):
    """Small CRISPRi-like Perturb-seq datasets with structure shared across cell lines.

    Five cell lines (SYN_A..SYN_E) plus a controls-only one (SYN_F). A knockdown's response
    is a context-specific generic stress response plus a target-specific program shared
    across lines (gain-modulated by each line's basal expression), plus the knockdown itself.
    Only for checking that the pipeline runs: it says nothing about real biology.
    """
    import anndata as ad
    import pandas as pd
    rng = np.random.default_rng(seed)
    G = n_genes
    genes = np.array([f"GENE{i:05d}" for i in range(G)])
    base = rng.lognormal(0, 1.6, G)
    targets = rng.choice(G, n_targets, replace=False)
    programs = rng.normal(0, 1, (8, G)) * (rng.random((8, G)) < 0.06)
    loadings = rng.normal(0, 0.6, (n_targets, 8)) * (rng.random((n_targets, 8)) < 0.4)
    specific = loadings @ programs + rng.normal(0, 0.5, (n_targets, G)) * (rng.random((n_targets, G)) < 0.01)
    common_generic = rng.normal(0, 0.12, G)
    os.makedirs(out_dir, exist_ok=True)
    toml = ['# synthetic smoke-test config written by `prepare.py --synthetic`', '',
            '[split]', 'val = ["SYN_D", "SYN_E"]', '',
            '[submit]', f'targets = "{os.path.join(out_dir, "targets.txt")}"', '']
    for k, name in enumerate(["SYN_A", "SYN_B", "SYN_C", "SYN_D", "SYN_E", "SYN_F"]):
        b = base * rng.lognormal(0, 0.5, G)
        b /= b.sum()
        gain = 0.6 + 0.8 * rng.random() + 0.2 * np.tanh(np.log(b * G))
        generic = common_generic + rng.normal(0, 0.08, G)
        npert = 0 if name == "SYN_F" else int(rng.integers(250, 450))
        chosen = rng.choice(n_targets, npert, replace=False)
        n_cells_p, n_ctrl = 60, 1200
        X, lab = [], []

        def cells(prof, n):
            depth = rng.lognormal(np.log(6000), 0.35, n)
            mu = depth[:, None] * prof[None, :]
            return sp.csr_matrix(rng.poisson(rng.gamma(4.0, mu / 4.0)).astype(np.float32))

        X.append(cells(b, n_ctrl)); lab += ["non-targeting"] * n_ctrl
        for j in chosen:
            eff = generic + gain * specific[j]
            eff[targets[j]] = -2.0 - rng.random()
            p = b * np.exp2(eff); p /= p.sum()
            X.append(cells(p, n_cells_p)); lab += [genes[targets[j]]] * n_cells_p
        A = ad.AnnData(sp.vstack(X).tocsr(), obs=pd.DataFrame({"target_gene": lab}, index=[f"{name}_{i}" for i in range(len(lab))]),
                       var=pd.DataFrame(index=genes))
        path = os.path.join(out_dir, f"{name}.h5ad")
        A.write_h5ad(path)
        toml += ['[[dataset]]', f'name = "{name}"', f'path = "{path}"', 'pert_col = "target_gene"',
                 'control = "non-targeting"', f'context = "{name}"', 'gene_col = ""',
                 f'role = "{"submit" if name == "SYN_F" else "train"}"', '']
        print(f"  wrote {path}: {A.n_obs} cells, {npert} perturbations")
    with open(os.path.join(out_dir, "targets.txt"), "w") as f:
        f.write("\n".join(genes[targets[:100]]) + "\n")
    cfg = os.path.join(out_dir, "datasets.toml")
    with open(cfg, "w") as f:
        f.write("\n".join(toml))
    return cfg

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch on the Virtual Cell Challenge")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="datasets config (TOML)")
    parser.add_argument("--synthetic", action="store_true", help="generate + prepare a small synthetic dataset")
    parser.add_argument("--inspect", metavar="H5AD", help="print the obs/var columns of an h5ad (to fill in datasets.toml)")
    args = parser.parse_args()
    if args.inspect:
        import anndata as ad
        a = ad.read_h5ad(os.path.expanduser(args.inspect), backed="r")
        print(a)
        for col in a.obs.columns:
            vc = a.obs[col].astype(str).value_counts()
            print(f"  obs[{col!r}]: {len(vc)} values, e.g. {list(vc.index[:5])}")
        print(f"  var_names e.g. {list(a.var_names[:5])}")
        for col in a.var.columns:
            print(f"  var[{col!r}] e.g. {list(a.var[col].astype(str).values[:5])}")
        X = a.X[:5]
        X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
        print(f"  X[:5] integer counts: {bool(np.all(X == np.round(X)))}, row sums {X.sum(1).round(1).tolist()}")
        sys.exit(0)
    config = args.config
    if args.synthetic:
        config = make_synthetic(os.path.join(CACHE_DIR, "synthetic"))
    prepare(config)
    # build the baseline/replicate anchors now so the first experiment does not pay for it
    for name in val_contexts():
        ref = _Reference(name)
        print(f"[{name}] anchors: " + " ".join(f"{m} b={ref.b[m]:.3f} r={ref.r[m]:.3f}" for m in METRICS))
