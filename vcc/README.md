# autoresearch × Virtual Cell Challenge 2026

This applies the autoresearch loop (an agent edits one training file, trains for 5 minutes, keeps or discards the change, repeats) to the Arc Institute [Virtual Cell Challenge 2026](https://virtualcellchallenge.org/).

**The 2026 task is zero-shot.** You get the non-targeting control cells of a cell line nobody has seen perturbed, plus a panel of 300 CRISPRi target genes. You predict 400 single cells per knockdown. Scoring uses `cell-eval2`'s `vcc2026` preset: six metrics, each scaled to 0 = generic-response baseline and 1 = replicate experiment, averaged over contexts. No training set is released, so you train on public Perturb-seq data.

This directory turns public Perturb-seq data into that exact task by **holding out whole cell lines**. It also gives the agent a fast local copy of the official score to optimise.

## Files

```
datasets.toml     — which public h5ad files to use, and which cell lines to hold out (edit once)
prepare.py        — ingestion, data API, the scorer, constants (read-only for the agent)
train.py          — model + training + predictions (the ONE file the agent edits)
program.md        — instructions for the agent
official_eval.py  — re-score the last run with the real cell-eval2 vcc2026 pipeline (slow)
submit.py         — write submission .h5ad files for the official VCC contexts
```

## Quick start

**Requirements:** Python 3.11/3.12, [uv](https://docs.astral.sh/uv/), and one NVIDIA GPU for real data. Everything also runs on CPU, slowly.

```bash
cd vcc
uv sync

# 0. smoke test on synthetic data (no downloads, ~2 min on CPU)
VCC_CACHE_DIR=/tmp/vcc-smoke uv run prepare.py --synthetic
VCC_CACHE_DIR=/tmp/vcc-smoke VCC_TIME_BUDGET=30 uv run train.py

# 1. real data: download the files listed in datasets.toml, check their columns, fix paths
uv run prepare.py --inspect ~/data/replogle2022/K562_essential_raw_singlecell_01.h5ad
uv run prepare.py                      # one-time ingest (streams files; large ones take a while)

# 2. one experiment (~5 min training + scoring)
uv run train.py
```

Then point your agent at `program.md` as in the top-level README: *"Have a look at vcc/program.md and let's kick off a new experiment! Let's do the setup first."*

## Data

Configure sources in `datasets.toml`. Any raw-count CRISPRi Perturb-seq h5ad works:

- Replogle et al. 2022: K562 (essential and genome-wide) and RPE1.
- Nadig et al. 2025: HepG2 and Jurkat.
- The VCC 2025 H1 hESC set.
- The official VCC 2026 control files, as `role = "submit"`.

Set `[genes] reference` to an official VCC 2026 file so predictions live on the 18,533 submission genes, in order.

`[split] val` names the held-out lines, 2–3 of them. `train.py` never sees their perturbations, only their controls and the panel to predict, and `val_score` is computed on them. Choose lines that differ from the rest, as the VCC 2026 lines do (A549, MCF7, HT29, HAP1, BxPC3, K562).

## The scorer

`prepare.evaluate_predictions` re-implements the six scored `vcc2026` members of `cell-eval2` 0.16.0:

| metric | official name |
|---|---|
| `pds` | `pds_cosine` |
| `mse` | `expr_mse_unbiased_capped_norm` |
| `nmae` | `de_wilcoxon_lfc_nmae` |
| `fid` | `de_wilcoxon_direction_fidelity_yield_raw` |
| `reach` | `de_wilcoxon_direction_reach_raw` |
| `jac` | `de_wilcoxon_sig_jaccard` |

It follows the official definitions:

- **Expression metrics** use `bulk_lognorm` profiles at 5e4.
- **DE** uses a tie-corrected Wilcoxon on log1p CPM, BH per target, over genes with control CPM > 5. The panel's target genes are excluded.
- **Expression MSE** uses the delete-1 jackknife sampling correction, with the predicted side capped and budgeted.
- **The 0 anchor** is the generic-response baseline: fractional "dispersed" emission.
- **The 1 anchor** is the mean of 5 split-half replicates, with lfc_nmae's anchor built "full-gate".

**Predictions are scored as cells.** Each prediction becomes the integer-count cells a submission would contain: resampled control cells, scaled per gene, stochastically rounded. Those cells are then tested exactly as the official pipeline would test them. Everything real-side is computed once per held-out line and cached.

Two shortcuts keep it fast:

- **Control sample.** Ranks are taken against a stored sample of ≤2000 control cells, while the test uses the true control count for its variance.
- **Fixed emission seed.** The emitted cells use a fixed seed, so scores are deterministic.

**Agreement check.** On synthetic data, two held-out lines were scored both ways. The real-side DE table matched the official one exactly, with identical significant sets. Every raw metric, baseline and anchor agreed to within the noise of emitting different random cells:

| context | proxy | official |
|---|---|---|
| line D | 0.727 | 0.720 |
| line E | 0.857 | 0.840 |

Re-check on real data with `uv run official_eval.py`. It rebuilds the held-out cells from the source files and runs `cell-eval2 baseline`, `prep-real-bundle`, `run` and `score --real-bundle`, then prints both scores side by side. Expect minutes per context on GPU and much longer on CPU. Run it every few kept improvements, not every experiment.

## Submitting

Add the official control files as `role = "submit"` datasets and set `[submit] targets` before running `prepare.py`. After a `train.py` run, `uv run submit.py --combined` writes the h5ad files. Each context gets 400 integer-count cells per target on the full gene list. Check the official submission instructions for the exact obs column names and layout, and use the flags to match them.

The way predictions become cells (`prepare.emit_cells`) is fixed, so the local score matches what is submitted. Smarter emission is a possible human-level lever. One example is choosing control cells so the emitted means match the prediction more exactly. If you change it, change it in `prepare.py`, so the scorer and the submission stay consistent.
