# autoresearch: Virtual Cell Challenge 2026

This is an experiment to have the LLM do its own research on the Arc Institute
[Virtual Cell Challenge 2026](https://virtualcellchallenge.org/): predict how a cell line
that has **never been seen perturbed** responds to CRISPRi knockdowns, given only its
unperturbed (non-targeting control) cells and the list of genes to knock down.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `oct2`). The branch `vcc/<tag>` must not already exist: this is a fresh run.
2. **Create the branch**: `git checkout -b vcc/<tag>` from the current branch.
3. **Read the in-scope files**. The directory is small; read these for full context:
   - `README.md`: what the task is and how it maps to VCC 2026.
   - `prepare.py`: fixed constants, data ingestion, the data-loading API, and the scorer (`evaluate_predictions`). Do not modify.
   - `train.py`: the file you modify. The model, its training loop and its predictions.
4. **Verify data exists**: check that `~/.cache/autoresearch-vcc/contexts/` holds prepared `.npz` files and `split.json`. If not, tell the human to fill in `datasets.toml` and run `uv run prepare.py`.
5. **Initialize results.tsv**: create `results.tsv` with just the header row. The baseline is recorded after the first run.
6. **Confirm and go**: confirm the setup looks good.

Once you get confirmation, kick off the experimentation.

## The task, concretely

Data comes as *contexts* (cell lines). `train.py` gets:

- `load_training_contexts()`: every non-held-out cell line, with summed raw counts per perturbation (`pert_sum`, `pert_n`), its control cells, and its measured genes.
- `load_val_tasks()`: the held-out cell lines. For each: **only** its control cells and the panel of target genes to predict (up to 300). This is exactly the information VCC 2026 gives you.
- `load_submit_tasks()`: the official VCC contexts, if configured. Controls plus the official target list, with no answers anywhere.

It must output, per held-out context, a `[n_targets, n_genes]` array of **log2 fold changes vs that context's control**. A fold change r = 2^lfc means "this knockdown's cells are the control cells with gene g scaled by r". The scorer turns that into cells exactly as a submission would be built, then scores the cells.

## Experimentation

Each experiment trains for a **fixed time budget of 5 minutes** (wall clock, excluding startup, data loading and scoring). Launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py`. This is the only file you edit. Everything is fair game: the model, features, training objective, ensembling, nonparametric components, hyperparameters.
- Use the held-out contexts' **control cells** any way you like (they are the input; VCC hands them to you too).

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only and holds the split, the scorer and the constants.
- Read a held-out context's perturbation data by any route. That means no `np.load` of its `.npz` and no `_load_npz` or `_Reference` calls; only the public functions listed above. Peeking makes `val_score` meaningless.
- Install new packages or add dependencies. You can only use what is already in `pyproject.toml`.
- Tune anything on `evaluate_predictions` output from inside one run, such as picking a checkpoint by val score. The score is called once, at the end.

**The goal is simple: get the highest `val_score`.** It is the VCC 2026 score on the held-out lines: the mean of six metrics, each scaled so that 0 = the generic-response baseline (the average knockdown response of that very line, which is a strong, transductive comparator) and 1 = a replicate experiment (half the real cells predicting the other half). Scores above 1 are possible, and negative scores are common.

The six metrics are:

| metric | what it rewards |
|---|---|
| `pds` | each prediction is closer, by cosine, to its own true effect than to other targets' effects |
| `mse` | expression error vs the true pseudobulk, corrected for sampling noise and normalised by the effect size |
| `nmae` | absolute log2FC error on the genes that are truly differentially expressed (DE) |
| `fid` | of the genes your cells make significant, the share moving the right way (min of precision and recall) |
| `reach` | how deep into the true DE genes, ranked by your confidence, direction stays ≥90% correct |
| `jac` | Jaccard overlap of your significant gene set with the true one |

DE significance is a Wilcoxon test of your *emitted cells* against the real controls. So the **magnitude** of your fold changes decides which genes come out significant: shrinking everything toward 0 kills `fid`, `reach` and `jac`; exaggerating adds false positives. Calibration matters, not just direction.

**Simplicity criterion**: all else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a great outcome. With only 2-3 held-out lines, `val_score` can be gamed by overfitting to them. Prefer ideas with a biological or statistical reason to generalize, and be suspicious of large gains from fiddly constants.

**The first run**: always establish the baseline first, by running the training script as is.

## Output format

Once the script finishes it prints a summary like this:

```
---
val_score:        0.123456
pds:              0.512000
mse:              0.100000
nmae:             0.050000
fid:              0.090000
reach:            -0.020000
jac:              0.010000
training_seconds: 300.0
total_seconds:    341.7
num_steps:        42000
num_params_M:     6.10
```

Extract the key metric from the log file:

```
grep "^val_score:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated, because commas break in descriptions). Do not commit it; leave it untracked.

The TSV has a header row and 5 columns:

```
commit	val_score	pds	status	description
```

1. git commit hash (short, 7 chars)
2. val_score achieved (e.g. 0.123456); use -9.000000 for crashes
3. the pds component (a quick sanity signal); use 0.000000 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

## The experiment loop

The experiment runs on a dedicated branch (e.g. `vcc/oct2`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything; do NOT use tee or let output flood your context).
5. Read out the results: `grep "^val_score:\|^pds:\|^mse:\|^nmae:\|^fid:\|^reach:\|^jac:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix. If you can't get things to work after a few attempts, give up on the idea.
7. Record the results in the tsv.
8. If val_score improved (higher), you "advance" the branch, keeping the git commit.
9. If val_score is equal or worse, you git reset back to where you started.

**Timeout**: each experiment should take ~6 minutes total. If a run exceeds 12 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: if it's something dumb and easy to fix (a typo, a shape mismatch), fix it and re-run. If the idea itself is broken, log "crash" and move on.

**NEVER STOP**: once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. The human might be asleep and expects you to continue working *indefinitely* until you are manually stopped. If you run out of ideas, think harder: re-read the scorer in `prepare.py` for what exactly is rewarded, combine previous near-misses, or try more radical changes.

## Research directions (starting points, not a recipe)

- **Decompose the response.** lfc(p, c) = generic(c) + specific(p) × gain(c). The generic CRISPRi/stress response of a line is largely shared across knockdowns; the target-specific part transfers across lines best for genes in the same complex or pathway.
- **Transfer by similarity.** Weight source lines by basal-expression similarity to the target line; weight a target's responses across source lines by how reproducible they are.
- **Magnitude calibration.** Responses in a new line rarely have the same size as in the sources. Rescale per target or per line ("norm-restored transfer"), and shrink noisy genes, not all genes.
- **Unseen targets.** For targets never perturbed in training, borrow from co-expressed or functionally related targets, using the control cells' gene-gene correlations and the training responses' similarity structure, and include the target's own knockdown.
- **Expression-dependent effects.** A gene can only go down if it is expressed: condition per-gene responses on the target line's basal level, and on the knocked-down gene's own basal level (knocking down a non-expressed gene does little).
- **Objective.** Train on what is scored: weight genes by the DE gate (control CPM > 5), add a sign/rank-aware term, or tune a temperature on fold-change magnitude so significance calls match reality.
- **Ensembling** the parametric model with a nonparametric transfer estimate.
