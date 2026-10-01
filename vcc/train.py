"""
Zero-shot CRISPRi response model for the Virtual Cell Challenge 2026. This is the
file the agent edits.

Given the non-targeting control cells of a cell line it has never seen perturbed,
and a list of target genes, predict each knockdown's log2 fold change vs control for
every gene. Prediction semantics (see prepare.py): a fold change r = 2**lfc means
"this knockdown's cells are the control cells with gene g scaled by r".

Baseline model: lfc(p, c) = generic(c) + specific(p, c)
  - h(c): an encoding of cell line c's basal (control) expression profile
  - generic(c): the context's shared response to any CRISPRi knockdown, decoded from h(c)
  - specific(p, c): a learned per-target embedding, modulated by h(c) and decoded to genes
Targets never perturbed in any training context fall back to the generic response.
Usage: uv run train.py
"""

import time
import math

import numpy as np
import torch
import torch.nn as nn

from prepare import (TIME_BUDGET, gene_universe, load_training_contexts, load_val_tasks,
                     load_submit_tasks, evaluate_predictions, save_predictions, METRICS)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly)
# ---------------------------------------------------------------------------

EMBED_DIM = 64          # per-target embedding size
CTX_DIM = 16            # PCA components of the basal profile fed to the context encoder
HIDDEN = 256            # context encoder width
PSEUDO_CPM = 5.0        # pseudocount (counts per million) for the training log fold changes
LOWEXPR_WEIGHT = 0.2    # loss weight for genes below the DE gate (control CPM <= 5)
BATCH_PERTS = 128       # perturbations per step (all from one training context)
LR = 3e-3
WEIGHT_DECAY = 1e-4
CTX_NOISE = 0.1         # noise on the context code: there are only a handful of training contexts
WARMDOWN_RATIO = 0.3    # fraction of the budget spent decaying the learning rate to 0
SEED = 42

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

universe = gene_universe()
G = len(universe)
gene_index = {g: i for i, g in enumerate(universe)}
train_ctx = load_training_contexts()
val_tasks = load_val_tasks()
try:
    submit_tasks = load_submit_tasks()
except ValueError as e:
    print(f"(no submission predictions: {e})")
    submit_tasks = []
print(f"universe {G} genes | train contexts {[c.name for c in train_ctx]} | "
      f"held out {[t.context.name for t in val_tasks]} | submit {[t.context.name for t in submit_tasks]}")


def cpm(sums):
    sums = np.asarray(sums, dtype=np.float64)
    return 1e6 * sums / np.maximum(sums.sum(-1, keepdims=True), 1.0)


def basal_profile(ctx):
    """log1p CPM of the summed control counts, on the gene universe (0 where unmeasured)."""
    x = np.zeros(G, dtype=np.float32)
    x[ctx.genes] = np.log1p(cpm(ctx.ctrl_sum))
    return x


# training targets: per (context, perturbation) log2 fold change vs that context's control
targets = sorted({p for c in train_ctx for p in c.perts})
target_id = {p: i for i, p in enumerate(targets)}
ctx_data = []
for c in train_ctx:
    ctrl = cpm(c.ctrl_sum)
    lfc = np.zeros((len(c.perts), G), dtype=np.float32)
    lfc[:, c.genes] = np.log2((cpm(c.pert_sum) + PSEUDO_CPM) / (ctrl + PSEUDO_CPM))
    w = np.zeros(G, dtype=np.float32)
    w[c.genes] = np.where(ctrl > 5.0, 1.0, LOWEXPR_WEIGHT)
    ctx_data.append(dict(
        name=c.name,
        y=torch.tensor(lfc, device=device),
        w=torch.tensor(w, device=device),
        tid=torch.tensor([target_id[p] for p in c.perts], device=device),
        tcol=torch.tensor([gene_index.get(p, -1) for p in c.perts], device=device),
    ))

# context features: PCA of basal profiles over every context we can see controls for
all_ctx = train_ctx + [t.context for t in val_tasks] + [t.context for t in submit_tasks]
basal = np.stack([basal_profile(c) for c in all_ctx])
mu = basal.mean(0)
_, _, Vt = np.linalg.svd(basal - mu, full_matrices=False)
k = min(CTX_DIM, Vt.shape[0])
proj = Vt[:k].T
feat = (basal - mu) @ proj
feat = feat / (feat.std(0, keepdims=True) + 1e-6)
ctx_feat = {c.name: torch.tensor(np.pad(f, (0, CTX_DIM - k)), dtype=torch.float32, device=device)
            for c, f in zip(all_ctx, feat)}
basal_t = {c.name: torch.tensor(b, device=device) for c, b in zip(all_ctx, basal)}

# mean observed knockdown of the target gene itself (used for the target's own column)
kd = [d["y"][i, t].item() for d in ctx_data for i, t in enumerate(d["tcol"].tolist()) if t >= 0]
KNOCKDOWN_LFC = float(np.median(kd)) if kd else -2.0

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ResponseModel(nn.Module):
    def __init__(self, n_targets):
        super().__init__()
        self.ctx_enc = nn.Sequential(nn.Linear(CTX_DIM, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU())
        self.generic0 = nn.Parameter(torch.zeros(G))
        self.generic = nn.Linear(HIDDEN, G)
        self.embed = nn.Embedding(n_targets, EMBED_DIM)
        self.gate = nn.Linear(HIDDEN, EMBED_DIM)
        self.decode = nn.Linear(EMBED_DIM, G, bias=False)
        self.gain = nn.Linear(HIDDEN, G)
        nn.init.normal_(self.embed.weight, std=0.1)
        for m in (self.generic, self.gate, self.gain):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, ctx_code, tid):
        """ctx_code [CTX_DIM]; tid [B] target ids (-1 = never seen) -> lfc [B, G]"""
        h = self.ctx_enc(ctx_code)
        generic = self.generic0 + self.generic(h)
        e = self.embed(tid.clamp_min(0)) * (tid >= 0).float()[:, None]
        e = e * (1 + self.gate(h))
        specific = self.decode(e) * torch.exp(self.gain(h))
        return generic[None, :] + specific


model = ResponseModel(len(targets)).to(device)
num_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
print(f"params {num_params / 1e6:.2f}M | {len(targets)} training targets | knockdown lfc {KNOCKDOWN_LFC:.2f}")

# ---------------------------------------------------------------------------
# Training (fixed wall-clock budget)
# ---------------------------------------------------------------------------

sizes = np.array([len(d["tid"]) for d in ctx_data], dtype=np.float64)
probs = sizes / sizes.sum()
step, train_time, smooth = 0, 0.0, None
t_train = time.time()
while True:
    progress = train_time / TIME_BUDGET
    lr_mult = 1.0 if progress < 1 - WARMDOWN_RATIO else max(0.0, (1 - progress) / WARMDOWN_RATIO)
    for g in opt.param_groups:
        g["lr"] = LR * lr_mult
    d = ctx_data[np.random.choice(len(ctx_data), p=probs)]
    idx = torch.randint(len(d["tid"]), (min(BATCH_PERTS, len(d["tid"])),), device=device)
    code = ctx_feat[d["name"]] + CTX_NOISE * torch.randn(CTX_DIM, device=device)
    pred = model(code, d["tid"][idx])
    err = (pred - d["y"][idx]) ** 2 * d["w"][None, :]
    loss = err.sum() / (d["w"].sum() * len(idx))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    lv = loss.item()
    if not math.isfinite(lv):
        raise SystemExit("FAIL: loss is not finite")
    smooth = lv if smooth is None else 0.98 * smooth + 0.02 * lv
    step += 1
    train_time = time.time() - t_train
    if step % 200 == 0:
        print(f"step {step:6d} | loss {smooth:.4f} | lr x{lr_mult:.2f} | {train_time:.0f}s", end="\r")
    if train_time >= TIME_BUDGET:
        break
print()

# ---------------------------------------------------------------------------
# Prediction + evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict(task):
    model.eval()
    tid = torch.tensor([target_id.get(t, -1) for t in task.targets], device=device)
    lfc = model(ctx_feat[task.context.name], tid).float().cpu().numpy()
    for i, t in enumerate(task.targets):          # the knockdown itself
        if t in gene_index:
            lfc[i, gene_index[t]] = KNOCKDOWN_LFC
    return lfc


val_preds = {t.context.name: predict(t) for t in val_tasks}
save_predictions(val_preds, "val")
if submit_tasks:
    save_predictions({t.context.name: predict(t) for t in submit_tasks}, "submit")
report = evaluate_predictions(val_preds)

print("---")
print(f"val_score:        {report['val_score']:.6f}")
for m in METRICS:
    print(f"{m + ':':<18}{np.nanmean([report[c]['scaled'][m] for c in val_preds]):.6f}")
print(f"training_seconds: {train_time:.1f}")
print(f"total_seconds:    {time.time() - t_start:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_M:     {num_params / 1e6:.2f}")
