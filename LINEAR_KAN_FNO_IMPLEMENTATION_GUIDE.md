# Linear KAN-FNO: Implementation Guide for Symbolic Recovery Study

> **Purpose**: This document is a complete implementation blueprint for the linear-case
> training and symbolic recovery study of the LS-KAN-FNO project. It covers repository
> structure, data generation, training pipeline, symbolic recovery procedure, and
> Colab/Kaggle integration. All existing code (linear FFT solver, linear LS-FNO with MLP,
> linear KAN-FNO) is already implemented and should be integrated into this structure.

---

## 1. Context and Goal

### 1.1 What already exists

The following components are already implemented and must be incorporated:

- `fft_linear.py` — Moulinec-Suquet basic scheme for linear elasticity
- `ls_fno_mlp.py` — LS-FNO with Yarotsky MLP double-contraction operator `τ_θ`
- `ls_fno_kan.py` — LS-FNO with KAN double-contraction operator `τ_θ`

### 1.2 What this study produces

Three deliverables for the linear case:

1. **A trained KAN-FNO** whose edge functions, after symbolic fitting, recover the
   bilinear double-contraction `T:ε` to 3–4 decimal places — confirming the training
   pipeline, the symbolic recovery procedure, and KAN's interpretability advantage
   over MLP on a problem where ground truth is analytically known.

2. **A comparison study** between the Yarotsky MLP baseline and the trained KAN on:
   accuracy vs. parameter count, iteration count to convergence, and behavior at
   small strains and high material contrast.

3. **A symbolic recovery procedure** (reusable for the nonlinear case) that prunes,
   fits, and symbolically simplifies KAN edge functions into closed-form expressions.

### 1.3 Role of the linear case in the paper

The linear case is the **validation harness**, not a novelty contribution. The payoff is:
*"The trained KAN recovers the known bilinear physics exactly — so we can trust it when
it recovers unknown nonlinear physics in the next section."*  
Treat it as a controlled falsification test, paralleling KANO's synthetic operator
benchmarks before the quantum Hamiltonian results.

---

## 2. Repository Structure

The code must be organized as an **installable Python package** hosted on GitHub.
This enables clean Colab/Kaggle usage via `git clone` + `pip install -e .` without
re-uploading files between sessions.

```
ls_kan_fno/                              # repository root
│
├── ls_kan_fno/                          # the installable package
│   │
│   ├── __init__.py
│   │
│   ├── fft/                             # FFT-based solvers (existing code)
│   │   ├── __init__.py
│   │   ├── linear.py                    # Moulinec-Suquet basic scheme
│   │   └── green.py                     # Eshelby-Green operator Γ̂(ξ)
│   │
│   ├── models/                          # network architectures (existing code)
│   │   ├── __init__.py
│   │   ├── ls_fno_mlp.py                # Yarotsky MLP baseline
│   │   └── ls_fno_kan.py                # KAN replacement
│   │
│   ├── data/                            # data generation and loading (NEW)
│   │   ├── __init__.py
│   │   ├── microstructure.py            # microstructure generators
│   │   ├── generate.py                  # run FFT solver over a sample grid
│   │   └── dataset.py                   # PyTorch Dataset + DataLoader factories
│   │
│   ├── training/                        # training infrastructure (NEW)
│   │   ├── __init__.py
│   │   ├── trainer.py                   # Trainer class
│   │   ├── losses.py                    # loss functions
│   │   └── metrics.py                   # evaluation metrics
│   │
│   ├── symbolic/                        # symbolic recovery utilities (NEW)
│   │   ├── __init__.py
│   │   ├── edge_inspect.py              # extract and normalize edge functions
│   │   ├── prune.py                     # prune low-contribution edges
│   │   └── regression.py               # fit edges to elementary functions
│   │
│   └── utils/                           # shared utilities (NEW)
│       ├── __init__.py
│       ├── io.py                        # HDF5 read/write helpers
│       ├── seed.py                      # deterministic seeding
│       └── plotting.py                  # standard visualization helpers
│
├── scripts/                             # CLI entry points
│   ├── generate_data.py                 # generate and save HDF5 dataset
│   ├── train.py                         # train from a YAML config
│   └── recover_symbolic.py             # run symbolic recovery on a checkpoint
│
├── configs/                             # one YAML per experiment
│   ├── linear_prototype.yaml            # small run for smoke-testing
│   └── linear_full.yaml                 # production run
│
├── notebooks/                           # thin Colab/Kaggle wrappers
│   ├── 00_smoke_test.ipynb
│   ├── 01_generate_data.ipynb
│   ├── 02_train_linear.ipynb
│   └── 03_symbolic_recovery.ipynb
│
├── tests/
│   └── test_fft_solver.py
│
├── pyproject.toml                       # package metadata + dependencies
├── requirements.txt                     # pinned for reproducibility
└── README.md
```

### 2.1 `pyproject.toml` minimum content

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.backends.legacy:BuildBackend"

[project]
name = "ls_kan_fno"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "scipy>=1.12",
    "h5py>=3.10",
    "pyyaml>=6.0",
    "pykan>=0.2",          # KAN implementation
    "wandb>=0.16",         # experiment tracking
    "matplotlib>=3.8",
    "tqdm>=4.66",
]
```

---

## 3. Data Generation (Linear Case)

### 3.1 What a sample contains

Each training sample is a pair `(input, target)`:

| Field | Shape | Description |
|---|---|---|
| `C_field` | `(N, N, 3, 3)` | Local stiffness tensor field (isotropic per phase) |
| `eps_bar` | `(3, 3)` | Prescribed macroscopic strain |
| `eps_star` | `(N, N, 3, 3)` | Converged strain field from FFT solver |
| `tau_star` | `(N, N, 3, 3)` | Converged polarization stress `τ* = (C - C⁰):ε*` |
| `T_field` | `(N, N, 3, 3)` | Normalized contrast field `T = (C - C⁰)/α₀` |
| `n_iter` | scalar int | Iterations to convergence (logged but not used as loss) |

`T_field` is derived from `C_field` and should be precomputed and stored — it is the
direct input to `τ_θ(T, ε)` and avoids recomputing `α₀` at every training step.

### 3.2 Microstructure generation (`data/microstructure.py`)

Three geometric families, all 2D, two-phase (matrix + inclusion):

**Family A — Random circles**  
- Circular inclusions with radii sampled uniformly in `[r_min, r_max] = [0.03, 0.15]`
  (fraction of unit cell side)
- Centers sampled uniformly, non-overlapping (rejection sampling)
- Volume fractions in `[0.1, 0.5]`
- Use this family for training

**Family B — Random ellipses**  
- Ellipses with aspect ratios sampled uniformly in `[1.0, 3.0]`, random orientation
- Same area-fraction range as circles
- Use this family for training (combined with A)

**Family C — Voronoi polycrystal**  
- Voronoi tessellation with 20–50 seeds
- Alternate phases by grain index (checkerboard-like)
- Volume fractions in `[0.4, 0.6]`
- **Use exclusively for out-of-distribution testing — never for training or validation**

The inclusion phase has isotropic stiffness:
```
C_inclusion = 2μ_inc * I_sym + λ_inc * (I ⊗ I)
C_matrix    = 2μ_mat * I_sym + λ_mat * (I ⊗ I)
```
where `I_sym` is the 4th-order symmetric identity and contrast is
`κ = μ_inclusion / μ_matrix`.

### 3.3 Parameter sampling ranges

| Parameter | Range | Sampling |
|---|---|---|
| Contrast `κ = μ_inc / μ_mat` | `[10, 100]` | log-uniform |
| Macroscopic strain magnitude `‖ε̄‖` | `[10⁻⁴, 10⁻¹]` | log-uniform |
| Load direction in `Sym(2)` | full sphere | uniform on unit sphere |
| Poisson's ratio (both phases) | `[0.2, 0.4]` | uniform |

**Critical**: include strains at the lower end (`‖ε̄‖ = 10⁻⁴, 10⁻³`) — this is
exactly where the Yarotsky MLP is worst (up to 10% error at 0.1% strain) and where
the KAN's adaptive grid should show the clearest improvement.

### 3.4 Dataset splits and sizes

| Split | Families | Size (prototype) | Size (full) |
|---|---|---|---|
| Train | A + B (random 80%) | 1,200 | 8,000 |
| Validation | A + B (random 20%) | 300 | 2,000 |
| Test — in-distribution | A + B (fresh) | 250 | 1,000 |
| Test — out-of-distribution | C (Voronoi only) | 250 | 1,000 |

Start with the prototype sizes. Scale only after the full pipeline runs end-to-end.

### 3.5 HDF5 storage format (`data/generate.py`)

Use a **single HDF5 file per split**. The schema is:

```
train.h5
├── C_field      (N_samples, N, N, 3, 3)   float32
├── eps_bar      (N_samples, 3, 3)          float32
├── eps_star     (N_samples, N, N, 3, 3)   float32
├── tau_star     (N_samples, N, N, 3, 3)   float32
├── T_field      (N_samples, N, N, 3, 3)   float32
├── n_iter       (N_samples,)               int32
└── attrs:
      resolution: 64
      n_samples: 1200
      contrast_range: [10, 100]
      strain_range: [1e-4, 1e-1]
      families: ['circles', 'ellipses']
      created: <timestamp>
      git_hash: <commit hash>
```

Use `compression='gzip', compression_opts=4` for storage. At 64×64 resolution,
one 1,200-sample train file is approximately 350 MB.

**Storage rule**: tag every dataset file with the git commit hash used to generate it.
If you change the generation logic, you must regenerate the data, not reuse old files.

### 3.6 Resolution

Use **64×64** throughout the linear study. Moving to 128×128 brings no additional
insight for the symbolic recovery goal (the target function `T:ε` is pointwise and
resolution-independent) and costs 4× memory per sample.

---

## 4. PyTorch Dataset (`data/dataset.py`)

The `Dataset` class reads from HDF5 with lazy loading (sample-at-a-time, not full
array into RAM). Key design choices:

- `__getitem__` returns a flat dict with all fields as `torch.Tensor`
- Stiffness tensors are flattened to `(9,)` vectors before being fed to the model
  (the KAN acts pointwise on flattened tensor pairs)
- `DataLoader` should use `num_workers=2` and `pin_memory=True` on GPU sessions
- Expose a `DataLoaderFactory` that builds train/val/test loaders from a config dict

```python
# expected usage
from ls_kan_fno.data.dataset import DataLoaderFactory

loaders = DataLoaderFactory.from_config(config['data'])
train_loader = loaders['train']
val_loader   = loaders['val']
test_loaders = loaders['test']   # dict: {'in_dist': ..., 'ood': ...}
```

---

## 5. Training Pipeline

### 5.1 Loss functions (`training/losses.py`)

Use a **combined loss** with two terms:

```
L_total = λ_field * L_field + λ_eff * L_eff
```

**Field loss** — relative L2 error on the strain field:
```
L_field = ‖ε_pred - ε_star‖² / ‖ε_star‖²
```
averaged over the spatial grid and the batch.

**Effective stiffness loss** — relative error on the homogenized modulus:
```
C_eff_pred = mean_x(σ_pred) / ε̄        (scalar for uniaxial loading)
L_eff = |C_eff_pred - C_eff_ref| / |C_eff_ref|
```
where `C_eff_ref` is computed from `tau_star` stored in the dataset.
`L_eff` is a scalar per sample and gives a cleaner training signal early in training.

Default weights: `λ_field = 1.0`, `λ_eff = 0.1`. Log both terms separately in W&B.

### 5.2 Optimizer and scheduler

Two-phase training:

**Phase 1 — Adam with cosine annealing (main training)**
```
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=N_epochs, eta_min=1e-5)
```
Run for 150–200 epochs with batch size 16.

**Phase 2 — LBFGS pass (edge refinement)**
After Adam converges, run 20–50 LBFGS steps on small batches (4–8 samples).
LBFGS is recommended by pykan for refining B-spline coefficients after gradient
descent because it optimizes the piecewise-smooth landscape better than Adam.
Only run this phase once Adam's validation loss has plateaued.

### 5.3 Memory: gradient checkpointing

The LS-FNO unrolls `K` Fourier layers sequentially. For `K = 20` and batch size 16
at 64×64, naively storing all intermediate activations for backprop will exceed
16 GB of VRAM. **Mandatory**: wrap each Fourier layer forward call in
`torch.utils.checkpoint.checkpoint(layer, *inputs)`. This recomputes activations
during the backward pass, trading ~30% more FLOPs for ~80% less VRAM.

### 5.4 Mixed precision

Enable `torch.amp.autocast(device_type='cuda')` and `GradScaler` from the start.
On V100/A100 this roughly doubles throughput with negligible accuracy impact for
this class of model.

### 5.5 Trainer class (`training/trainer.py`)

The `Trainer` takes a config dict and exposes a single `.fit()` method:

```python
# expected usage
from ls_kan_fno.training.trainer import Trainer

trainer = Trainer(config)
trainer.fit()
```

Internally `Trainer` handles:
- Data loading via `DataLoaderFactory`
- Model construction and device placement
- Optimizer + scheduler setup
- Training loop with validation at end of each epoch
- Gradient checkpointing (always on)
- Mixed precision (always on if CUDA available)
- Checkpoint saving every 10 epochs to `config['output_dir']`
- W&B logging: train loss, val loss, L_field, L_eff, learning rate, GPU memory

### 5.6 Checkpointing

Save checkpoints to Google Drive (or `/kaggle/working/` on Kaggle). Each checkpoint
is a dict:

```python
{
    'epoch': int,
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scheduler_state_dict': ...,
    'val_loss': float,
    'config': dict,
    'git_hash': str,
}
```

Keep the **best checkpoint** (lowest val loss) plus the **last checkpoint**.
Delete all intermediate checkpoints after training to save Drive space.

### 5.7 YAML config structure

```yaml
# configs/linear_full.yaml

data:
  train_path: /content/drive/MyDrive/ls_kan_fno/data/train.h5
  val_path:   /content/drive/MyDrive/ls_kan_fno/data/val.h5
  test_paths:
    in_dist: /content/drive/MyDrive/ls_kan_fno/data/test_in_dist.h5
    ood:     /content/drive/MyDrive/ls_kan_fno/data/test_ood.h5
  batch_size: 16
  num_workers: 2

model:
  type: ls_fno_kan        # or ls_fno_mlp for the baseline
  n_layers: 20            # K = number of LS iterations
  kan_grid_size: 50       # number of B-spline knots per edge
  kan_spline_order: 3
  input_dim: 18           # flattened (T, ε) ∈ R^9 × R^9
  output_dim: 9           # flattened τ ∈ R^9

training:
  n_epochs: 200
  lr: 1.0e-3
  lr_min: 1.0e-5
  weight_decay: 1.0e-4
  lambda_field: 1.0
  lambda_eff: 0.1
  lbfgs_phase: true
  lbfgs_steps: 30
  lbfgs_batch_size: 4

output:
  dir: /content/drive/MyDrive/ls_kan_fno/runs/linear_full/
  checkpoint_every: 10

wandb:
  project: ls_kan_fno
  run_name: linear_full_kan
  entity: <your_wandb_entity>
```

---

## 6. Evaluation Metrics (`training/metrics.py`)

Compute and log the following on the test set after training:

| Metric | Description |
|---|---|
| `rel_L2_field` | Relative L2 error on `ε*` field, averaged over test set |
| `rel_err_C_eff` | Relative error on effective stiffness tensor (per component) |
| `n_iter_mean` | Mean LS iterations to convergence over test set |
| `n_iter_max` | Worst-case iterations (stress-test for contractivity) |
| `gamma_theta` | Empirical Lipschitz constant of `τ_θ` on test pairs (contractivity check) |

**Contractivity check** is non-optional. Compute `γ_θ` empirically by sampling
random pairs `(T, ε)` and `(T, ε')` and computing:
```
γ_θ = max_{pairs} ‖τ_θ(T, ε) - τ_θ(T, ε')‖ / ‖ε - ε'‖
```
If `γ_θ ≥ 1`, the trained model is not a contraction and iteration counts will blow up.
This check must be done before any iteration-count experiment.

The contractivity requirement from theory is:
```
γ_θ < 2 / (κ + 1)
```
For `κ = 96`: `γ_θ < 0.020`. Log this bound alongside the empirical `γ_θ`.

**Contrast sweep** (key experiment for paper): evaluate `rel_err_C_eff` and
`n_iter_mean` across `κ ∈ {12, 24, 48, 96}` and tabulate side-by-side with the
Yarotsky MLP baselines from the FNO Micromechanics paper (Table 3 and Table 5 in
that paper). This is the direct replication + comparison.

**Strain sweep** (key experiment for paper): evaluate `rel_err_C_eff` across
`‖ε̄‖ ∈ {0.1%, 1%, 10%, 50%}` for both KAN and MLP. Expected finding: KAN
significantly outperforms MLP at small strains due to adaptive B-spline grid.

---

## 7. Symbolic Recovery Procedure (`symbolic/`)

### 7.1 Overview

The symbolic recovery pipeline takes a trained KAN checkpoint and produces:
(1) a pruned, sparse KAN; (2) per-edge symbolic fits; (3) an assembled closed-form
expression; (4) a verification that the symbolic expression matches the trained
network on held-out data.

### 7.2 Step 1 — Edge normalization (`edge_inspect.py`)

Before fitting, normalize each B-spline edge to remove gauge freedom. Gauge freedom
means a scale factor can be moved between adjacent edges without changing the function,
giving different edge shapes for the same function.

**Normalization**: scale each edge output to have unit L2 norm over the training
distribution. Store the scale factors separately; they are needed when assembling
the final formula.

```python
# conceptual API
from ls_kan_fno.symbolic.edge_inspect import extract_edges

edges = extract_edges(
    model,                       # trained KAN-FNO
    loader=train_loader,         # for computing normalization statistics
    normalize=True,
)
# edges: list of (input_points, output_values, metadata) per edge
```

### 7.3 Step 2 — Pruning (`prune.py`)

Many edges will have near-zero contribution to the output. Prune edges whose
**attribution score** (gradient-based saliency averaged over training data) falls
below a threshold `τ_prune = 0.01`. After pruning, verify that the pruned network's
predictions deviate from the full network's predictions by less than 1% on the
validation set. If the deviation exceeds 1%, raise `τ_prune` to a lower threshold
and retry.

```python
from ls_kan_fno.symbolic.prune import prune_kan

pruned_model = prune_kan(model, train_loader, threshold=0.01, max_deviation=0.01)
```

### 7.4 Step 3 — Per-edge symbolic fitting (`regression.py`)

For each surviving edge, sample `(input, output)` pairs and fit to a library of
candidate elementary functions using scipy `curve_fit` with multiple restarts.

**Candidate function library** for the linear case:

| Symbol | Function |
|---|---|
| `linear` | `a·x + b` |
| `quadratic` | `a·x² + b·x + c` |
| `abs_linear` | `a·|x| + b` |
| `sqrt` | `a·√(|x|) + b` |
| `const` | `c` (zero edge) |

For the linear case, all surviving edges should fit `linear` with R² ≥ 0.999.
If they don't, the training has not converged correctly or the pruning was too
aggressive.

**Goodness of fit** is reported as R² and maximum absolute residual. An edge
is considered symbolically recovered if R² ≥ 0.999.

```python
from ls_kan_fno.symbolic.regression import fit_edges_to_library

fit_results = fit_edges_to_library(
    pruned_model,
    candidate_library=['linear', 'quadratic', 'abs_linear', 'sqrt', 'const'],
    n_points=1000,
    input_range=(-1.0, 1.0),
)
# fit_results: list of {edge_id, best_symbol, params, R2, max_residual}
```

### 7.5 Step 4 — Symbolic assembly

Once all surviving edges have symbolic fits, compose them layer-by-layer to
produce a single closed-form expression for `τ_θ(T, ε)`. For the linear case,
the assembled expression should reduce to:

```
τ_θ(T, ε) ≈ T : ε     (bilinear double contraction)
```

with recovered coefficients matching the analytical result to 3–4 decimal places.

Print the assembled symbolic expression and also compute the discrepancy:

```
δ_symbolic = ‖τ_symbolic(T, ε) - τ_θ(T, ε)‖ / ‖τ_θ(T, ε)‖
```

averaged over the test set. This should be < 1% for a clean recovery.

### 7.6 Step 5 — Verification

Run the full LS iteration replacing `τ_θ` with the symbolic expression and verify
that effective stiffness predictions remain within the original model's accuracy
band. This confirms the symbolic formula is operationally equivalent.

### 7.7 pykan integration notes

The `pykan` library (`pip install pykan`) provides:
- `KAN.auto_symbolic(lib=[...])` — automatic symbolic regression per edge
- `KAN.prune()` — remove low-attribution edges
- `KAN.plot()` — visualize the edge functions

Use pykan's built-in methods where possible and only write custom code for:
- gauge normalization (not handled by pykan)
- the custom candidate library tailored to this physics domain
- the LS-iteration-level symbolic verification (Step 5)

**Compatibility note**: pykan is under active development. Pin the version in
`pyproject.toml` and test with the pinned version.

---

## 8. Colab/Kaggle Integration

### 8.1 Standard Colab notebook header

Every Colab notebook starts with these cells, in order:

```python
# Cell 1 — Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — Clone / pull repo
import os
REPO_DIR = '/content/ls_kan_fno'
if not os.path.exists(REPO_DIR):
    !git clone https://github.com/<your_username>/ls_kan_fno.git {REPO_DIR}
else:
    !cd {REPO_DIR} && git pull

# Cell 3 — Install package
!pip install -e {REPO_DIR} -q

# Cell 4 — Verify GPU
import torch
print(torch.cuda.get_device_name(0))
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 5 — W&B login
import wandb
wandb.login()

# Cell 6 — Set deterministic seed
from ls_kan_fno.utils.seed import set_seed
set_seed(42)
```

### 8.2 Kaggle equivalent

```python
# Cell 1 — Install from attached code dataset
!pip install /kaggle/input/ls-kan-fno-code/ -q

# Cell 2 — Set up paths
DATA_DIR  = '/kaggle/input/ls-kan-fno-data/'
OUT_DIR   = '/kaggle/working/runs/'
import os; os.makedirs(OUT_DIR, exist_ok=True)

# Cell 3 — GPU check and seed (same as Colab)
```

For Kaggle: upload the package as a **Code dataset** (zip of the repo) and the
HDF5 files as a **Data dataset**. Attach both to every notebook.

### 8.3 Session management

- **Colab disconnects without warning.** Checkpoint every 10 epochs to Drive.
  Set a cell to reload from the latest checkpoint at the start of every training session.

- **Never run data generation in Colab.** Generate all HDF5 files locally on your
  laptop. Generating in Colab wastes GPU time and the data is lost on disconnect.

- **Free Colab T4** (16 GB VRAM) is sufficient for 64×64 with batch size 16 and
  gradient checkpointing. Colab Pro A100 (40 GB VRAM) removes memory constraints
  and allows batch size 64 — significantly faster.

- **Kaggle weekly GPU quota** is 30 hours. Use it for long training runs. Use Colab
  for debugging and short iteration experiments.

### 8.4 Data path strategy

Store all data in a fixed Drive folder structure:
```
MyDrive/
└── ls_kan_fno/
    ├── data/
    │   ├── train.h5
    │   ├── val.h5
    │   ├── test_in_dist.h5
    │   └── test_ood.h5
    └── runs/
        ├── linear_prototype/
        │   ├── best_checkpoint.pt
        │   └── last_checkpoint.pt
        └── linear_full/
            ├── best_checkpoint.pt
            └── last_checkpoint.pt
```

Config files reference absolute paths using `/content/drive/MyDrive/ls_kan_fno/`
as the prefix.

---

## 9. Experiment Tracking

Use **Weights & Biases** (free academic tier at wandb.ai).

Log per epoch:
- `train/loss_total`, `train/loss_field`, `train/loss_eff`
- `val/loss_total`, `val/loss_field`, `val/loss_eff`
- `val/rel_L2_field`, `val/rel_err_C11`
- `train/lr` (current learning rate)
- `system/gpu_memory_GB`

Log at end of training:
- `test/rel_L2_field_in_dist`, `test/rel_L2_field_ood`
- `test/n_iter_mean`, `test/n_iter_max`
- `test/gamma_theta`
- Full contrast sweep table (κ = 12, 24, 48, 96)
- Full strain sweep table (‖ε̄‖ = 0.1%, 1%, 10%, 50%)
- Edge visualization plots (uploaded as W&B media)

Tag every run with its config file name and git commit hash.

---

## 10. Success Criteria (Linear Case)

The linear training study is considered complete when all of the following hold:

| Criterion | Target | Notes |
|---|---|---|
| `rel_err_C11` at κ=24 | < 0.5% | Match FNO11 paper baseline |
| `rel_err_C11` at κ=96 | < 1.0% | Must not blow up like FNO7 |
| `n_iter` at κ=96 | < 750 | Within 5% of FFT (716) |
| `gamma_theta` | < 0.02 | Contractivity at κ=96 |
| `rel_err_C11` at ε̄=0.1% | < 1.0% | Key improvement over FNO7 (10.13%) |
| KAN vs MLP parameter count | KAN < MLP at same accuracy | Compactness claim |
| Symbolic R² (all edges) | ≥ 0.999 | Clean bilinear recovery |
| δ_symbolic | < 1.0% | Symbolic formula is operationally valid |
| OOD (Voronoi) `rel_err_C11` | < 2.0% | Generalization check |

If any criterion fails, diagnose in this order:
1. Contractivity — check `gamma_theta` first. If ≥ 1, training diverged.
2. Small-strain failure — check if B-spline grid is correctly adapted.
3. OOD failure — check if training microstructure distribution is diverse enough.
4. Symbolic R² failure — check normalization and pruning threshold.

---

## 11. Sequence of Implementation Steps

Follow this exact order to avoid wasted effort:

1. **Restructure existing code** into the package layout (Section 2). Run `pytest tests/`
   to confirm the FFT solver still gives the same results.

2. **Implement `data/microstructure.py`** — circles and ellipses generators only.
   Verify visually that generated microstructures look correct.

3. **Implement `data/generate.py`** — call FFT solver for each sample, save to HDF5.
   Generate the prototype dataset (1,200 / 300 / 250 / 250 samples).

4. **Implement `data/dataset.py`** — PyTorch Dataset + DataLoader factory. Verify
   a batch has the correct shapes and dtypes.

5. **Implement `training/losses.py`** and **`training/metrics.py`**.

6. **Implement `training/trainer.py`** with gradient checkpointing and mixed precision.
   Run the smoke test: 3 epochs on prototype data, confirm loss decreases.

7. **Run `notebooks/00_smoke_test.ipynb`** on Colab to verify the full
   Drive ↔ Colab ↔ GitHub loop.

8. **Run `notebooks/02_train_linear.ipynb`** with `configs/linear_prototype.yaml`
   (50 epochs, prototype data). Confirm W&B logging is working.

9. **Generate the full dataset** locally and upload to Drive.

10. **Run `configs/linear_full.yaml`** (200 epochs, full data).

11. **Implement `symbolic/`** and run `notebooks/03_symbolic_recovery.ipynb` on the
    best checkpoint.

12. **Evaluate and tabulate** all success criteria (Section 10).

---

## 12. Key References for Implementation

- **Moulinec-Suquet (1994)**: Basic scheme, Eshelby-Green operator definition
- **Nguyen & Schneider (2025)** (FNO Micromechanics paper): Table 3, Table 5, Table 6,
  Figure 6 — the direct baselines this study replicates and extends
- **pykan documentation**: `auto_symbolic()`, `prune()`, B-spline grid management
- **KANO paper (ICLR 2026)**: Table 4 — template for symbolic recovery reporting format
- **Schneider (2020)** (porous FFT): needed for Direction 3 but not Direction 1
