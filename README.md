# QEP paper figures

Code for three Quantum Equilibrium Propagation (QEP) studies built on the
[QEP method](https://www.nature.com/articles/s41467-025-61665-6) (two-phase
adiabatic evolution for computing gradients of quantum systems).

**This package intentionally includes no data and no rendered
images** — only the code and notebooks that produce the figures. To get a
figure, first run the data-generation script(s) listed below for that
project (they write into a local `data/` directory next to the notebook),
then run the notebook, which writes its own PDFs into a local `figs/` or
`figures/` directory.

## Dependencies

```bash
pip install jax dynamiqs qutip numpy matplotlib diffrax optax scipy
```

## Layout

```
QEP/                          core QEP package (model, gradient, training loop)
AFM_searching/                antiferromagnetic order-parameter search, chains + 2D lattices
QEP_multiple_shot/AFM_searching/   finite-shot-noise version of the same search
H_detecting/                  detecting a probe Hamiltonian via QEP-trained response
```

### `AFM_searching/`

- `mod_func.ipynb` — standalone (no local imports); reproduces `params_func.pdf`.
- `code/afmqep/` — the model/gradient/training package this study's notebooks
  and scripts import.
- `code/train_run.py`, `code/phase_diagram.py` — regenerate the training-run
  and phase-diagram `.npz` files under `paper_figs/data/{train,phase}/`
  (used by `figs_N5.ipynb`/`figs_L3x3.ipynb`). See each script's `--help`
  for the exact CLI used per run (geometry, `T`, order parameter, learning
  rate); `paper_figs/README.md` documents which invocations produced which
  files.
- `paper_figs/code/sample_benchmark_windows.py` — regenerates the
  `bias_benchmarks_deep/*.npz` gradient-decay-benchmark data used by
  `grad_figs.ipynb` (run per chain length `N`, `--Omega 1.0 --Delta 1.0`).
- `paper_figs/grad_figs.ipynb`, `paper_figs/figs_N5.ipynb`,
  `paper_figs/figs_L3x3.ipynb` — the figure notebooks; see
  `paper_figs/README.md` for full provenance of every figure/data file.
- `theory_check/n1_gradient_limits.md` — note cited directly by `grad_figs.ipynb`.

### `QEP_multiple_shot/AFM_searching/`

- `figs_N5.ipynb`, `figs_N9.ipynb` — the figure notebooks. They locate their
  own project root by checking for `shot_study/config.py`, so the directory
  structure here must stay intact.
- `afm_shots/` — model/gradient/training package the `shot_study/*.py`
  generation scripts import (via `afm_shots/_pathfix.py`, which also
  requires the top-level `QEP/` package included in this repo, one level up
  from `AFM_searching/`).
- `shot_study/run.py`, `beta_shot_scan.py`, `overlap_training.py`,
  `summarize_overlap.py`, `trajectory_expectations.py`,
  `figure11b_resampling.py`, `green_start_budget.py`,
  `green_trajectory_expectations.py`, `purple_start_budget.py`,
  `verify_budget_noise.py`, `analyze_budget_noise_check.py` — regenerate,
  respectively, the gradient/trajectory/budget/audit `.npz`/`.json` files
  both notebooks load from `data/shots_2026-09-11/`. `shot_study/README.md`
  documents the pipeline; `submit_*.sbatch` record the exact cluster
  invocations used.
- **Cross-project dependency:** both notebooks also load
  `AFM_searching/data/phase/phase_chain_N5.npz` and `phase_chain_N9.npz`
  from the sibling top-level `AFM_searching/` project (as the background of
  every `phase_trajectories_*` figure) — regenerate those first via
  `AFM_searching/code/phase_diagram.py`.
- `reseach_plan_2026-09-11.md`, `reports/2026-09-11/main.tex` — the research
  plan and paper draft these figures were made for.

### `H_detecting/`

- `paper_figs.ipynb` — the figure notebook; self-contained aside from
  `code/qep.py`.
- `code/qep.py` — the model/gradient/training module `paper_figs.ipynb` imports.
- `code/run_boundary.py` — regenerates the `bd_*.npz` files
  (`PYTHONPATH=. python run_boundary.py all`, writes to `../data/`).
- `code/train_experiments.py` — alternative/independent regenerator for
  `train_multi_init_paths_TS.npz` / `train_probe_sets_TS.npz`; the notebook
  can also regenerate these itself in-cell (flip `REGEN`/`REGEN_PS` to
  `True`) if `code/qep.py`'s `dynamiqs`-based training is available.
- `README.md`, `intro.md`, `log/STATUS.md` — code/physics documentation and
  a detailed research log explaining the provenance of every figure.
