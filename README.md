# RL-Brand-Bound CO

PPO+GAE branching policy for Branch-and-Bound on MILP instances, pretrained via
imitation learning on Reliability Branching demonstrations. Built on Ecole +
PySCIPOpt with a bipartite GCNN backbone.

Course project for AIE 635 (Reinforcement Learning).

---

## Project Overview

- **Primary claim:** A PPO+GAE branching policy, pretrained via behavioral
  cloning on Reliability Branching demonstrations, matches or approaches FSB on
  set covering and combinatorial auction instances, and transfers to
  1.5× / 2× larger instances.
- **Baselines:** Random branching and FSB (Full Strong Branching).
- **Metrics:** wall-clock solve time, B&B node count, dual integral.

---

## Setup

> **Important:** Ecole does not ship official Windows binaries. Windows users
> should run the project inside **WSL2 + Ubuntu 22.04**. Your existing
> Windows-side Anaconda installation is *not* affected — WSL2 uses an isolated
> Linux filesystem and we install a separate Miniconda inside it.

### Windows users (WSL2 + Ubuntu)

1. **Enable WSL2 and install Ubuntu** (PowerShell as Administrator):
   ```powershell
   wsl --install -d Ubuntu-22.04
   ```
   Reboot if prompted. Launch "Ubuntu" from the Start menu and create a Linux
   user.

2. **Install Miniconda inside Ubuntu** (Ubuntu shell):
   ```bash
   curl -L -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
   $HOME/miniconda3/bin/conda init bash
   exec bash
   ```

3. **Clone the repository** (Ubuntu shell, inside your Linux home dir):
   ```bash
   git clone https://github.com/Mustafa-Er/RL_Brand_Bound_CO.git
   cd RL_Brand_Bound_CO
   ```

4. **Create the environment**:
   ```bash
   conda env create -f environment.yml
   conda activate rl_bb
   ```

5. **GPU (optional).** Check whether CUDA is visible from WSL:
   ```bash
   nvidia-smi
   ```
   If yes, replace the `pytorch::cpuonly` line in `environment.yml` with the
   appropriate `pytorch-cuda=X.Y` channel directive before creating the env.
   See [PyTorch install matrix](https://pytorch.org/get-started/locally/).

6. **Smoke test**:
   ```bash
   python -c "import ecole, pyscipopt, torch, torch_geometric; print('ok')"
   ```

### Linux / macOS users

Skip the WSL step:
```bash
git clone https://github.com/Mustafa-Er/RL_Brand_Bound_CO.git
cd RL_Brand_Bound_CO
conda env create -f environment.yml
conda activate rl_bb
python -c "import ecole, pyscipopt, torch, torch_geometric; print('ok')"
```

---

## Directory Structure

```
RL_Brand_Bound_CO/
├── config/             # YAML configs (base.yaml + experiment overrides)
├── src/rl_bb/          # Library code
│   ├── instances/      # Instance generators (Stage 1)
│   ├── envs/           # Ecole env wrappers, DFS node selection (Stage 2)
│   ├── experts/        # RB expert (Stage 3) + FSB/Random baselines (Stage 7)
│   ├── models/         # GCNN policy + value heads (Stage 4)
│   ├── training/       # pretrain.py (Stage 5), ppo.py (Stage 6)
│   ├── eval/           # Evaluation pipeline (Stage 7)
│   └── utils/          # Logging, seeding, paths
├── scripts/            # CLI entry points
├── tests/              # Unit tests
├── data/               # Generated instances (git-ignored)
└── logs/               # Training / eval logs (git-ignored)
```

---

## How to Run

Install the package in editable mode once (so `scripts/` and `rl_bb` resolve):

```bash
pip install -e .
```

### Stage 1 — Generate instances

Dummy mode (small instances, end-to-end verification, ~1 min):

```bash
python -m scripts.generate_instances \
    --config config/base.yaml --config config/dummy.yaml
```

Full mode (literature-size, much longer):

```bash
python -m scripts.generate_instances --config config/base.yaml
```

Limit to one problem type with `--problem set_covering` or
`--problem combinatorial_auction`. Output goes to
`data/<experiment.name>/<problem>/<regime>/<split>/instance_XXXX.mps`.

Run unit tests:

```bash
pytest tests/
```

### Stage 2 — Random-policy environment smoke test

Roll out a uniform-random branching policy through a few dummy instances to
verify the DFS-forced Ecole env, the bipartite observation, and the
dual-bound-gain reward all wire up correctly:

```bash
python -m scripts.run_env_smoke \
    --config config/base.yaml --config config/dummy.yaml \
    --problem set_covering --n-instances 3
```

Logs land in `logs/env_smoke.log`.

### Stage 3 — Collect Reliability Branching demonstrations

Reliability Branching is the sole expert for imitation pretraining. Roll
it out on both the train and val splits so Stage 5 has data for both:

```bash
python -m scripts.collect_demonstrations \
    --config config/base.yaml --config config/dummy.yaml \
    --problem combinatorial_auction --split train

python -m scripts.collect_demonstrations \
    --config config/base.yaml --config config/dummy.yaml \
    --problem combinatorial_auction --split val
```

Output goes to
`data/<experiment.name>/demonstrations/<problem>/<regime>/<split>/rb/`.

FSB and Random are kept as evaluation baselines (Stage 7); they are not
collected as demonstrations.

### Stage 4 — GCNN forward-pass check

Verify the bipartite GCNN constructs, runs forward, and backpropagates on a
real demonstration sample (any non-empty pickle from Stage 3 works):

```bash
python -m scripts.check_model \
    --demo data/rl_bb_dummy/demonstrations/combinatorial_auction/train_size/train/rb/instance_0018.pkl
```

### Stage 5 — Imitation pretraining (behavioral cloning)

After collecting RB demonstrations for the chosen problem on both train and
val splits, run:

```bash
python -m scripts.pretrain \
    --config config/base.yaml --config config/dummy.yaml \
    --problem combinatorial_auction
```

The best-by-val-loss checkpoint lands at
`checkpoints/<experiment.name>/pretrain_best.pt`, alongside
`pretrain_history.json`.

### Stage 6 — PPO training

Warm-starts from `pretrain_best.pt` and updates the GCNN with PPO+GAE on
rollouts through the branching environment:

```bash
python -m scripts.ppo \
    --config config/base.yaml --config config/dummy.yaml \
    --problem combinatorial_auction
```

Outputs:
- `checkpoints/<experiment.name>/ppo_latest.pt` — latest weights
- `checkpoints/<experiment.name>/ppo_best.pt`   — best-by-mean-reward
- `checkpoints/<experiment.name>/ppo_history.json` — per-iteration metrics

### Stage 7 — Evaluation

Runs Random, FSB, and the trained PPO policy on the **test** split of all
three size regimes (training-size + transfer-medium + transfer-large) across
multiple seeds, then writes:

- `logs/<experiment.name>/eval_detail.csv`  — per-(policy, regime, seed, instance) rows
- `logs/<experiment.name>/eval_summary.csv` — mean ± std per (policy, regime)
- `logs/<experiment.name>/eval_summary.json` — same summary, machine-readable

```bash
python -m scripts.eval \
    --config config/base.yaml --config config/dummy.yaml \
    --problem combinatorial_auction
```

Subset the policies with `--policies random fsb` or restrict instances with
`--max-instances 5` for quick smoke runs.

---

## Reproducibility

- Every script consumes a single YAML config (defaults in `config/base.yaml`).
- All randomness is seeded from `experiment.seed`.
- The full conda environment is captured in `environment.yml`; pin updates
  through PRs so the team's environments stay aligned.

---

## Contributing

See `GIT_GUIDE.md` for the team Git workflow.
