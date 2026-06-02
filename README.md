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
   conda env create -f environment.yml -n rl_bb
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
conda env create -f environment.yml -n rl_bb
conda activate rl_bb
python -c "import ecole, pyscipopt, torch, torch_geometric; print('ok')"
```

---

## Directory Structure

```
RL_Brand_Bound_CO/
├── config/                  # YAML configs: base.yaml + dummy.yaml override
├── notebooks/
│   └── demo.ipynb           # End-to-end walkthrough of all four stages
├── src/rl_bb/               # Library code (one module per concern)
│   ├── utils.py             # Config loader, seeding, logging, paths
│   ├── envs.py              # Ecole branching env, DFS, dual-bound-gain reward
│   ├── experts.py           # RB expert + FSB / Random baselines
│   ├── model.py             # GCNN + observation→tensor + checkpoint I/O
│   ├── data.py              # BC dataset + manual bipartite batching
│   ├── stage_1_instances.py # CLI: instance generation (+ optional env smoke)
│   ├── stage_2_pretrain.py  # CLI: GCNN + SL with caching
│   ├── stage_3_ppo.py       # CLI: PPO + GAE warm-started from Stage 2
│   └── stage_4_eval.py      # CLI: Random vs FSB vs PPO across regimes
├── tests/                   # pytest suite covering each stage
├── data/                    # Generated instances + demonstrations (git-ignored)
├── checkpoints/             # Model checkpoints (git-ignored)
└── logs/                    # Stage logs + eval CSV/JSON (git-ignored)
```

Each stage module is runnable both from the command line
(`python -m rl_bb.stage_N_...`) and from the demo notebook
(`from rl_bb.stage_N_... import run_stage_N`), so you can mix scripted and
interactive workflows freely.

---

## How to Run

Install the package in editable mode once (so `scripts/` and `rl_bb` resolve):

```bash
pip install -e .
```

### End-to-end reproduction (dummy mode)

The project is organized into **four stages**, each a single Python module
under `src/rl_bb/`. Run them in order; dummy mode finishes in 3–5 minutes on
a CPU and exercises every stage end to end:

```bash
# 0. One-time setup (see Setup section above first)
conda activate rl_bb
pip install -e .

# 1. Generate MILP instances (set covering + combinatorial auction)
#    Add --smoke to also run a random-policy rollout for sanity.
python -m rl_bb.stage_1_instances \
    --config config/base.yaml --config config/dummy.yaml \
    --problem combinatorial_auction

# 2. GCNN + Supervised Learning (behavioral cloning on RB demonstrations).
#    Cache modes via config.pretrain.mode or --mode:
#      auto           load checkpoint if present, else collect demos and train
#      force_retrain  always retrain, overwrite any checkpoint
#      load_only      require an existing checkpoint, error if missing
python -m rl_bb.stage_2_pretrain \
    --config config/base.yaml --config config/dummy.yaml \
    --problem combinatorial_auction

# 3. PPO+GAE training (warm-starts from the Stage 2 checkpoint).
python -m rl_bb.stage_3_ppo \
    --config config/base.yaml --config config/dummy.yaml \
    --problem combinatorial_auction

# 4. Evaluation: Random vs. FSB vs. PPO across all three size regimes.
python -m rl_bb.stage_4_eval \
    --config config/base.yaml --config config/dummy.yaml \
    --problem combinatorial_auction
```

Results land under `data/rl_bb_dummy/` (instances + demonstrations),
`checkpoints/rl_bb_dummy/` (model weights), and `logs/rl_bb_dummy/`
(stage logs + final eval CSV/JSON).

For the **full-scale run** that backs the paper's results, drop
`--config config/dummy.yaml` so only `config/base.yaml` is used. This grows
instances from a few hundred to ten thousand and PPO from 3 to 50
iterations; expect hours to a day depending on hardware.

### Stage details

#### Stage 1 — `rl_bb.stage_1_instances`
Generates `.mps` files for set covering and combinatorial auction at three
size regimes (`train_size`, `transfer_medium`, `transfer_large`). Pass
`--smoke` to additionally roll a random policy through a couple of
instances and confirm the Ecole env wires up correctly.

#### Stage 2 — `rl_bb.stage_2_pretrain` ("GCNN + SL")
Three things in one module:
1. Collects Reliability Branching demonstrations (if missing).
2. Trains a bipartite GCNN on those demos with supervised cross-entropy
   over each sample's action set.
3. Writes the best-by-val-loss checkpoint to
   `checkpoints/<exp>/pretrain_best.pt`.

`config.pretrain.mode` controls whether to retrain or reuse the cached
checkpoint (`auto` / `force_retrain` / `load_only`). Override at the
command line with `--mode`.

#### Stage 3 — `rl_bb.stage_3_ppo`
PPO+GAE on the Branching env; warm-starts from `pretrain_best.pt`. Writes
`ppo_latest.pt` (every iter), `ppo_best.pt` (best mean reward), and
`ppo_history.json`. Errors out if the Stage 2 checkpoint is missing.

#### Stage 4 — `rl_bb.stage_4_eval`
Runs Random, FSB, and the trained PPO policy on the **test** split of all
three regimes across the seeds listed in `config.eval.seeds`. Writes:

- `logs/<exp>/eval_detail.csv`  — one row per (instance, policy, seed)
- `logs/<exp>/eval_summary.csv` — mean / std per (policy, regime, split)
- `logs/<exp>/eval_summary.json`

Subset the policies with `--policies random fsb` or restrict instances with
`--max-instances 5` for quick smoke runs.

### Run unit tests

```bash
pytest tests/
```

---

## Reproducibility

- Every script consumes a single YAML config (defaults in `config/base.yaml`).
- All randomness is seeded from `experiment.seed`.
- The full conda environment is captured in `environment.yml`; pin updates
  through PRs so the team's environments stay aligned.

