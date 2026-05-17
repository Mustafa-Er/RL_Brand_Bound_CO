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
- **Baselines:** Random branching, FSB (Full Strong Branching), and — if
  available cleanly from Ecole — GCNN+SL (Gasse et al. 2019).
- **Metrics:** wall-clock solve time, B&B node count, dual integral.

The full project spec lives in `CLAUDE_CODE_PROMPT.md`.

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
│   ├── experts/        # RB and FSB expert interfaces (Stage 3)
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

Stage 0 only ships the skeleton. Subsequent stages will fill in commands here:

- Stage 1: `python -m rl_bb.instances.generate --config config/base.yaml`
- Stage 5: `python -m rl_bb.training.pretrain --config config/base.yaml`
- Stage 6: `python -m rl_bb.training.ppo --config config/base.yaml`
- Stage 7: `python -m rl_bb.eval.run --config config/base.yaml`

---

## Reproducibility

- Every script consumes a single YAML config (defaults in `config/base.yaml`).
- All randomness is seeded from `experiment.seed`.
- The full conda environment is captured in `environment.yml`; pin updates
  through PRs so the team's environments stay aligned.

---

## Contributing

See `GIT_GUIDE.md` for the team Git workflow.
