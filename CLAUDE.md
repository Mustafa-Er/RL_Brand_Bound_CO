# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

RL-Brand-Bound CO is a PPO+GAE branching policy for Branch-and-Bound (B&B) on MILP instances. It is pretrained via imitation learning (behavioral cloning) on Reliability Branching (RB) demonstrations, using a bipartite GCNN backbone (Gasse et al. 2019). Built on Ecole + PySCIPOpt + PyTorch Geometric.

**Key claim:** The PPO-trained policy matches or approaches Full Strong Branching (FSB) on set covering and combinatorial auction instances, and generalizes to 1.5× / 2× larger instances.

## Platform Note

Ecole has no Windows binaries. All development must happen inside **WSL2 + Ubuntu 22.04** using a separate Miniconda installation. The package is installed in editable mode.

## Environment Setup (Linux/WSL2)

```bash
conda env create -f environment.yml -n rl_bb
conda activate rl_bb
pip install -e .
# Smoke test:
python -c "import ecole, pyscipopt, torch, torch_geometric; print('ok')"
```

## Common Commands

**Run all tests:**
```bash
pytest tests/
```

**Run a single test file:**
```bash
pytest tests/test_gcnn.py
```

**Full dummy pipeline (end-to-end, ~2 min on CPU):**
```bash
python -m scripts.generate_instances --config config/base.yaml --config config/dummy.yaml
python -m scripts.collect_demonstrations --config config/base.yaml --config config/dummy.yaml --problem combinatorial_auction --split train
python -m scripts.collect_demonstrations --config config/base.yaml --config config/dummy.yaml --problem combinatorial_auction --split val
python -m scripts.pretrain --config config/base.yaml --config config/dummy.yaml --problem combinatorial_auction
python -m scripts.ppo --config config/base.yaml --config config/dummy.yaml --problem combinatorial_auction
python -m scripts.eval --config config/base.yaml --config config/dummy.yaml --problem combinatorial_auction
```

All scripts are run with `python -m scripts.<name>` (not directly), because the package must be on the Python path.

## Configuration System

All behavior is driven by YAML configs. `config/base.yaml` holds the full-scale defaults; `config/dummy.yaml` overrides specific keys for fast verification. Multiple `--config` flags are deep-merged left-to-right — later files win.

`experiment.name` controls where all output lands:
- Instances: `data/<experiment.name>/`
- Checkpoints: `checkpoints/<experiment.name>/`
- Logs/eval results: `logs/<experiment.name>/`

`experiment.seed` seeds all randomness. `device: auto` resolves to `cuda` or `cpu` at runtime.

## Architecture

### Seven-stage pipeline

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | `src/rl_bb/instances/generate.py` | Generate `.mps` MILP instances (set covering / combinatorial auction) across 3 size regimes (train_size, transfer_medium, transfer_large) × 3 splits (train/val/test) |
| 2 | `src/rl_bb/envs/` | Ecole env wrappers; DFS node selection forced via SCIP params; bipartite observation; dual-bound-gain reward |
| 3 | `src/rl_bb/experts/` | RB expert policy for demonstration collection; FSB/Random as eval baselines |
| 4 | `src/rl_bb/models/gcnn.py` | Bipartite GCNN: shared backbone → policy head (logits) + value head (mean-pooled scalar) |
| 5 | `src/rl_bb/training/pretrain.py` | Behavioral cloning via cross-entropy over the action set; saves `pretrain_best.pt` |
| 6 | `src/rl_bb/training/ppo.py` | PPO+GAE; warm-starts from `pretrain_best.pt`; saves `ppo_latest.pt` and `ppo_best.pt` |
| 7 | `src/rl_bb/eval/` | Runs all three policies (Random, FSB, PPO) across all regimes and seeds; writes CSV/JSON summaries |

### GCNN model (`src/rl_bb/models/gcnn.py`)

The `GCNN` class takes a `BipartiteTensors` namedtuple containing variable features, constraint features, edge features, and an edge index. It applies alternating cons→var and var→cons message passing (`BipartiteConv`) for `n_layers` rounds. The backbone output (variable embeddings) feeds two heads:
- **PolicyHead**: per-variable logit scalar. Callers mask non-candidates with `-inf` before sampling.
- **ValueHead**: mean-pool variable embeddings → MLP → scalar V(s).

Feature dimensions (`var_dim`, `cons_dim`, `edge_dim`) are inferred at pretrain time from the first sample and stored in every checkpoint — this is how `load_pretrained_gcnn` re-instantiates the model without hard-coded sizes.

### Environment (`src/rl_bb/envs/`)

Two env factories:
- `make_branching_env(cfg)` — bipartite observation only; used for RL rollouts.
- `make_expert_env(cfg)` — tuple `(bipartite, sb_scores)`; used by FSB and RB experts.

Both force DFS node selection via SCIP parameters defined in `envs/dfs.py`, use `DualBoundGain` as the reward, and expose `nb_nodes`, `lp_iterations`, `wall_time`, and `dual_integral` via Ecole's information functions. The `extra_scip_params` config key (used in `dummy.yaml` to disable presolving) is merged into the SCIP params at construction time.

### Expert policies (`src/rl_bb/experts/policies.py`)

All policies share the interface `act(observation, action_set, model) -> int`. The `model` here is the Ecole env's `model` (a PySCIPOpt model), not the GCNN.
- `RBPolicy`: reliability pseudocost branching — uses FSB scores until a variable has been branched `reliability` times, then switches to SCIP pseudocosts. This is the sole imitation expert.
- `FSBPolicy`: argmax over strong-branching scores (eval baseline only).
- `RandomPolicy`: uniform random (sanity-check baseline).

### Training (`src/rl_bb/training/`)

- `dataset.py`: `BCDataset` loads `.pkl` demonstration files; `collate_bipartite` batch-packs heterogeneous bipartite graphs by concatenating node arrays and tracking graph membership via `graph_ids`.
- `pretrain.py`: standard supervised loop; CE loss computed over each sample's action set (not all variables).
- `gae.py`: computes GAE advantages and bootstrapped returns for a trajectory.
- `rollout.py`: `collect_trajectory` rolls out the current GCNN policy through one instance, recording `(observation, action, action_set, reward, log_prob, value)` per step.
- `ppo.py`: outer loop over iterations; collects rollouts, applies GAE, runs `update_epochs` passes over shuffled minibatches with the clipped PPO surrogate + MSE value loss + entropy bonus. Gradient norm clipped to 1.0.

### Checkpoints

Checkpoint dicts always contain `model_state`, `feature_dims` (tuple), `model_config` (hidden, n_layers), and the relevant training metadata. `load_pretrained_gcnn` uses `feature_dims` and `model_config` to re-build the GCNN before loading state.

### Eval (`src/rl_bb/eval/`)

`runner.py`: `evaluate_on_instance` drives a policy through one instance. Records wall time, SCIP-reported time, node count, LP iterations, dual integral, and decision count.

`aggregate.py`: collects `InstanceResult` rows, writes `eval_detail.csv` (per-instance), `eval_summary.csv`, and `eval_summary.json` (mean ± std per policy × regime).

## Key Implementation Details

- **Dummy mode disables presolving** (`presolving/maxrounds: 0`) because tiny instances get solved at the root with no branching decisions; full-scale runs leave presolve on.
- **`graph_ids` tensor** in the GCNN forward pass identifies which graph each variable node belongs to in a batched forward; used for per-graph mean-pooling in the value head.
- **PySCIPOpt API compatibility**: `RBPolicy` and pseudocost helpers probe multiple API method names (e.g. `getVarPseudocostCountCurrentRun` vs. `getVarPseudocostCount`) to handle version differences across SCIP builds.
- **Ecole reward name discovery**: `branching_env.py` iterates over `("DualIntegral", "PrimalDualIntegral", "PrimalIntegral")` to find whichever name Ecole exposes in the installed version.
- **Data/checkpoint paths are git-ignored**: `data/`, `logs/`, `checkpoints/`, `*.pkl`, `*.pt`, `*.mps` are all excluded. Only `.gitkeep` files track empty directories.
