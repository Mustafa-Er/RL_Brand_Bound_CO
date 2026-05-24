"""Shared pytest fixtures for all test modules."""
from __future__ import annotations

import pytest

from rl_bb.envs import EnvConfig


@pytest.fixture(scope="session")
def no_presolve_env_cfg() -> EnvConfig:
    """EnvConfig with presolving disabled — ensures branching decisions occur on small instances."""
    return EnvConfig(time_limit_s=30.0, extra_scip_params={"presolving/maxrounds": 0})
