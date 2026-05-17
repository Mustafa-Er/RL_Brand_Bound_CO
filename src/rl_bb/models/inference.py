"""``Policy``-compatible wrapper around a trained :class:`GCNN`.

Use cases:
- Stage 6/7: drop the trained model into ``run_env_smoke`` and the eval loop
  as just another policy.
- Stage 5: validate that pretrained weights produce action choices.
"""
from __future__ import annotations

import torch

from rl_bb.models.gcnn import GCNN
from rl_bb.models.obs_to_tensors import obs_to_tensors


class RLPolicy:
    """Argmax (or sampled) policy backed by a GCNN.

    Parameters
    ----------
    model:
        A trained :class:`GCNN` instance. Must be in ``.eval()`` mode for
        deterministic argmax behavior.
    device:
        Device on which ``model`` lives. Observations are moved here at
        inference time.
    stochastic:
        If ``True``, sample from the masked softmax; otherwise argmax.
    """

    def __init__(self, model: GCNN, device: str = "cpu", stochastic: bool = False) -> None:
        self.model = model
        self.device = device
        self.stochastic = stochastic
        self.model.to(device)

    def reset(self) -> None:
        pass

    @torch.no_grad()
    def act(self, observation, action_set, model) -> int:
        # The expert env yields (bipartite, sb_scores) — strip the scores.
        bipartite = observation[0] if isinstance(observation, tuple) else observation
        tensors = obs_to_tensors(bipartite).to(self.device)
        logits, _value = self.model.forward_with_mask(tensors, action_set)
        if self.stochastic:
            probs = torch.softmax(logits, dim=-1)
            action = int(torch.multinomial(probs, num_samples=1).item())
        else:
            action = int(torch.argmax(logits).item())
        return action
