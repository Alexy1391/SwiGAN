"""Inference-time module modes that the probes and the shipping path both switch.

Kept dependency-free on purpose: ``swigan/inference.py`` needs ``dropout_active`` and the
probes under ``utils/`` import ``swigan/inference.py``, so it cannot live next to a probe
without closing a cycle.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def dropout_active(model: torch.nn.Module, enabled: bool) -> Iterator[None]:
    """Put only the ``nn.Dropout`` modules into training mode, leaving everything else in eval.

    This is the precise question being asked: ``model.eval()`` stays in force for batch norm and
    for stochastic depth (whose own ``training`` flag lives on a different module class and is
    untouched here), so any spread that appears is dropout's and nothing else's.
    """
    if not enabled:
        yield
        return
    mods = [m for m in model.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    if not mods:
        raise RuntimeError(
            "no nn.Dropout module with p > 0 -- this checkpoint was trained with dropout "
            "disabled everywhere, so there is nothing to switch on"
        )
    for m in mods:
        m.train()
    try:
        yield
    finally:
        for m in mods:
            m.eval()
