"""Which generator passes may write the BatchNorm running statistics.

A BatchNorm keeps a moving average of the batch statistics it sees in train mode, and eval
mode -- the free-running rollout, the teacher-forced validation, inference -- normalizes by
that average alone. What goes into it is therefore part of the model a checkpoint describes.
``critic_step`` runs the generator in train mode ``num_critic_iterations_per_epoch`` times per
batch, all on the same batch and noise vector, and the generator step once more, so at
``momentum=0.1`` each batch was written six times and the last one of an epoch made up
1 - 0.9^6 = 47% of every statistic the validation rollout was scored with. The critic passes
only need the generator's output, not its statistics; :func:`frozen_running_statistics` keeps
them out.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from torch import nn


def batchnorm_layers(module: nn.Module) -> list[nn.modules.batchnorm._BatchNorm]:
    """Every BatchNorm under ``module`` that keeps running statistics."""
    return [
        layer
        for layer in module.modules()
        if isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.track_running_stats
    ]


@contextmanager
def frozen_running_statistics(module: nn.Module) -> Iterator[None]:
    """Run the block with every BatchNorm under ``module`` leaving its running statistics alone.

    Train mode still normalizes by the batch, so the forward pass returns exactly what it
    would otherwise; only ``running_mean``, ``running_var`` and ``num_batches_tracked`` are
    not written. Eval mode is unaffected: it reads the running statistics either way.
    """
    layers = batchnorm_layers(module)
    for layer in layers:
        layer.track_running_stats = False
    try:
        yield
    finally:
        for layer in layers:
            layer.track_running_stats = True
