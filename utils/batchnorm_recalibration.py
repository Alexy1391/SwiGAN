"""Recalibrate the generator's BatchNorm running statistics on the training split.

WHY THIS EXISTS. Every eval-mode number the project produces -- the free-running
``val/rollout_rmse`` that ``ModelCheckpoint`` selects on, the teacher-forced
``val/generator_rmse``, every trajectory ``swigan/inference.py`` publishes -- normalizes by the
generator's BatchNorm running statistics. Those are a moving average weighted towards the last
training batches of the epoch the checkpoint happens to end on, so on the 500-epoch
Bourgogne-Franche-Comte run the validation score was mostly a draw of those batches: epoch 249,
trained twice from the same ``last.ckpt``, scored 0.171 once and 0.222 the other time, and the
checkpoints from e199 on spread over 0.161-0.368 with their stored statistics against
0.167-0.192 once recalibrated. Selecting on the stored-statistics score picked a checkpoint that
was among the worst on test.

WHAT IT DOES. Reset every generator BatchNorm, switch it to a cumulative average
(``momentum=None``, so every batch weighs the same instead of the last one most) and take
``num_passes`` no-grad passes over the training split in batches of the training batch size,
with the generator in train mode as it was when the statistics were first written. No weight
moves, only ``running_mean``, ``running_var`` and ``num_batches_tracked``. The pass runs on a
private RNG stream, so calling it mid-training does not shift the draws the training steps
consume. On BFC one pass is 12 forward passes and 1.2 s on a T4.

HOW STABLE IT IS. On BFC e219, e429 and e499, five seeds of one pass put the recalibrated val
rollout RMSE at a standard deviation of 0.004-0.005 and test at 0.006, against 0.053 across the
stored statistics of the same run; four passes bring val to 0.001-0.002 and test to 0.004. The
seed is held fixed, so the draw is the same at every epoch and the curve moves with the weights.

WHY THE WHOLE GENERATOR IN TRAIN MODE. With stochastic depth active the average is over the
same mixture of dropped and kept residual blocks that the training-time moving average
estimates, so this is that estimate with the lottery taken out; it reproduces the recalibrated
numbers of the report that found the problem. Putting only the BatchNorms in train mode, which
collects statistics on exactly the eval-mode path, is quieter (val s.d. 0.001 or less at four
passes) but on those three checkpoints its five-seed means scored 0.011-0.016 worse on test and up
to 0.004 worse on val.
"""

from __future__ import annotations

import torch
from lightning import LightningModule
from torch.utils.data import Dataset

from modules.batchnorm import batchnorm_layers


def recalibrate_batchnorm(
    model: LightningModule,
    dataset: Dataset,
    batch_size: int = 64,
    num_passes: int = 1,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> int:
    """Replace the generator's BatchNorm running statistics with their average over ``dataset``.

    Args:
    ----
        model: The ``TTTSWIGAN``. Only ``model.generator`` is touched: it is the only part the
            forward pass runs, so a critic BatchNorm reset here would never be refilled.
        dataset: The training split, as a :class:`~utils.swi_dataset.SWIDataset`.
        batch_size: Samples per forward pass. Use the training batch size: the running variance
            averages within-batch variances, which is what train mode normalized by.
        num_passes: Passes over ``dataset``, each in a fresh order. The cumulative average runs
            over all of them.
        seed: Seed of the private RNG stream that draws the batch order, the bottleneck vectors,
            the injection noise and the stochastic-depth draws. Held fixed so two calls on the
            same weights agree.
        device: Where to run. Defaults to the device the model is already on.

    Returns:
    -------
        The number of BatchNorm layers recalibrated; 0 for a generator without BatchNorm.

    """
    if num_passes < 1:
        raise ValueError(f"num_passes must be at least 1, got {num_passes}.")
    layers = batchnorm_layers(model.generator)
    if not layers:
        return 0
    if device is None:
        device = next(model.parameters()).device

    # Restored module by module rather than with ``model.train(mode)``, which would flatten
    # a caller's mixed state -- inference's dropout-active mode, for one.
    modes = [(module, module.training) for module in model.modules()]
    momenta = [layer.momentum for layer in layers]
    try:
        for layer in layers:
            layer.reset_running_stats()
            layer.momentum = None
        model.generator.train()
        # The private stream of ``utils.rollout._frozen_rng``, which this module cannot import:
        # utils.rollout's imports come back round to swigan.inference, which imports this one.
        with torch.random.fork_rng(), torch.no_grad():
            torch.manual_seed(seed)
            for _ in range(num_passes):
                order = torch.randperm(len(dataset)).tolist()
                for start in range(0, len(order), batch_size):
                    samples = [dataset[i] for i in order[start : start + batch_size]]
                    model(
                        input_maps=torch.stack([s["input_maps"] for s in samples]).to(device),
                        input_timestamps=torch.stack(
                            [s["timestamps"].reshape(()) for s in samples]
                        ).to(device),
                        mask=torch.stack([s["mask"] for s in samples]).to(device),
                        noise_vector=torch.randn(len(samples), model.hparams.z_dim, device=device),
                    )
    finally:
        for layer, momentum in zip(layers, momenta, strict=True):
            layer.momentum = momentum
        for module, training in modes:
            module.training = training
    return len(layers)
