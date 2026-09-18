"""The lightning module for the GAN."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa
from lightning import LightningModule
from torch import autograd, nn

from modules.discriminator.base_discriminator import BaseDiscriminator
from modules.discriminator.frame_discriminator import FrameDiscriminator
from modules.discriminator.patch_gan_discriminator import PatchGANDiscriminator
from modules.generator.unet_generator import UNetGenerator
from modules.masking import diff_augment_with_mask, mask_pyramid, masked_mean
from modules.utils import COHERENT_NOISE_INITS, NOISE_WEIGHT_INITS

# The generator parameters that scale injected noise rather than transform activations, and
# so belong in the param group `noise_weight_lr_scale` multiplies. Matched as exact attribute
# names: a substring test would be a standing invitation for a future gain to join the group,
# or miss it, by accident of what it was called.
NOISE_GAIN_ATTRIBUTES = (
    "noise_weights1",
    "noise_weights2",
    "coherent_noise_weights1",
    "coherent_noise_weights2",
)

# The article's critic-side augmentation, applied to the maps the critics score. The
# region mask is carried through it so each critic pass knows where its data is.
DIFF_AUGMENT_POLICY = "translation,cutout"

# Pixel terms that score a mini-ensemble rather than one sample. See ``ensemble_pixel_distance``.
ENSEMBLE_LOSSES: tuple[str, ...] = ("crps", "energy")


def _generator_padding(height: int, width: int, block: int = 48) -> tuple[int, int, int, int]:
    """Compute the padding that grows an (H, W) map to the next multiple of ``block``.

    The UNet's five stride-2 downsamples take 48 -> 24 -> 12 -> 6 -> 3 -> 1, and the first
    decoder block compensates the odd 3 -> 1 step with its own ``additional_pad``. Anything
    that is not a multiple of ``block`` desynchronises the skip connections, so the input is
    padded up to one before it reaches the generator and cropped back afterwards. Note that
    the compensation is hardcoded for that 3 -> 1 step, so 48 is currently the only working
    size: 96 downsamples to 3 and the decoder's fixed pad then overshoots the skip. Any grid
    up to 48x48 works; a larger canvas needs the decoder pad made adaptive first.

    ``forward`` crops the output with the same (left, top) offsets returned here rather than
    with a center crop, which rounds the odd pixel the other way.

    The surplus is split evenly with the odd pixel going to the top/left, which reproduces
    the ``(2, 2, 6, 5)`` that was hardcoded here for the 37x44 Grand Est grid.

    Args:
    ----
        height: Height of the map fed to the generator.
        width: Width of the map fed to the generator.
        block: The multiple the padded map must be a multiple of.

    Returns:
    -------
        A ``F.pad`` tuple ``(left, right, top, bottom)``.

    """
    pad_h = -height % block
    pad_w = -width % block
    return (pad_w // 2 + pad_w % 2, pad_w // 2, pad_h // 2 + pad_h % 2, pad_h // 2)


class TTTSWIGAN(LightningModule):
    """Lightning module for training the GAN.

    The model is trained with a Wasserstein loss with gradient penalty.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        input_map_dims: list[int],
        encoder_channels: list[int],
        decoder_channels: list[int],
        timestamps_dim: int,
        spatial_dropout: float,
        apply_center_block: bool,
        z_dim: int,
        lr: float,
        weight_decay: float,
        loss_fn: nn.Module | str,
        optim: torch.optim.Optimizer,
        normalization: str,
        patch_critic_loss: str = "hinge",
        patch_aggregation: str = "mean",
        gradient_penalty_reduction: str = "sum",
        num_critic_iterations_per_epoch: int = 5,
        lambda_penalty: float = 10.0,
        lambda_penalty_patch: float | None = None,
        lambda_penalty_frame: float | None = None,
        image_distance_weight: float = 100.0,
        feature_matching_weight: float = 1.5,
        max_epochs: int = 100,
        min_lr: float = 1e-9,
        noise_weight_init: str = "randn",
        coherent_noise_init: str | float | None = None,
        noise_weight_lr_scale: float = 1.0,
        encoder_late_dropout: float = 0.0,
        critic_mask_channel: bool = True,
        critic_normalization: str | None = "instancenorm",
        generator_mask_channel: bool = True,
        generator_masked_norm: bool = True,
        critic_on_padded_canvas: bool = False,
        diff_augment_on_region: bool = False,
        ensemble_size: int = 1,
    ) -> None:
        """Initialize the input arguments.

        Args:
        ----
            input_channels: Number of input channels of the input maps.
            output_channels: Number of output channels of the predicted maps.
            input_map_dims: A tuple (H, W) specifying the height and width of the
                input maps. Only used to derive the critics' tile grid, which the
                "learned" patch aggregator needs to size its readout.
            temporal_dims: The dimension of the time vectors.
            encoder_channels: The output channels of the convolution blocks
                for the frame encoder.
            decoder_channels: The output channels of the convolution blocks
                for the frame decoder.
            timestamps_dim: Number of dimensions of the timestamps vector.
                Encoded using nn.Embedding.
            latent_code_dim: The output dimension of the frame and temporal encoders.
            spatial_dropout: The dropout rate to apply after each convolutional block.
            apply_center_block: Whether to apply the center block at the end of the frame
                encoding.
            z_dim: The input dimension of the noise vector and the output dimension of the
                temporal decoder.
            num_temporal_layers: The number of layers in the temporal encoder/decoder.
            rnn_cell_type: Either 'gru' or 'lstm'.
            bidirectional: Whether to use a bidirectional LSTM for the temporal encoder/decoder.
            lr: Learning rate.
            weight_decay: The weight decay.
            loss_fn: The loss function to apply as an added pixel reconstruction term.
                "l1" and "mse" score the one sample the critics see. "crps" and "energy"
                are proper scores over a mini-ensemble of ``ensemble_size`` samples per
                input -- the fair CRPS per cell and the energy score per map -- and are the
                only terms in the objective that credit spread. The L1 term is minimised by
                the conditional median, so every step of training charges the generator for
                noise sensitivity and nothing pays it back; that is what trained the
                dispersion away (region-mean spread 0.011 SWI on Corse against a required
                0.04-0.18) and left a bias that is conditional rather than seasonal. A
                proper score is minimised by the true predictive distribution, so it credits
                spread exactly where the data warrant it, month by month. See
                ``ensemble_pixel_distance`` for the two forms.
            optim: The optimizer.
            normalization: normalization: The type of normalization to apply.
                If None, no normalization is applied. Supported normalization are
                "instancenorm" for InstanceNorm2D, "batchnorm" for BatchNorm2D.
            patch_critic_loss: The adversarial loss applied to the patch critic grid.
                Either "wasserstein" (Eq. 3.5, the unweighted mean of the raw scores)
                or "hinge" (a per-tile margin loss). Under "wasserstein" the loss is
                linear in the critic output, so the mean commutes through the 1x1
                final conv and the whole 4x5 grid collapses to a single linear readout
                of the globally pooled features. "hinge" scores each tile against its
                own margin, which keeps the tiles as separate degrees of freedom.
            patch_aggregation: How the patch critic's tile grid is reduced to the scalar
                the loss consumes. "mean" is Eq. 3.5's unweighted average, applied in the
                loss; since the final conv is 1x1 and the WGAN loss is linear, that mean
                commutes through the conv and the 20 tiles collapse to one linear readout
                of the globally pooled features. "learned" reduces the grid inside the
                critic through a non-linear MLP over the tile scores, so the critic is
                still one scalar function -- duality and the gradient penalty intact --
                but no longer a linear readout, and each tile gets its own gradient.
            gradient_penalty_reduction: How the critic grid is reduced to a scalar before
                differentiating for the gradient penalty. Either "sum" or "mean". "sum"
                constrains the 20-output patch critic ~20x more tightly than the single
                output frame critic, so lambda_penalty does not mean the same thing for
                the two heads; "mean" differentiates the functional the loss consumes.
            num_critic_iterations_per_epoch: Number of critic updates per epoch.
            lambda_penalty: The weight to apply to the gradient penalty of the
                critic loss. Used for whichever of the two per-head weights below is
                left unset.
            lambda_penalty_patch: Gradient penalty weight for the patch head only.
                Defaults to lambda_penalty. Split from the frame head's weight because
                the two heads' penalties are not the same constraint: the frame critic
                emits a single value, so "sum" and "mean" reduction agree on it and its
                weight has meant the same thing in every round, while the patch critic's
                20 outputs made its weight reduction-dependent. Sweeping one shared
                weight moves both heads at once.
            lambda_penalty_frame: Gradient penalty weight for the frame head only.
                Defaults to lambda_penalty.
            image_distance_weight: The weight to apply to the pixel reconstruction
                loss term.
            feature_matching_weight: Weight to apply to the feature matching term in the loss.
            max_epochs: Max number o fepochs of the training.
            min_lr: Minimum learning rate for the CosineAnnelaing scheduler.
            scheduler_factor: Factor of the scheduler.
            scheduler_patience: How often to update the learning rate through
                the scheduler.
            scheduler_threshold: The minimum threshold on the monitored metric over
                which the learning rate is updated.
            coherent_noise_init: How the generator's SPATIALLY COHERENT noise gains are
                initialized, or None to leave that channel out of the model entirely, which
                is what every checkpoint up to the 250-epoch arm was trained without. The
                other noise channel is drawn per pixel, so the region mean averages it away:
                the shipped ensemble's region/pixel spread ratio is 0.13-0.25 against the
                0.63-0.90 of the error it has to span. This channel's draw is constant over
                height and width and so survives that averaging. "zeros" adds it switched
                off, which is what a fine-tune wants; "randn" starts it at rms 1.0 for
                training from scratch. It shares the ``noise_weight_lr_scale`` param group,
                and at scale 1 it cannot travel any further than the gains below can. See
                ``modules.utils.init_coherent_noise_weights``.
            noise_weight_init: How the 20 per-channel noise gains in the generator's
                encoder and decoder blocks are initialized. "randn" is what rounds 1-8
                trained with; "zeros" is StyleGAN's convention, where a block starts as
                its deterministic self and has to earn any noise it uses. Read
                ``modules.utils.init_noise_weights`` before choosing: at this run's
                learning rate the gains move ~1e-3 over 120 epochs against a randn draw
                at rms 1.0, so the init is closer to a fixed hyperparameter than to a
                starting point, and "zeros" behaves more like training with injection off
                than like letting the model find its own noise scale. "randn3" is rms 3.0,
                which exists only to start the gains ABOVE round 10 arm C's fitted asymptote
                of 1.489 and see whether they are pulled back down to it; it requires a large
                ``noise_weight_lr_scale`` to mean anything.
            encoder_late_dropout: Dropout rate for the last three downsampling blocks only.
                Paper §8.2.1 prescribes dropout in the last three downsampling and first three
                upsampling layers, citing Isola et al. [2018] on generators that ignore the
                noise vector; only the upsampling half was ever implemented. 0.0 is rounds 1-10.
            noise_weight_lr_scale: Multiplier on the generator learning rate for the noise
                gains only, applied as a separate AdamW param group. 1.0 reproduces rounds
                1-8. The gains need their own scale because the shared learning rate is set
                for Glorot-initialized convs at rms 0.017-0.16; a gain that has to reach
                O(0.1) to affect the output cannot get there on a ~1e-3 travel budget. At
                scale 100 that budget becomes ~1e-1, which is the range where the gains can
                actually settle where the loss wants them rather than where they started.
            critic_mask_channel: Whether the critic trunk reads the region mask as an extra
                input channel. Real and generated maps share the mask per sample, so it is
                not a real/fake shortcut; it lets the first convolution tell padding from a
                cell whose standardized value happens to be 0.
            critic_normalization: "instancenorm" for the masked InstanceNorm2d -- statistics
                over the region's cells only, padding zeroed at every level -- or null for
                no normalization in the critic beyond spectral norm, which is the WGAN-GP
                authors' recommendation and a useful ablation.
            generator_mask_channel: Whether the generator reads the region mask as an extra
                input channel. Without it the coast is only visible as "every channel is
                exactly 0", and 0 is the standardized mean.
            generator_masked_norm: Whether the generator's 37 normalization layers take their
                statistics over the region's cells, its ten scSE channel gates pool over
                them, and every block re-zeroes its output. Off, BatchNorm averages the whole
                padded canvas: on Corse that is 94% padding, so where the padding sits near
                zero the variance is diluted by the coverage fraction and the region's
                normalized activations come out 1/sqrt(c) times *too large* -- measured here,
                3.8x at the first encoder block and a median 1.1x over the 24 spatial norms,
                the depth decay coming from the bias-filled plateau the deeper padding
                carries. The three parts are one change. Re-zeroing alone leaves the dilution
                intact, because a zero is still a cell BatchNorm counts; a masked norm alone
                writes an out-of-range constant onto the padding that the next 3x3
                convolution bleeds straight back across the coast, which is why the critic's
                ``masked_instance_norm`` ends in ``* mask``.

                The size of that factor is the smaller half of what this buys. Gamma and the
                following convolution can in principle absorb a per-channel constant over
                1500 epochs; what they cannot absorb is its *dependence on the coverage*. A
                batch mixing Grand Est (38.9% of its grid) with Corse (6.1% of the canvas)
                normalizes the same map differently depending on who it is batched with, so
                this is a prerequisite for the per-region batch sampler and for Grand Est to
                Corse transfer, not a polish on the single-region runs.

                Note what re-zeroing asserts: outside the mask there is nothing. That is
                true of Corse, whose outside is the sea, and true of any region whose mask
                is "the cells this run has data for" -- which is what the loss, the critic
                and ``forward``'s closing ``* mask`` already assume. It would be the wrong
                boundary condition for a Grand Est run whose outside is French land held in
                another file; the switch follows the mask, so give it a mask that means it.
            critic_on_padded_canvas: Whether the critics score the maps on the 48x48
                canvas the generator pads to, with the region at the same offset, rather
                than on the native grid. The two are not equivalent: a fully convolutional
                critic on the native grid drops every window that hangs off the grid's
                edge, at every level, so Corse's 23x11 reaches the heads as 6 trunk cells
                and 2 tiles, while on the canvas the same island is under 11 trunk cells
                and 19 tiles. Padding tiles never enter the loss either way. Note that
                DiffAugment's translation and cutout are sized relative to the map unless
                ``diff_augment_on_region`` is set, so on the canvas they are otherwise the
                canvas's 20% and 30%, not the region's.
            diff_augment_on_region: Whether DiffAugment sizes and places its translation
                and cutout on each sample's region bounding box instead of on the whole
                map. False is the stock policy every round so far trained on, and is the
                same computation either way on a native grid whose extent is the region.
                It is ``critic_on_padded_canvas`` that makes the two differ: the canvas
                widens the translation to 20% of 48 and grows the cutout to a 14x14 box,
                which on Corse's 23x11 island removes more than half of it on 6.2% of
                draws against a 15% ceiling on the native grid. See
                :func:`modules.masking.diff_augment_with_mask`.
            ensemble_size: Generator samples drawn per input in the generator step. 1 is
                every round so far. With "crps" or "energy" it must be at least 2 -- the
                spread term needs a pair -- and 4 is the working value: the K samples run
                as one generator batch of K x batch_size, so this is the memory knob, and
                the five critic iterations are unchanged, so the epoch costs roughly 2x at
                K = 4. The adversarial, feature-matching and diagnostic terms stay on the
                first sample.

        """
        super().__init__()
        self.timestamps_embedding = nn.Embedding(12, timestamps_dim)
        self.generator = UNetGenerator(
            input_dim=input_channels + timestamps_dim + int(generator_mask_channel),
            output_dim=output_channels,
            noise_dim=z_dim,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=spatial_dropout,
            normalization=normalization,
            apply_center_block=apply_center_block,
            noise_weight_init=noise_weight_init,
            coherent_noise_init=coherent_noise_init,
            encoder_late_dropout=encoder_late_dropout,
        )

        # The tile grid the heads score is derived from the grid size instead of being
        # asserted to be the (5, 6) of the 36x44 Grand Est grid, so the same critic runs on
        # Corse's 23x11 (tiles 2x1), on the 48x48 canvas the generator pads to (5x5) or on
        # a 64x64 canvas (7x7).
        critic_dims = self.critic_map_dims(input_map_dims, critic_on_padded_canvas)
        tile_grid = mask_pyramid(torch.ones(1, 1, *critic_dims))[-1]
        tile_shape = (int(tile_grid.shape[-2]), int(tile_grid.shape[-1]))
        self.base_critic = BaseDiscriminator(
            input_channels=output_channels,
            output_channels=[32, 64, 128],
            mbd_output_channels=64,
            mask_channel=critic_mask_channel,
            normalization=critic_normalization,
        )
        self.patch_critic = PatchGANDiscriminator(
            input_channels=128 + 64,
            aggregation=patch_aggregation,
            grid_shape=tile_shape,
            normalization=critic_normalization,
        )
        self.frame_critic = FrameDiscriminator(
            input_channels=128 + 64, normalization=critic_normalization
        )

        # No geometric augmentation is applied to the (input, target, mask) triplet. A
        # RandomHorizontalFlip + RandomResizedCrop pipeline used to sit here, but neither is
        # admissible on a fixed projected grid: the flip mirrors the Grand Est, and the crop
        # rescales and stretches the maps (its default ratio=(0.75, 1.33) distorts the aspect
        # by up to 1.42x), so both train the generator on a geometry that never occurs at
        # inference. The article only prescribes DiffAugment (translation + cutout) on the maps
        # fed to the critics, which is applied in generator_step and critic_step.

        if patch_critic_loss not in ("wasserstein", "hinge"):
            raise NotImplementedError(
                f"Unknown patch critic loss: {patch_critic_loss}. "
                f"Please provide one of 'wasserstein', 'hinge'."
            )
        if gradient_penalty_reduction not in ("sum", "mean"):
            raise NotImplementedError(
                f"Unknown gradient penalty reduction: {gradient_penalty_reduction}. "
                f"Please provide one of 'sum', 'mean'."
            )
        if noise_weight_init not in NOISE_WEIGHT_INITS:
            raise NotImplementedError(
                f"Unknown noise weight init: {noise_weight_init}. "
                f"Please provide one of {NOISE_WEIGHT_INITS}."
            )
        if (
            coherent_noise_init is not None
            and isinstance(coherent_noise_init, str)
            and coherent_noise_init not in COHERENT_NOISE_INITS
        ):
            raise NotImplementedError(
                f"Unknown coherent noise init: {coherent_noise_init}. Please provide one of "
                f"{COHERENT_NOISE_INITS}, a float for a constant gain, or None to leave the "
                "channel out of the model."
            )

        # Resolved before save_hyperparameters so the checkpoint records the weight each
        # head actually ran with, not None.
        if lambda_penalty_patch is None:
            lambda_penalty_patch = lambda_penalty
        if lambda_penalty_frame is None:
            lambda_penalty_frame = lambda_penalty

        # ``pixel_distance`` computes the reconstruction term itself for the string
        # choices, averaged over the region's cells; ``loss_fn`` is kept for a custom module
        # and for the diagnostics that call it directly. The two proper scores read as L1
        # wherever a single sample is scored, which is their one-member value.
        self.pixel_distance_kind = loss_fn if isinstance(loss_fn, str) else None
        if isinstance(loss_fn, str):
            if loss_fn in ("l1", *ENSEMBLE_LOSSES):
                self.loss_fn = nn.L1Loss(reduction="mean")
            elif loss_fn == "mse":
                self.loss_fn = nn.MSELoss(reduction="mean")
            else:
                raise NotImplementedError(f"Unknown loss function: {loss_fn}")
        else:
            self.loss_fn = loss_fn()
        self._check_ensemble_size(self.pixel_distance_kind, int(ensemble_size))

        self.automatic_optimization = False
        self.save_hyperparameters(ignore="loss_fn")

    @staticmethod
    def _check_ensemble_size(pixel_distance_kind: str | None, ensemble_size: int) -> None:
        """Reject an ensemble size the pixel term cannot work with."""
        if ensemble_size < 1:
            raise ValueError(f"ensemble_size must be at least 1, got {ensemble_size}")
        if pixel_distance_kind in ENSEMBLE_LOSSES and ensemble_size < 2:
            raise ValueError(
                f"loss_fn={pixel_distance_kind!r} scores a mini-ensemble and needs "
                f"ensemble_size >= 2, got {ensemble_size}"
            )

    @staticmethod
    def critic_map_dims(input_map_dims: list[int], on_padded_canvas: bool) -> list[int]:
        """(H, W) of the maps the critics score."""
        if not on_padded_canvas:
            return list(input_map_dims)
        height, width = input_map_dims
        left, right, top, bottom = _generator_padding(height, width)
        return [height + top + bottom, width + left + right]

    def to_critic_canvas(
        self, maps: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Place ``maps`` and ``mask`` where the critics score them.

        With ``critic_on_padded_canvas`` this is the canvas the generator padded to, with
        the region at the same offset ``forward`` used; otherwise the native grid,
        untouched. Every consumer of the critics -- the two training steps and the
        diagnostics under ``utils/`` -- goes through here so they all score the same maps.

        Args:
        ----
            maps: Maps of shape (batch_size, channels, H, W), zero outside the mask.
            mask: Region mask of shape (batch_size, 1, H, W).

        Returns:
        -------
            The maps and the mask, padded to the canvas if enabled.

        """
        if not self.hparams.critic_on_padded_canvas:
            return maps, mask
        padding = _generator_padding(int(maps.shape[-2]), int(maps.shape[-1]))
        return F.pad(maps, padding), F.pad(mask, padding)

    def critic_forward(self, maps: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
        """Score ``maps`` with the trunk and both heads under ``mask``.

        The coverage pyramid is derived once here and handed to the heads, whose tile grid
        is its last level; the trunk derives the same pyramid internally for its
        normalization layers.

        Args:
        ----
            maps: Maps of shape (batch_size, channels, H, W), zero outside the mask.
            mask: Region mask of shape (batch_size, 1, H, W).

        Returns:
        -------
            The patch grid (or scalar under "learned" aggregation), the frame score, the
            three feature lists for feature matching, and the coverage pyramid.

        """
        levels = mask_pyramid(mask, len(self.base_critic.blocks))
        base, features_base = self.base_critic(maps, mask)
        patch, features_patch = self.patch_critic(base, weight=levels[-1])
        frame, features_frame = self.frame_critic(base, weight=levels[-1])
        return {
            "patch": patch,
            "frame": frame,
            "features_base": features_base,
            "features_patch": features_patch,
            "features_frame": features_frame,
            "levels": levels,
        }

    @staticmethod
    def patch_scalar(scores: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """One patch score per sample.

        Under "mean" aggregation the head returns the tile grid and this is Eq. 3.5's mean
        weighted by coverage, so a tile with no data under it contributes nothing and a
        coastal tile counts in proportion to its land. Under "learned" the head already
        returns the scalar.

        Args:
        ----
            scores: The patch head output, (batch_size, 1, h, w) or (batch_size, 1).
            weight: The tile-grid coverage, (batch_size, 1, h, w).

        Returns:
        -------
            A tensor of shape (batch_size,).

        """
        if scores.dim() == 4:
            return masked_mean(scores, weight, dims=(1, 2, 3))
        return scores.flatten(1).mean(dim=1)

    @staticmethod
    def valid_patch_scores(scores: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Flatten the patch scores, keeping only the tiles with data under them."""
        if scores.dim() == 4:
            return scores[(weight > 0).expand_as(scores)]
        return scores.flatten()

    def pixel_distance(
        self, fake_maps: torch.Tensor, target_maps: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruction distance averaged over the region's cells only.

        ``nn.L1Loss(reduction="mean")`` averages over the whole raster, so its effective
        weight scales with the fill fraction: 0.56 on the 36x44 Grand Est grid, 0.55 on
        Corse's 23x11, 0.034 for Corse on a 64x64 canvas. Averaging over the mask keeps
        ``image_distance_weight`` meaning the same thing on every grid.

        Args:
        ----
            fake_maps: The generated maps, (batch_size, channels, H, W).
            target_maps: The observed maps, same shape.
            mask: The region mask, (batch_size, 1, H, W).

        Returns:
        -------
            The distance, a scalar.

        """
        if self.pixel_distance_kind in ("l1", *ENSEMBLE_LOSSES):
            error = (fake_maps - target_maps).abs()
        elif self.pixel_distance_kind == "mse":
            error = (fake_maps - target_maps) ** 2
        else:
            return self.loss_fn(fake_maps, target_maps)
        return masked_mean(error, mask, dims=(1, 2, 3)).mean()

    def ensemble_pixel_distance(
        self, samples: torch.Tensor, target_maps: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a mini-ensemble against the observed map with a proper score; also its spread.

        Both estimators are the fair (unbiased in K) forms of Ferro et al. [2008]: the pair
        sum runs over unordered pairs with weight ``1 / (K (K - 1))``, the form that does not
        reward under-dispersion at small K -- the failure mode this term exists to undo.

          crps    per cell, ``(1/K) sum_k |x_k - y| - (1/(K(K-1))) sum_{i<j} |x_i - x_j|``,
                  averaged over the region's cells and the batch. Credits marginal spread
                  cell by cell, which pixel-scale speckle also earns.
          energy  per map, the same with the rms distance over the region's cells in place
                  of ``|.|``: ``(1/K) sum_k ||x_k - y|| - (1/(K(K-1))) sum_{i<j} ||x_i - x_j||``.
                  The energy score with beta = 1 over the whole field, its norm divided by
                  the number of cells so it sits on the L1 term's scale and
                  ``image_distance_weight`` keeps its meaning. It pays most for spread that
                  is COHERENT over the region: at the same per-cell size, coherent spread
                  scores better than pixel speckle, which the per-cell CRPS cannot tell
                  apart. Coherent spread is what the region-mean quantities of the paper's
                  Section 6 need and what the shipped ensemble lacks.

        With "l1" or "mse" the samples are scored one by one and averaged; the spread term
        is then 0.

        Args:
        ----
            samples: The generated maps, (K, batch_size, channels, H, W).
            target_maps: The observed maps, (batch_size, channels, H, W).
            mask: The region mask, (batch_size, 1, H, W).

        Returns:
        -------
            The score (a scalar, lower is better) and the spread term it subtracted, for
            the logs.

        """
        k = int(samples.shape[0])
        if k < 2:
            raise ValueError("ensemble_pixel_distance needs at least 2 samples")
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        if self.pixel_distance_kind == "crps":
            skill = (samples - target_maps.unsqueeze(0)).abs().mean(dim=0)
            pair_sum = torch.zeros_like(skill)
            for i, j in pairs:
                pair_sum = pair_sum + (samples[i] - samples[j]).abs()
            spread = pair_sum / (k * (k - 1))
            score = masked_mean(skill - spread, mask, dims=(1, 2, 3)).mean()
            return score, masked_mean(spread, mask, dims=(1, 2, 3)).mean()
        if self.pixel_distance_kind == "energy":

            def rms(diff: torch.Tensor) -> torch.Tensor:
                return torch.sqrt(masked_mean(diff**2, mask, dims=(1, 2, 3)) + 1e-8)

            skill = torch.stack([rms(samples[i] - target_maps) for i in range(k)]).mean(dim=0)
            spread = torch.stack([rms(samples[i] - samples[j]) for i, j in pairs]).sum(dim=0) / (
                k * (k - 1)
            )
            return (skill - spread).mean(), spread.mean()
        distances = torch.stack(
            [self.pixel_distance(samples[i], target_maps, mask) for i in range(k)]
        )
        return distances.mean(), torch.zeros((), device=samples.device)

    def generate_samples(
        self,
        input_maps: torch.Tensor,
        input_timestamps: torch.Tensor,
        mask: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """``ensemble_size`` generator samples of the batch, stacked as (K, batch_size, ...).

        Sample 0 uses the ``z`` handed in, so with ``ensemble_size`` 1 this is ``forward``
        unchanged; the others draw their own bottleneck vector, and every one draws its own
        injection noise inside the forward, so the K samples differ exactly as inference
        members do. They run as one generator batch of K x batch_size: BatchNorm then sees
        the same batch_size inputs, each counted K times, so its statistics are unchanged.
        """
        k = int(self.hparams.ensemble_size)
        if k == 1:
            return self(
                input_maps=input_maps, input_timestamps=input_timestamps, mask=mask, noise_vector=z
            ).unsqueeze(0)
        batch_size = input_maps.shape[0]
        extra = torch.randn(
            (k - 1) * batch_size, self.hparams.z_dim, device=z.device, dtype=z.dtype
        )
        fake = self(
            input_maps=input_maps.repeat(k, 1, 1, 1),
            input_timestamps=input_timestamps.repeat(k),
            mask=mask.repeat(k, 1, 1, 1) if mask.shape[0] == batch_size else mask,
            noise_vector=torch.cat([z, extra], dim=0),
        )
        return fake.view(k, batch_size, *fake.shape[1:])

    def gradient_penalty(
        self,
        real_maps: torch.Tensor,
        fake_maps: torch.Tensor,
        mask: torch.Tensor,
        critic_type: str,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """Gradient penalty to apply to the Wasserstein loss, restricted to the region.

        Real and generated maps are both zero outside the mask, so every interpolate lives
        in the subspace of maps that vanish there. The Lipschitz constraint the
        Kantorovich-Rubinstein duality needs is on that subspace, and the norm of the
        gradient projected onto it -- ``gradients * mask`` -- is the right quantity. The
        unmasked norm spends the "equals one" budget on directions the data never moves
        in: 45% of it on the Grand Est grid, 96% for Corse on a canvas.

        Args:
        ----
            real_maps: The target maps, of shape (batch_size, channels, H, W).
            fake_maps: The generated maps, same shape.
            mask: The region mask the interpolates live in, of shape (batch_size, 1, H, W).
                When real and fake were augmented separately this is the union of their two
                masks.
            critic_type: Which head to penalize, "patch" or "frame".
            alpha: The interpolation factor, of shape (batch_size, 1, 1, 1).

        Returns:
        -------
            The gradient penalty.

        """
        batch_size = real_maps.shape[0]

        interpolated = alpha * real_maps + (1 - alpha) * fake_maps
        interpolated = (
            interpolated.clone().detach().requires_grad_(True)
        )  # Make it a leaf tensor with grad
        levels = mask_pyramid(mask, len(self.base_critic.blocks))
        base_output, _ = self.base_critic(interpolated, mask)
        if critic_type == "patch":
            d_interpolated, _ = self.patch_critic(base_output, weight=levels[-1])
            lambda_penalty = self.hparams.lambda_penalty_patch
        elif critic_type == "frame":
            d_interpolated, _ = self.frame_critic(base_output, weight=levels[-1])
            lambda_penalty = self.hparams.lambda_penalty_frame
        else:
            raise RuntimeError(
                f"Unknown critic type: {critic_type}. Please provide one of 'patch', 'frame'"
            )
        # Reduce to a scalar for autograd. "mean" differentiates the same functional the
        # loss consumes -- the coverage-weighted tile mean -- so lambda_penalty means the
        # same thing for both heads. "sum" adds up the valid patch tiles but only the single
        # frame output, so it constrains the patch critic more tightly than the loss does.
        # The frame head and "learned" aggregation emit one scalar, on which both agree.
        if self.hparams.gradient_penalty_reduction == "mean" or d_interpolated.dim() != 4:
            scalar_output = self.patch_scalar(d_interpolated, levels[-1]).sum()
        else:
            scalar_output = (d_interpolated * (levels[-1] > 0)).sum()
        gradients = autograd.grad(
            outputs=scalar_output,
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # Project onto the region before taking the norm.
        gradients = (gradients * mask).view(batch_size, -1)
        gradient_penalty = lambda_penalty * ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
        return gradient_penalty

    def compute_feature_loss(
        self,
        features_fake: list[torch.Tensor],
        features_real: list[torch.Tensor],
        weights: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Squared distance between critic features of real and fake maps.

        With ``weights`` -- one coverage map per feature level -- the distance is averaged
        over the cells with data under them, so it does not shrink with the padding
        fraction. Without them it is the plain MSE over the whole feature map.

        Args:
        ----
            features_fake: The critic features of the generated maps, one per level.
            features_real: The critic features of the observed maps, same levels.
            weights: The coverage weights at each level, or None for the unmasked MSE.

        Returns:
        -------
            The summed distance, a scalar.

        """
        loss = torch.zeros((), device=features_fake[0].device)
        level_weights: list[torch.Tensor | None] = (
            list(weights) if weights is not None else [None] * len(features_fake)
        )
        for feature_fake, feature_real, weight in zip(
            features_fake, features_real, level_weights, strict=True
        ):
            if weight is None:
                loss = loss + F.mse_loss(feature_fake, feature_real)
            else:
                loss = (
                    loss
                    + masked_mean((feature_fake - feature_real) ** 2, weight, dims=(1, 2, 3)).mean()
                )
        return loss

    def generator_step(
        self, batch: dict[str, torch.Tensor], z: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run a single generator step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            z: Noise vector. Of shape (batch_size, input_channels).

        Returns:
        -------
            The generator loss and the pixel distance loss.

        """
        (input_maps, target_maps, input_timestamps, mask) = (
            batch["input_maps"],
            batch["target_maps"],
            batch["timestamps"],
            batch["mask"],
        )

        samples = self.generate_samples(input_maps, input_timestamps, mask, z)
        # The adversarial, feature-matching and diagnostic terms stay on one sample; only the
        # pixel term below sees the mini-ensemble.
        fake_maps = samples[0]

        # Where the critics score: the native grid, or the canvas the generator padded to.
        critic_fake_maps, critic_mask = self.to_critic_canvas(fake_maps, mask)
        critic_target_maps, _ = self.to_critic_canvas(target_maps, mask)
        if self.training:
            # Augment the maps the critics see, carrying the mask through the same
            # translation and cutout so each critic pass knows where its data is.
            augmented_fake_maps, mask_fake = diff_augment_with_mask(
                critic_fake_maps,
                critic_mask,
                DIFF_AUGMENT_POLICY,
                self.hparams.diff_augment_on_region,
            )
            augmented_target_maps, mask_real = diff_augment_with_mask(
                critic_target_maps,
                critic_mask,
                DIFF_AUGMENT_POLICY,
                self.hparams.diff_augment_on_region,
            )
        else:
            augmented_fake_maps, mask_fake = critic_fake_maps, critic_mask
            augmented_target_maps, mask_real = critic_target_maps, critic_mask

        fake = self.critic_forward(augmented_fake_maps, mask_fake)
        real = self.critic_forward(augmented_target_maps, mask_real)

        # Pixel term, over the region's cells only: the single-sample distance, or with a
        # mini-ensemble the proper score that is the one term crediting spread.
        if samples.shape[0] > 1:
            pixel_score, pixel_spread = self.ensemble_pixel_distance(samples, target_maps, mask)
        else:
            pixel_score = self.pixel_distance(fake_maps, target_maps, mask)
            pixel_spread = torch.zeros((), device=pixel_score.device)
        pixel_distance_loss = self.hparams.image_distance_weight * pixel_score

        # Feature loss. Real and fake were translated independently, so a cell can hold
        # data under one and padding under the other; the union of the two coverages
        # counts it.
        num_blocks = len(self.base_critic.blocks)
        union = [
            torch.maximum(weight_fake, weight_real)
            for weight_fake, weight_real in zip(fake["levels"], real["levels"], strict=True)
        ]
        feature_loss = self.hparams.feature_matching_weight * (
            self.compute_feature_loss(
                fake["features_base"], real["features_base"], union[1 : num_blocks + 1]
            )
            + self.compute_feature_loss(
                fake["features_patch"],
                real["features_patch"],
                [union[-1]] * len(fake["features_patch"]),
            )
            + self.compute_feature_loss(
                fake["features_frame"],
                real["features_frame"],
                [union[-1]] * len(fake["features_frame"]),
            )
        )

        smape = self._compute_smape(target_maps.detach(), fake_maps.detach(), mask.detach())
        rmse = self._compute_rmse(target_maps.detach(), fake_maps.detach(), mask.detach())

        # The four additive terms are returned separately as well as summed, so
        # ``utils.generator_gradient`` can differentiate each one without re-deriving
        # this forward path.
        patch_term = -self.patch_scalar(fake["patch"], fake["levels"][-1]).mean()
        frame_term = -torch.mean(fake["frame"])
        loss_generator = patch_term + frame_term + pixel_distance_loss + feature_loss

        return {
            "loss_generator": loss_generator,
            "patch_term": patch_term,
            "frame_term": frame_term,
            "pixel_distance_loss": pixel_distance_loss,
            "pixel_spread": pixel_spread,
            "smape": smape,
            "rmse": rmse,
            "feature_loss": feature_loss,
        }

    def critic_step(
        self, batch: dict[str, torch.Tensor], z: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Run a single critic step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            z: Noise vector. Of shape (batch_size, input_channels).

        Returns:
        -------
            The Wasserstein loss.

        """
        input_maps, target_maps, input_timestamps, mask = (
            batch["input_maps"],
            batch["target_maps"],
            batch["timestamps"],
            batch["mask"],
        )
        device = input_maps.device
        # Generate fake images
        fake_imgs = self(
            input_maps=input_maps,
            input_timestamps=input_timestamps,
            mask=mask,
            noise_vector=z,
        )
        fake_imgs = fake_imgs.detach()

        # Where the critics score: the native grid, or the canvas the generator padded to.
        fake_imgs, critic_mask = self.to_critic_canvas(fake_imgs, mask)
        target_maps, _ = self.to_critic_canvas(target_maps, mask)
        mask = critic_mask

        if self.training:
            # Augment the maps the critics see, carrying the mask through the same
            # translation and cutout so each critic pass knows where its data is.
            fake_imgs, mask_fake = diff_augment_with_mask(
                fake_imgs, mask, DIFF_AUGMENT_POLICY, self.hparams.diff_augment_on_region
            )
            target_maps, mask_real = diff_augment_with_mask(
                target_maps, mask, DIFF_AUGMENT_POLICY, self.hparams.diff_augment_on_region
            )
        else:
            mask_fake, mask_real = mask, mask

        real = self.critic_forward(target_maps, mask_real)
        fake = self.critic_forward(fake_imgs, mask_fake)
        weight_real, weight_fake = real["levels"][-1], fake["levels"][-1]
        patch_critic_real = self.patch_scalar(real["patch"], weight_real)
        patch_critic_fake = self.patch_scalar(fake["patch"], weight_fake)
        frame_critic_real, frame_critic_fake = real["frame"], fake["frame"]

        # Compute the losses for each discriminator and aggregate them. Every reduction over
        # the tile grid is weighted by coverage, so padding tiles do not enter the loss.
        if self.hparams.patch_critic_loss == "hinge":
            loss_critic_patch = (
                self.patch_scalar(F.relu(1.0 - real["patch"]), weight_real).mean()
                + self.patch_scalar(F.relu(1.0 + fake["patch"]), weight_fake).mean()
            )
        else:
            loss_critic_patch = torch.mean(patch_critic_fake) - torch.mean(patch_critic_real)
        loss_critic_frame = torch.mean(frame_critic_fake) - torch.mean(frame_critic_real)
        total_loss_critic = loss_critic_patch + loss_critic_frame

        # Scale-free separation of the patch grid, logged so the curve stays comparable
        # across runs even though the hinge loss is on a different scale to Eq. 3.5.
        patch_separation = torch.mean(patch_critic_real) - torch.mean(patch_critic_fake)
        # Fraction of scores still inside +/-1, over the tiles with data under them. Under
        # "hinge" this is the fraction still receiving gradient; under "wasserstein" it is
        # a readout of the critic's output scale, which is what the gradient penalty is
        # there to anchor.
        valid_real = self.valid_patch_scores(real["patch"], weight_real)
        valid_fake = self.valid_patch_scores(fake["patch"], weight_fake)
        patch_margin_active = 0.5 * (
            (valid_real < 1.0).float().mean() + (valid_fake > -1.0).float().mean()
        )

        # Output scale, logged directly so a change in lambda_penalty or in the
        # aggregation can be read off the curves instead of rescoring checkpoints.
        patch_scores = torch.cat([valid_real, valid_fake])
        frame_scores = torch.cat([frame_critic_real.flatten(), frame_critic_fake.flatten()])
        patch_score_sd = patch_scores.std()
        frame_score_sd = frame_scores.std()
        patch_score_absmax = patch_scores.abs().max()
        frame_score_absmax = frame_scores.abs().max()

        if self.training:
            # Add the gradient penalty. Real and fake were translated independently, so the
            # interpolates live in the union of the two masks.
            batch_size = target_maps.shape[0]
            alpha = torch.rand(batch_size, 1, 1, 1, device=device)
            mask_union = torch.maximum(mask_real, mask_fake)
            gradient_penalty = self.gradient_penalty(
                target_maps,
                fake_imgs,
                mask_union,
                critic_type="patch",
                alpha=alpha,
            )
            gradient_penalty += self.gradient_penalty(
                target_maps,
                fake_imgs,
                mask_union,
                critic_type="frame",
                alpha=alpha,
            )
            total_loss_critic += gradient_penalty

        return {
            "total_loss_critic": total_loss_critic,
            "loss_patch_critic": loss_critic_patch,
            "loss_frame_critic": loss_critic_frame,
            "patch_separation": patch_separation,
            "patch_margin_active": patch_margin_active,
            "patch_score_sd": patch_score_sd,
            "frame_score_sd": frame_score_sd,
            "patch_score_absmax": patch_score_absmax,
            "frame_score_absmax": frame_score_absmax,
            "gradient_penalty": gradient_penalty
            if self.training
            else torch.tensor(0.0, device=total_loss_critic.device),
        }

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run a single training step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            batch_idx: Index of the batch.

        Returns:
        -------
            The generator loss.

        """
        batch_size = batch["target_maps"].shape[0]
        opt_generator, opt_critic = self.optimizers()
        z = torch.randn(batch_size, self.hparams.z_dim, device=batch["target_maps"].device)

        # Train Critic multiple times
        for _ in range(self.hparams.num_critic_iterations_per_epoch):
            opt_critic.zero_grad()
            critic_outputs = self.critic_step(batch, z)
            (
                loss_critic,
                loss_patch_critic,
                loss_frame_critic,
                gradient_penalty,
                patch_separation,
                patch_margin_active,
            ) = (
                critic_outputs["total_loss_critic"],
                critic_outputs["loss_patch_critic"],
                critic_outputs["loss_frame_critic"],
                critic_outputs["gradient_penalty"],
                critic_outputs["patch_separation"],
                critic_outputs["patch_margin_active"],
            )
            self.manual_backward(loss_critic)
            opt_critic.step()

        # Train Generator
        opt_generator.zero_grad()
        generator_outputs = self.generator_step(batch, z)
        (loss_generator, pixel_distance_loss, smape, rmse, feature_loss) = (
            generator_outputs["loss_generator"],
            generator_outputs["pixel_distance_loss"],
            generator_outputs["smape"],
            generator_outputs["rmse"],
            generator_outputs["feature_loss"],
        )
        self.manual_backward(loss_generator)
        opt_generator.step()

        # Logging
        self.log_dict(
            {
                "train/total_critic_loss": loss_critic,
                "train/critic_patch_loss": loss_patch_critic,
                "train/critic_frame_loss": loss_frame_critic,
                "train/critic_patch_separation": patch_separation,
                "train/critic_patch_margin_active": patch_margin_active,
                "train/critic_patch_score_sd": critic_outputs["patch_score_sd"],
                "train/critic_frame_score_sd": critic_outputs["frame_score_sd"],
                "train/critic_patch_score_absmax": critic_outputs["patch_score_absmax"],
                "train/critic_frame_score_absmax": critic_outputs["frame_score_absmax"],
                "train/gradient_penalty": gradient_penalty,
                "train/generator_loss": loss_generator,
                "train/generator_pixel_distance_loss": pixel_distance_loss
                / self.hparams.image_distance_weight,
                # The spread term of the proper score (0 under l1 / mse): what the
                # mini-ensemble is being paid for disagreeing.
                "train/generator_pixel_spread": generator_outputs["pixel_spread"],
                "train/feature_loss": feature_loss,
                "train/generator_smape": smape,
                "train/generator_rmse": rmse,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        return loss_generator

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run a single validation step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            batch_idx: Index of the batch.

        Returns:
        -------
            The generator loss.

        """
        batch_size = batch["input_maps"].shape[0]
        z = torch.randn(batch_size, self.hparams.z_dim, device=batch["input_maps"].device)
        critic_outputs = self.critic_step(batch, z)
        loss_critic, loss_patch_critic, loss_frame_critic, patch_separation = (
            critic_outputs["total_loss_critic"],
            critic_outputs["loss_patch_critic"],
            critic_outputs["loss_frame_critic"],
            critic_outputs["patch_separation"],
        )
        generator_outputs = self.generator_step(batch, z)
        (loss_generator, pixel_distance_loss, smape, rmse, feature_loss) = (
            generator_outputs["loss_generator"],
            generator_outputs["pixel_distance_loss"],
            generator_outputs["smape"],
            generator_outputs["rmse"],
            generator_outputs["feature_loss"],
        )

        # Logging
        self.log_dict(
            {
                "val/total_critic_loss": loss_critic,
                "val/critic_patch_loss": loss_patch_critic,
                "val/critic_frame_loss": loss_frame_critic,
                "val/critic_patch_separation": patch_separation,
                "val/generator_loss": loss_generator,
                "val/generator_pixel_distance_loss": pixel_distance_loss
                / self.hparams.image_distance_weight,
                "val/generator_pixel_spread": generator_outputs["pixel_spread"],
                "val/feature_loss": feature_loss,
                "val/generator_smape": smape,
                "val/generator_rmse": rmse,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        return loss_generator

    def test_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Run a single test step.

        Args:
        ----
            batch: A dictionary containing the inputs, targets, and timestamps.
            batch_idx: Index of the batch.

        Returns:
        -------
            The generator loss.

        """
        batch_size = batch["input_maps"].shape[0]
        z = torch.randn(batch_size, self.hparams.z_dim, device=batch["input_maps"].device)
        critic_outputs = self.critic_step(batch, z)
        loss_critic, loss_patch_critic, loss_frame_critic, patch_separation = (
            critic_outputs["total_loss_critic"],
            critic_outputs["loss_patch_critic"],
            critic_outputs["loss_frame_critic"],
            critic_outputs["patch_separation"],
        )
        generator_outputs = self.generator_step(batch, z)
        (loss_generator, pixel_distance_loss, smape, rmse, feature_loss) = (
            generator_outputs["loss_generator"],
            generator_outputs["pixel_distance_loss"],
            generator_outputs["smape"],
            generator_outputs["rmse"],
            generator_outputs["feature_loss"],
        )
        # Logging
        self.log_dict(
            {
                "test/total_critic_loss": loss_critic,
                "test/critic_patch_loss": loss_patch_critic,
                "test/critic_frame_loss": loss_frame_critic,
                "test/critic_patch_separation": patch_separation,
                "test/generator_loss": loss_generator,
                "test/generator_pixel_distance_loss": pixel_distance_loss,
                "test/feature_loss": feature_loss,
                "test/generator_smape": smape,
                "test/generator_rmse": rmse,
            },
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        return loss_generator

    def forward(
        self,
        input_maps: torch.Tensor,
        input_timestamps: torch.Tensor,
        mask: torch.Tensor,
        noise_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the forward pass.

        Args:
        ----
            input_maps: The input maps as torch Tensors. Of shape
                (batch_size, input_length, input_channels, height, width)
            input_timestamps: The timesteps vectors corresponding to each input map.
                Must be of shape (batch_size, input_length, temporal_dim).
            output_timestamps: The timesteps vectors corresponding to outputs.
                Must be of shape (batch_size, output_length, temporal_dim).
            mask: The region mask of shape (batch_size, 1, height, width) or
                (1, 1, height, width).
            noise_vector: A noise vector for stochasticity in the predictions. If
                None a vector will be generated from the standard normal distribution.
                Must be of shape (batch_size, z_dim).

        Returns:
        -------
            The generated output maps of shape
            (batch_size, output_length, output_channels, height, width).


        """
        batch_size, channels, H, W = input_maps.shape  # noqa
        mask = mask.to(input_maps.dtype).expand(batch_size, -1, -1, -1)
        timesteps_channels = self.timestamps_embedding(input_timestamps.long())[..., None, None]
        timesteps_channels = timesteps_channels.expand(-1, -1, H, W).to(input_maps.device)
        channels_to_stack = [input_maps, timesteps_channels]
        if self.hparams.generator_mask_channel:
            channels_to_stack.append(mask)
        inputs = torch.cat(channels_to_stack, dim=1)
        left, right, top, bottom = _generator_padding(H, W)
        inputs = F.pad(inputs, (left, right, top, bottom))
        # The generator normalizes over the region's cells on the canvas it works on, so the
        # mask has to make the same trip as the maps -- same offsets, same canvas.
        generator_mask = (
            F.pad(mask, (left, right, top, bottom)) if self.hparams.generator_masked_norm else None
        )
        outputs = self.generator(
            inputs=inputs,
            noise_vector=noise_vector,
            mask=generator_mask,
        )
        # Crop with the offsets the padding used. torchvision's center_crop rounds the odd
        # pixel the other way, which shifted every odd-padded grid -- Corse's 23x11 among
        # them -- by one cell; the even padding of the 36x44 Grand Est grid hid it.
        outputs = outputs[..., top : top + H, left : left + W]
        return outputs * mask

    def _compute_smape(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean symmetric absolute percentage error."""
        targets_mean_value, targets_std_value = (
            self.input_statistics["targets_mean"],
            self.input_statistics["targets_std"],
        )
        mean_v, std_v = (
            torch.tensor(targets_mean_value).to(y_true.device),
            torch.tensor(targets_std_value).to(y_true.device),
        )

        y_true_p, y_pred_p = y_true * std_v + mean_v, y_pred * std_v + mean_v
        absolute_diff = torch.abs(y_true_p - y_pred_p)
        denominator = (torch.abs(y_true_p) + torch.abs(y_pred_p)) / 2

        mape = absolute_diff / (denominator + 1e-8)
        masked_mape = mape * mask
        masked_mape = masked_mape.sum(dim=(-2, -1)) / (mask.sum(dim=(-2, -1)) + 1e-8)
        masked_mape = masked_mape.mean()
        return 100 * masked_mape

    def _compute_rmse(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial RMSE per predictions timestep."""
        n_elements = mask.sum(dim=(-2, -1))
        targets_mean_value, targets_std_value = (
            self.input_statistics["targets_mean"],
            self.input_statistics["targets_std"],
        )
        mean_v, std_v = (
            torch.tensor(targets_mean_value).to(y_true.device),
            torch.tensor(targets_std_value).to(y_true.device),
        )

        y_true_p, y_pred_p = y_true * std_v + mean_v, y_pred * std_v + mean_v

        masked_y_pred = y_pred_p * mask
        masked_y_true = y_true_p * mask

        mean_rmse = ((masked_y_true - masked_y_pred) ** 2).sum(axis=(-2, -1)) / n_elements
        mean_rmse = torch.sqrt(mean_rmse.mean(dim=-1)).mean()

        return mean_rmse

    def on_validation_epoch_end(self) -> None:
        """To run at the end of the validation epoch."""
        scheduler_generator, scheduler_critic = self.lr_schedulers()
        scheduler_generator.step()
        scheduler_critic.step()

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[Any]]:
        """Configure the optimizers."""
        # The noise gains get their own param group so their learning rate can be set
        # independently of the convs'. At a shared rate they move ~1e-3 over a 120-epoch
        # run, which is far below the O(0.1) they would need to reach to change the output
        # -- so without this split the gains are fixed by their init rather than learned.
        named_generator_params = list(self.generator.named_parameters())
        noise_params = [
            param
            for name, param in named_generator_params
            if name.rsplit(".", 1)[-1] in NOISE_GAIN_ATTRIBUTES
        ]
        conv_params = [
            param
            for name, param in named_generator_params
            if name.rsplit(".", 1)[-1] not in NOISE_GAIN_ATTRIBUTES
        ]
        generator_param_groups: list[dict[str, Any]] = [{"params": conv_params}]
        if noise_params:
            generator_param_groups.append(
                {
                    "params": noise_params,
                    "lr": self.hparams.lr * self.hparams.noise_weight_lr_scale,
                }
            )
        optimizer_generator = self.hparams.optim(
            generator_param_groups,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.5, 0.999),
        )
        optimizer_critic = self.hparams.optim(
            list(self.base_critic.parameters())
            + list(self.patch_critic.parameters())
            + list(self.frame_critic.parameters()),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.5, 0.999),
        )

        scheduler_generator = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_generator,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.min_lr,
        )
        scheduler_critic = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_critic,
            T_max=self.hparams.max_epochs,
            eta_min=self.hparams.min_lr,
        )

        return [optimizer_generator, optimizer_critic], [scheduler_generator, scheduler_critic]
