"""The lightning module for the GAN."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F  # noqa
from lightning import LightningModule
from torch import autograd, nn
from torchvision.transforms import v2 as v2_transforms
from torchvision.transforms.functional import center_crop

from modules.diff_augment import DiffAugment
from modules.discriminator.base_discriminator import BaseDiscriminator
from modules.discriminator.frame_discriminator import FrameDiscriminator
from modules.discriminator.patch_gan_discriminator import PatchGANDiscriminator
from modules.generator.unet_generator import UNetGenerator
from modules.utils import NOISE_WEIGHT_INITS


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
        noise_weight_lr_scale: float = 1.0,
        encoder_late_dropout: float = 0.0,
    ) -> None:
        """Initialize the input arguments.

        Args:
        ----
            input_channels: Number of input channels of the input maps.
            output_channels: Number of output channels of the predicted maps.
            input_map_dims: A tuple (H, W) specifying the height and width of the
                input maps.
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

        """
        super().__init__()
        self.timestamps_embedding = nn.Embedding(12, timestamps_dim)
        self.generator = UNetGenerator(
            input_dim=input_channels + timestamps_dim,
            output_dim=output_channels,
            noise_dim=z_dim,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            dropout=spatial_dropout,
            normalization=normalization,
            apply_center_block=apply_center_block,
            noise_weight_init=noise_weight_init,
            encoder_late_dropout=encoder_late_dropout,
        )

        self.base_critic = BaseDiscriminator(
            input_channels=output_channels, output_channels=[32, 64, 128], mbd_output_channels=64
        )
        self.patch_critic = PatchGANDiscriminator(
            input_channels=128 + 64, aggregation=patch_aggregation
        )
        self.frame_critic = FrameDiscriminator(input_channels=128 + 64)

        self.transforms = v2_transforms.Compose(
            [
                v2_transforms.RandomHorizontalFlip(p=0.5),
                v2_transforms.RandomResizedCrop(input_map_dims, scale=(0.7, 1.0)),
            ]
        )

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

        # Resolved before save_hyperparameters so the checkpoint records the weight each
        # head actually ran with, not None.
        if lambda_penalty_patch is None:
            lambda_penalty_patch = lambda_penalty
        if lambda_penalty_frame is None:
            lambda_penalty_frame = lambda_penalty

        if isinstance(loss_fn, str):
            if loss_fn == "l1":
                self.loss_fn = nn.L1Loss(reduction="mean")
            elif loss_fn == "mse":
                self.loss_fn = nn.MSELoss(reduction="mean")
            else:
                raise NotImplementedError(f"Unknown loss function: {loss_fn}")
        else:
            self.loss_fn = loss_fn()

        self.automatic_optimization = False
        self.save_hyperparameters(ignore="loss_fn")

    def gradient_penalty(
        self,
        real_maps: torch.Tensor,
        fake_maps: torch.Tensor,
        critic_type: str,
        alpha: float,
    ) -> torch.Tensor:
        """Gradient penalty to apply to the Wasserstein loss.

        Args:
        ----
            real_maps: The target maps. A tensor of shape
                (batch_size, output_length, output_channels, H, W).
            fake_maps: The maps generated by the generator.
                Of shape (batch_size, output_length, output_channels, H, W).
            timesteps: The output timestamps vectors. Of shape
                (batch_size, output_length, temporal_dims).
            critic_type: Whether the gradient penalty is applied to the
                patchGAN critic or the frame critic or the temporal critic.
                Must be one of "patch", "frame", "temporal"
            alpha: The interpolation factor.

        Returns:
        -------
            The gradient penalty.

        """
        batch_size = real_maps.shape[0]

        interpolated = alpha * real_maps + (1 - alpha) * fake_maps
        interpolated = (
            interpolated.clone().detach().requires_grad_(True)
        )  # Make it a leaf tensor with grad
        base_output, _ = self.base_critic(interpolated)
        if critic_type == "patch":
            d_interpolated, _ = self.patch_critic(base_output)
            lambda_penalty = self.hparams.lambda_penalty_patch
        elif critic_type == "frame":
            d_interpolated, _ = self.frame_critic(base_output)
            lambda_penalty = self.hparams.lambda_penalty_frame
        else:
            raise RuntimeError(
                f"Unknown critic type: {critic_type}. " f"Please provide one of 'patch', 'frame'"
            )
        d_interpolated = d_interpolated.requires_grad_(True)
        # Reduce to a scalar for autograd. "sum" adds up the 20 patch outputs but only the
        # single frame output, so it constrains the patch critic ~20x more tightly than the
        # loss that consumes it -- which keeps the patch scores too small to ever reach a
        # hinge margin. "mean" differentiates the same functional the loss uses, so
        # lambda_penalty means the same thing for both heads. Under
        # patch_aggregation="learned" the patch critic already emits one scalar, so the
        # two branches agree and this setting only affects "mean" aggregation.
        if self.hparams.gradient_penalty_reduction == "mean":
            scalar_output = d_interpolated.flatten(1).mean(dim=1).sum()
        else:
            scalar_output = d_interpolated.sum()
        gradients = autograd.grad(
            outputs=scalar_output,
            inputs=interpolated,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = lambda_penalty * ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
        return gradient_penalty

    def compute_feature_loss(
        self, features_fake: list[torch.Tensor], features_real: list[torch.Tensor]
    ) -> torch.Tensor:
        """Compute mse loss between discriminator features of real and fake maps."""
        loss = 0.0
        for feature_fake, feature_real in zip(features_fake, features_real, strict=True):
            loss += torch.nn.functional.mse_loss(feature_fake, feature_real)
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

        if self.training:
            input_maps, target_maps, mask = self.transforms(input_maps, target_maps, mask)

        fake_maps = self(
            input_maps=input_maps,
            input_timestamps=input_timestamps,
            mask=mask,
            noise_vector=z,
        )

        if self.training:
            # Augment images
            augmented_fake_maps = DiffAugment(fake_maps.contiguous(), policy="translation,cutout")
            augmented_target_maps = DiffAugment(
                target_maps.contiguous(), policy="translation,cutout"
            )
        else:
            augmented_fake_maps = fake_maps
            augmented_target_maps = target_maps

        base_critic_fake, features_base_fake = self.base_critic(augmented_fake_maps)
        patch_critic_fake, features_patch_fake = self.patch_critic(base_critic_fake)
        frame_critic_fake, features_frame_fake = self.frame_critic(base_critic_fake)

        base_critic_real, features_base_real = self.base_critic(augmented_target_maps)
        _, features_patch_real = self.patch_critic(base_critic_real)
        _, features_frame_real = self.frame_critic(base_critic_real)

        # Pixel distance loss
        pixel_distance_loss = self.hparams.image_distance_weight * self.loss_fn(
            fake_maps, target_maps
        )

        # Feature loss
        feature_loss = self.hparams.feature_matching_weight * (
            self.compute_feature_loss(features_base_fake, features_base_real)
            + self.compute_feature_loss(features_patch_fake, features_patch_real)
            + self.compute_feature_loss(features_frame_fake, features_frame_real)
        )

        smape = self._compute_smape(target_maps.detach(), fake_maps.detach(), mask.detach())
        rmse = self._compute_rmse(target_maps.detach(), fake_maps.detach(), mask.detach())

        loss_generator = (
            -torch.mean(patch_critic_fake)
            - torch.mean(frame_critic_fake)
            + pixel_distance_loss
            + feature_loss
        )

        return {
            "loss_generator": loss_generator,
            "pixel_distance_loss": pixel_distance_loss,
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
        if self.training:
            input_maps, target_maps, mask = self.transforms(input_maps, target_maps, mask)

        fake_imgs = self(
            input_maps=input_maps,
            input_timestamps=input_timestamps,
            mask=mask,
            noise_vector=z,
        )
        fake_imgs = fake_imgs.detach()

        if self.training:
            # Augment images
            fake_imgs = DiffAugment(fake_imgs.contiguous(), policy="translation,cutout")
            target_maps = DiffAugment(target_maps.contiguous(), policy="translation,cutout")

        base_critic_real, _ = self.base_critic(target_maps)
        base_critic_fake, _ = self.base_critic(fake_imgs)

        patch_critic_real, _ = self.patch_critic(base_critic_real)
        patch_critic_fake, _ = self.patch_critic(base_critic_fake)
        frame_critic_real, _ = self.frame_critic(base_critic_real)
        frame_critic_fake, _ = self.frame_critic(base_critic_fake)

        # Compute the losses for each discriminator and aggregate them
        if self.hparams.patch_critic_loss == "hinge":
            loss_critic_patch = (
                F.relu(1.0 - patch_critic_real).mean() + F.relu(1.0 + patch_critic_fake).mean()
            )
        else:
            loss_critic_patch = torch.mean(patch_critic_fake) - torch.mean(patch_critic_real)
        loss_critic_frame = torch.mean(frame_critic_fake) - torch.mean(frame_critic_real)
        total_loss_critic = loss_critic_patch + loss_critic_frame

        # Scale-free separation of the patch grid, logged so the curve stays comparable
        # across runs even though the hinge loss is on a different scale to Eq. 3.5.
        patch_separation = torch.mean(patch_critic_real) - torch.mean(patch_critic_fake)
        # Fraction of scores still inside +/-1. Under "hinge" this is the fraction still
        # receiving gradient; under "wasserstein" it is a readout of the critic's output
        # scale, which is what the gradient penalty is there to anchor.
        patch_margin_active = 0.5 * (
            (patch_critic_real < 1.0).float().mean() + (patch_critic_fake > -1.0).float().mean()
        )

        # Output scale, logged directly so a change in lambda_penalty or in the
        # aggregation can be read off the curves instead of rescoring checkpoints.
        patch_scores = torch.cat([patch_critic_real.flatten(), patch_critic_fake.flatten()])
        frame_scores = torch.cat([frame_critic_real.flatten(), frame_critic_fake.flatten()])
        patch_score_sd = patch_scores.std()
        frame_score_sd = frame_scores.std()
        patch_score_absmax = patch_scores.abs().max()
        frame_score_absmax = frame_scores.abs().max()

        if self.training:
            # add gradient penalty
            batch_size = target_maps.shape[0]
            alpha = torch.rand(batch_size, 1, 1, 1, device=device)
            gradient_penalty = self.gradient_penalty(
                target_maps,
                fake_imgs,
                critic_type="patch",
                alpha=alpha,  # noqa
            )
            gradient_penalty += self.gradient_penalty(
                target_maps,
                fake_imgs,
                critic_type="frame",
                alpha=alpha,  # noqa
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
            mask: The mask of shape (1, 1, 1, height, width).
            noise_vector: A noise vector for stochasticity in the predictions. If
                None a vector will be generated from the standard normal distribution.
                Must be of shape (batch_size, z_dim).

        Returns:
        -------
            The generated output maps of shape
            (batch_size, output_length, output_channels, height, width).


        """
        batch_size, channels, H, W = input_maps.shape  # noqa
        timesteps_channels = self.timestamps_embedding(input_timestamps.long())[..., None, None]
        timesteps_channels = timesteps_channels.expand(-1, -1, H, W).to(input_maps.device)
        inputs = torch.cat([input_maps, timesteps_channels], dim=1)
        inputs = F.pad(inputs, (2, 2, 6, 5))
        outputs = self.generator(
            inputs=inputs,
            noise_vector=noise_vector,
        )
        outputs = center_crop(outputs, self.hparams.input_map_dims)
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
        noise_params = [p for name, p in named_generator_params if "noise_weights" in name]
        conv_params = [p for name, p in named_generator_params if "noise_weights" not in name]
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
