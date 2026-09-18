"""The dataset module for training the GAN."""

import numpy as np
import torch
from torch.utils.data import Dataset


class SWIDataset(Dataset):
    """The dataset for the training the SWIGAN UNet variant.

    Concatenates the input_maps to the target_values at each input step.
    """

    def __init__(
        self,
        maps_feats: np.ndarray,
        target_maps: np.ndarray,
        mask: np.ndarray,
        timestamps: np.ndarray,
        num_input_steps: int,
        history_dropout_prob: float = 0.0,
        history_dropout_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        """Dataset for the maps.

        Args:
        ----
            maps_feats: The input maps.
            target_maps: The maps to predict.
            mask: A 2D boolean mask representing the region of interest.
                Of shape (1, H, W).
            timestamps: The time vectors corresponding to each map.
            num_input_steps: Number of input timesteps for the data generator.
            history_dropout_prob: Share of samples whose ``num_input_steps`` SWI history
                frames are scaled by one random factor before being handed out. 0.0 is
                every round so far. WHY: on Corse the history vetoes the covariates at a
                regime break -- October 2024's +1.1 sigma rainfall reached the model and a
                bone-dry eight-month history overrode it, while the same month with a
                neutral history landed on the observation. Showing the model a damped
                history against the true target teaches it how much to weigh the covariates
                when the state is uncertain, which is what the neutral-history members of
                ``utils/dispersion.py`` patch at inference. Training split only.
            history_dropout_range: The factor is drawn uniformly from this interval; 0 is
                the climatologically neutral history (the training mean, in standardised
                units), 1 leaves the sample untouched.

        """
        self.num_input_steps = num_input_steps
        self.history_dropout_prob = float(history_dropout_prob)
        low, high = (float(v) for v in history_dropout_range)
        if not 0.0 <= self.history_dropout_prob <= 1.0:
            raise ValueError(f"history_dropout_prob must be in [0, 1], got {history_dropout_prob}")
        if not 0.0 <= low <= high <= 1.0:
            raise ValueError(
                "history_dropout_range must satisfy 0 <= low <= high <= 1, "
                f"got {history_dropout_range}"
            )
        self.history_dropout_range = (low, high)
        self.mask = torch.tensor(mask).float()
        self.samples = self._build_samples(
            torch.tensor(maps_feats) * self.mask[None, ...],
            torch.tensor(target_maps) * self.mask[None, ...],
            torch.tensor(timestamps.squeeze()).long(),
        )

    def __len__(self) -> int:
        """Size of the dataset.

        Since a sequence of 'num_input_maps' is sent to the model,
        """
        return len(self.samples)

    def _build_samples(
        self, input_maps: torch.Tensor, target_maps: torch.Tensor, timestamps: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Build samples from the dataset."""
        samples = [
            {
                "input_maps": torch.concat(
                    [
                        input_maps[i],
                        target_maps[i - self.num_input_steps : i, 0],
                    ],
                    dim=0,
                ).float(),
                "target_maps": target_maps[i].float(),
                "timestamps": timestamps[i],
            }
            for i in range(self.num_input_steps, len(input_maps))
        ]
        return samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        input_maps = self.samples[idx]["input_maps"]
        # torch's RNG rather than numpy's: DataLoader re-seeds it per worker, so forked
        # workers do not replay one another's draws.
        if self.history_dropout_prob > 0 and float(torch.rand(())) < self.history_dropout_prob:
            low, high = self.history_dropout_range
            factor = low + (high - low) * float(torch.rand(()))
            input_maps = input_maps.clone()
            input_maps[-self.num_input_steps :] *= factor
        return {
            "input_maps": input_maps,
            "timestamps": self.samples[idx]["timestamps"],
            "target_maps": self.samples[idx]["target_maps"],
            "mask": self.mask,
        }


def build_train_val_test_datasets(
    input_maps: np.ndarray,
    target_maps: np.ndarray,
    timesteps: np.ndarray,
    mask: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    num_input_steps: int,
    history_dropout_prob: float = 0.0,
    history_dropout_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict[str, SWIDataset], dict[str, np.ndarray]]:
    """Build the train validation and test datasets.

    Args:
    ----
        input_maps: The input rasters.
        target_maps: The output rasters with the target variable as channels.
        timesteps: The time vectors containing the indices of the months.
        mask: A boolean mask representing the region of interest.
        train_ratio: Ratio of training split.
        val_ratio: Ratio of the validation split.
        num_input_steps: Number of timesteps to consider as input.
            Used for training, validation and test datasets.
        num_output_steps_train: Number of output timesteps in the training dataset.
        num_output_steps_val: Number of output timesteps in the validation and test dataset.
        history_dropout_prob: See :class:`SWIDataset`. Applied to the training split only.
        history_dropout_range: See :class:`SWIDataset`.

    Returns:
    -------
        Two dictionaries:
            The first one containing Train, validation and test datasets, with keys
            "train", "val", "test.
            The second one containing standardization metrics for the feature maps and
            the target maps.

    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            "Please provide 'train_ratio' and 'val_ratio' such that "
            "train_ratio + val_ratio < 1.0."
        )
    train_length = int(train_ratio * len(input_maps))
    val_length = int(val_ratio * len(input_maps))

    (
        maps_train,
        maps_val,
        maps_test,
        targets_train,
        targets_val,
        targets_test,
        timestamps_train,
        timestamps_val,
        timestamps_test,
    ) = (
        input_maps[:train_length],
        input_maps[train_length : train_length + val_length],
        input_maps[train_length + val_length :],
        target_maps[:train_length],
        target_maps[train_length : train_length + val_length],
        target_maps[train_length + val_length :],
        timesteps[:train_length],
        timesteps[train_length : train_length + val_length],
        timesteps[train_length + val_length :],
    )

    # Standardize train data
    # First let's remove the mask
    maps_train_without_mask = np.where(mask.squeeze(), maps_train, np.nan)
    maps_val_without_mask = np.where(mask.squeeze(), maps_val, np.nan)
    maps_test_without_mask = np.where(mask.squeeze(), maps_test, np.nan)

    mean_value, std_value = (
        np.nanmean(maps_train_without_mask, axis=(0, -2, -1), keepdims=True),
        np.nanstd(maps_train_without_mask, axis=(0, -2, -1), keepdims=True),
    )
    maps_train = (maps_train_without_mask - mean_value) / std_value
    maps_val = (maps_val_without_mask - mean_value) / std_value
    maps_test = (maps_test_without_mask - mean_value) / std_value

    maps_train = np.where(mask.squeeze(), maps_train, 0.0)
    maps_val = np.where(mask.squeeze(), maps_val, 0.0)
    maps_test = np.where(mask.squeeze(), maps_test, 0.0)

    # # Standardize target data
    targets_train_without_mask = np.where(mask.squeeze(), targets_train, np.nan)
    targets_val_without_mask = np.where(mask.squeeze(), targets_val, np.nan)
    targets_test_without_mask = np.where(mask.squeeze(), targets_test, np.nan)

    targets_mean_value, targets_std_value = (
        np.nanmean(targets_train_without_mask, axis=(0, -2, -1), keepdims=True),
        np.nanstd(targets_train_without_mask, axis=(0, -2, -1), keepdims=True),
    )
    targets_train = (targets_train_without_mask - targets_mean_value) / targets_std_value
    targets_val = (targets_val_without_mask - targets_mean_value) / targets_std_value
    targets_test = (targets_test_without_mask - targets_mean_value) / targets_std_value

    targets_train = np.where(mask.squeeze(), targets_train, 0.0)
    targets_val = np.where(mask.squeeze(), targets_val, 0.0)
    targets_test = np.where(mask.squeeze(), targets_test, 0.0)

    train_dataset = SWIDataset(
        maps_feats=maps_train,
        target_maps=targets_train,
        timestamps=timestamps_train,
        mask=mask,
        num_input_steps=num_input_steps,
        history_dropout_prob=history_dropout_prob,
        history_dropout_range=history_dropout_range,
    )
    val_dataset = SWIDataset(
        maps_feats=maps_val,
        target_maps=targets_val,
        timestamps=timestamps_val,
        mask=mask,
        num_input_steps=num_input_steps,
    )
    test_dataset = SWIDataset(
        maps_feats=maps_test,
        target_maps=targets_test,
        timestamps=timestamps_test,
        mask=mask,
        num_input_steps=num_input_steps,
    )

    return (
        {"train": train_dataset, "val": val_dataset, "test": test_dataset},
        {
            "feats_mean": mean_value,
            "feats_std": std_value,
            "targets_mean": targets_mean_value,
            "targets_std": targets_std_value,
        },
    )


def build_datasets_from_split_rasters(
    split_maps: dict[str, np.ndarray],
    split_targets: dict[str, np.ndarray],
    split_timesteps: dict[str, np.ndarray],
    mask: np.ndarray,
    num_input_steps: int,
    history_dropout_prob: float = 0.0,
    history_dropout_range: tuple[float, float] = (0.0, 1.0),
) -> tuple[dict[str, SWIDataset], dict[str, np.ndarray]]:
    """Build datasets from rasters that are already split into train/val/test.

    Same standardization as :func:`build_train_val_test_datasets` -- per-channel mean and
    standard deviation taken over the *training* split only, with masked-out pixels excluded
    from the statistics and zeroed afterwards -- but the chronological split is taken from the
    caller instead of from ratios. Use it when the splits ship as separate files.

    Args:
    ----
        split_maps: The input rasters per split. Must contain a "train" key.
        split_targets: The target rasters per split, same keys as ``split_maps``.
        split_timesteps: The time vectors per split, same keys as ``split_maps``.
        mask: A boolean mask representing the region of interest, shared by every split.
        num_input_steps: Number of timesteps to consider as input.
        history_dropout_prob: See :class:`SWIDataset`. Applied to the "train" split only.
        history_dropout_range: See :class:`SWIDataset`.

    Returns:
    -------
        Two dictionaries: the datasets keyed by split name, and the standardization
        statistics of the training split.

    """
    if "train" not in split_maps:
        raise ValueError("'split_maps' must contain a 'train' key to standardize against.")
    if set(split_maps) != set(split_targets) or set(split_maps) != set(split_timesteps):
        raise ValueError(
            "'split_maps', 'split_targets' and 'split_timesteps' must share the same keys."
        )

    squeezed_mask = mask.squeeze()

    def unmasked(maps: np.ndarray) -> np.ndarray:
        return np.where(squeezed_mask, maps, np.nan)

    def standardize(maps: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return np.where(squeezed_mask, (unmasked(maps) - mean) / std, 0.0)

    feats_mean, feats_std = (
        np.nanmean(unmasked(split_maps["train"]), axis=(0, -2, -1), keepdims=True),
        np.nanstd(unmasked(split_maps["train"]), axis=(0, -2, -1), keepdims=True),
    )
    targets_mean, targets_std = (
        np.nanmean(unmasked(split_targets["train"]), axis=(0, -2, -1), keepdims=True),
        np.nanstd(unmasked(split_targets["train"]), axis=(0, -2, -1), keepdims=True),
    )

    datasets = {
        name: SWIDataset(
            maps_feats=standardize(split_maps[name], feats_mean, feats_std),
            target_maps=standardize(split_targets[name], targets_mean, targets_std),
            timestamps=split_timesteps[name],
            mask=mask,
            num_input_steps=num_input_steps,
            history_dropout_prob=history_dropout_prob if name == "train" else 0.0,
            history_dropout_range=history_dropout_range,
        )
        for name in split_maps
    }

    return datasets, {
        "feats_mean": feats_mean,
        "feats_std": feats_std,
        "targets_mean": targets_mean,
        "targets_std": targets_std,
    }
