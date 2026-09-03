"""What the "global bias" is actually made of: a lead-time ramp, not an offset.

WHY THIS EXISTS. ``utils/bias_correction.py`` estimated a global offset on windows that contain no
information from the scored window, and the transfer failed in a specific and repeatable way: the
residual val->test gap is +0.32 to +0.43 sd on ALL SEVEN checkpoints, near enough the same number
whatever the model. A quantity that is identical across seven different models is not a property
of any of them. It is a property of the window. This script finds out which property.

WHAT IT SEPARATES. The mean error at lead time k decomposes into two things that look identical in
a single scalar and behave completely differently:

  model climatology offset   Where the free-running rollout SETTLES once the initial condition has
                             decayed -- its second-year seasonal cycle -- measured against the
                             1960-2019 observed seasonal climatology the model was trained on.
                             This is a property of the weights. It is estimable from any long
                             historical period and it transfers.

  window climate anomaly     How far the observations over the scored window sit from that same
                             1960-2019 climatology. This is a property of the weather. A
                             free-running rollout cannot know it, and no calibration window that
                             ends before the forecast starts can supply it.

The measured bias at long lead is the difference of the two. Reporting it as one number attributes
a climate anomaly to the model, and then invites a correction that cannot work.

WHY THE GLOBAL SCALAR IS THE WRONG OBJECT. The ramp changes sign. At the seed-61 round-11
checkpoint the bias runs -0.56 sd at leads 1-6 and +0.61 sd at leads 19-24, so its 24-month mean is
-0.05 sd -- a number that says "unbiased" about a model that is badly wrong at both ends and merely
wrong in opposite directions. A per-lead-time table is not a refinement of the global offset here;
it is the first description that is not actively misleading.

All windows start in January and run 24 months, so lead time k is the same calendar month in every
one of them, and the year-2 block of every window covers the same twelve months. That is what lets
the model's settled seasonal cycle be read off directly and compared across windows.
"""

import copy
import logging

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils.bias_correction import WINDOWS, rollout
from utils.preprocessing import coerce_comma_decimal_columns

logger = logging.getLogger(__file__)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
TRAIN_END_YEAR = 2019  # the chronological split puts train at 1960-01..2019-09


def observed_climatology(cfg: DictConfig) -> tuple[pd.Series, pd.Series]:
    """Masked-mean swi_u per calendar month over the training period, and per (year, month).

    The climatology is the reference a free-running forecast should fall back to once it has lost
    the initial condition, so it is taken over the training period only -- that is the climate the
    weights actually saw.
    """
    dataset_cfg = cfg["dataset"]
    frame = pd.read_parquet(dataset_cfg["filepath"])
    frame["mask"] = 1.0 * (frame.loc[:, "scenario"] != 0)
    frame = coerce_comma_decimal_columns(frame, [dataset_cfg["target_column"]])
    frame = frame[frame["mask"] > 0]
    frame["Y"] = frame["year"].astype(int)
    frame["M"] = frame["month"].astype(int)
    target = dataset_cfg["target_column"]
    climatology = frame[frame["Y"] <= TRAIN_END_YEAR].groupby("M")[target].mean()
    by_month = frame.groupby(["Y", "M"])[target].mean()
    return climatology, by_month


def window_months(year: int, month: int, steps: int) -> list[tuple[int, int]]:
    """The (year, month) pairs a rollout of ``steps`` steps from ``year``-``month`` covers."""
    return [(year + (month - 1 + k) // 12, (month - 1 + k) % 12 + 1) for k in range(steps)]


@hydra.main(config_path="../config", config_name="inference_swigan", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Decompose the mean error of every window into a model offset and a climate anomaly."""
    climatology, observed = observed_climatology(cfg)
    runs = {key: rollout(cfg, year, month, steps)
            for key, (_, year, month, steps) in WINDOWS.items()}
    target_std = runs["test"]["target_std"]
    key = "on"  # inference dropout ON: settled 7 of 7 by utils/ensemble_scores.py

    print("\n" + "=" * 108)
    print("WHAT THE BIAS IS MADE OF".center(108))
    print(f"masked-mean swi_u · training climatology is 1960-{TRAIN_END_YEAR} · "
          f"targets_std {target_std:.4f}".center(108))
    print("=" * 108)

    print(f"\n  BIAS BY LEAD TIME, in sd (mean of y - ensemble mean over pixels)")
    print(f"  {'window':22s} {'m1-6':>8s} {'m7-12':>8s} {'m13-18':>8s} {'m19-24':>8s} "
          f"{'24m mean':>9s} {'slope/yr':>9s}")
    print("  " + "-" * 78)
    ramps = {}
    for name, (label, *_rest) in WINDOWS.items():
        resid = runs[name]["truth"] - runs[name][key].mean(axis=0)  # (T, P)
        by_lead = resid.mean(axis=1) / target_std
        ramps[name] = by_lead
        slope = float(np.polyfit(np.arange(len(by_lead)), by_lead, 1)[0]) * 12
        print(f"  {label:22s} {by_lead[:6].mean():+8.3f} {by_lead[6:12].mean():+8.3f} "
              f"{by_lead[12:18].mean():+8.3f} {by_lead[18:].mean():+8.3f} "
              f"{by_lead.mean():+9.3f} {slope:+9.3f}")
    print("  " + "-" * 78)
    print("  A global offset is the '24m mean' column. Read it against the four block columns it")
    print("  averages: where they straddle zero, that single number is describing nothing.")

    saved = {"lead_bias_" + n: r for n, r in ramps.items()}
    for name, (label, year, month, steps) in WINDOWS.items():
        members = runs[name][key]
        pred = members.mean(axis=0).mean(axis=1)  # (T,) masked-mean per rollout step
        months = window_months(year, month, steps)
        year2 = [(k, m) for k, (_y, m) in enumerate(months) if k >= 12]

        model_offset = np.array([pred[k] - climatology[m] for k, m in year2])
        climate_anom = np.array([observed[months[k]] - climatology[m] for k, m in year2])

        print(f"\n  YEAR 2 OF THE {label} ROLLOUT  (the initial condition has decayed by here)")
        print(f"  {'month':6s} {'settled pred':>13s} {'clim 60-19':>11s} {'MODEL offset':>13s} "
              f"{'observed':>9s} {'CLIMATE anom':>13s} {'bias':>8s}")
        print("  " + "-" * 82)
        for (k, m), off, anom in zip(year2, model_offset, climate_anom, strict=True):
            print(f"  {MONTHS[m - 1]:6s} {pred[k]:13.4f} {climatology[m]:11.4f} {off:+13.4f} "
                  f"{observed[months[k]]:9.4f} {anom:+13.4f} {anom - off:+8.4f}")
        print("  " + "-" * 82)
        print(f"  {'mean':6s} {'':13s} {'':11s} {model_offset.mean():+13.4f} {'':9s} "
              f"{climate_anom.mean():+13.4f} {climate_anom.mean() - model_offset.mean():+8.4f}")
        print(f"  {'in sd':6s} {'':13s} {'':11s} {model_offset.mean() / target_std:+13.3f} {'':9s} "
              f"{climate_anom.mean() / target_std:+13.3f} "
              f"{(climate_anom.mean() - model_offset.mean()) / target_std:+8.3f}")
        saved[f"model_offset_{name}"] = model_offset
        saved[f"climate_anom_{name}"] = climate_anom
        saved[f"settled_pred_{name}"] = np.array([pred[k] for k, _m in year2])

    print(f"\n  MODEL offset is a property of the weights: where this rollout settles, against the")
    print(f"  climate it was trained on. It is the same seasonal cycle in every window, so it is")
    print(f"  estimable from history and it transfers. CLIMATE anom is a property of the weather")
    print(f"  over the scored window. No window that ends before the forecast starts contains it.")
    print(f"  Only the first column is correctable.")
    print("=" * 108)

    saved["climatology"] = climatology.to_numpy()
    saved["target_std"] = target_std
    out = cfg["model"].get("saving_path")
    if out:
        np.savez(f"{out}_structure.npz", **saved)
        logger.info(f"Saved to '{out}_structure.npz'.")


if __name__ == "__main__":
    main()