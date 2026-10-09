"""Persist weights, preprocessing, and the selected data for a lightweight UI."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from m5_forecast.data import CALENDAR_FEATURES, HISTORY, HORIZON, PreparedData, Split
from m5_forecast.model import DemandCNN


def save_artifacts(
    directory: Path, model: DemandCNN, data: PreparedData, run_config: dict, metrics: dict
) -> None:
    """Save a CPU-portable state dict and readable JSON preprocessing metadata."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {key: value.detach().cpu() for key, value in model.state_dict().items()},
        directory / "model.pt",
    )
    metadata = {
        "history": HISTORY,
        "horizon": HORIZON,
        "calendar_features": CALENDAR_FEATURES,
        "series_ids": data.series_ids,
        "scales": data.scales.tolist(),
        "split": asdict(data.split),
        "run_config": run_config,
        "test_metrics": metrics,
        "model_config": {
            "seasonal_residual": model.seasonal_residual,
            "reconcile_alpha": model.reconcile_alpha,
        },
    }
    (directory / "preprocessing.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.savez_compressed(
        directory / "demo_data.npz", sales=data.sales, calendar=data.calendar, dates=data.dates
    )


def load_artifacts(directory: Path) -> tuple[DemandCNN, PreparedData, dict]:
    """Restore the saved model and data on CPU; no CSV loading or training."""
    metadata = json.loads((directory / "preprocessing.json").read_text())
    if (
        metadata["history"] != HISTORY
        or metadata["horizon"] != HORIZON
        or metadata["calendar_features"] != CALENDAR_FEATURES
    ):
        raise ValueError("Artifact feature definitions differ from this code; train again.")
    with np.load(directory / "demo_data.npz", allow_pickle=False) as arrays:
        data = PreparedData(
            arrays["sales"],
            arrays["calendar"],
            arrays["dates"],
            metadata["series_ids"],
            np.asarray(metadata["scales"], dtype=np.float32),
            Split(**metadata["split"]),
        )
    model = DemandCNN(len(data.series_ids), **metadata.get("model_config", {}))
    model.load_state_dict(torch.load(directory / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, data, metadata
