"""Run the notebook's Python cells on small inputs and verify its exported runs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from m5_forecast.artifacts import load_artifacts
from m5_forecast.evaluate import predict


def test_notebook_reproduces_both_stages_and_exports_models(m5_data, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    matplotlib = pytest.importorskip("matplotlib", reason="Install the notebook dependency group.")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    root = Path(__file__).parents[1]
    monkeypatch.chdir(root)
    notebook = json.loads((root / "notebooks/m5_experiments.ipynb").read_text())
    namespace = {"__name__": "__main__"}
    run_dir = tmp_path / "experiment"
    overrides = {
        "DATA_DIR": m5_data,
        "RUN_DIR": run_dir,
        "NUM_SERIES": 3,
        "VAL_DAYS": 28,
        "TEST_DAYS": 28,
        "CNN_EPOCHS": 2,
        "RESIDUAL_EPOCHS": 2,
        "XGB_ROUNDS": 3,
        "THREADS": 1,
    }
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"notebook:{cell['id']}", "exec"), namespace)
        if "parameters" in cell["metadata"].get("tags", []):
            namespace.update(overrides)

    assert len(namespace["test_origins"]) == 4
    results = pd.read_csv(run_dir / "comparison.csv", index_col="Method")
    assert len(results) == 5
    assert np.isfinite(results.to_numpy()).all()
    assert namespace["original_cnn"].reconcile_alpha == 0
    assert namespace["original_trees"].attr("reconcile_alpha") is None
    choices = namespace["validation_results"]
    for family, choice_name in [
        ("cnn_candidates", "cnn_choice"),
        ("tree_candidates", "tree_choice"),
    ]:
        candidates = choices[choices.Method.isin(namespace[family])]
        assert namespace[choice_name]["Validation WRMSSE"] == candidates["Validation WRMSSE"].min()

    for stage, prefix in [("original", "Original"), ("improved", "Selected")]:
        directory = run_dir / stage
        model, data, metadata = load_artifacts(directory)
        metrics = json.loads((directory / "metrics.json").read_text())
        assert metadata["test_metrics"] == metrics
        assert metrics["num_predictions"] == 84
        for label, key in [("Conv1D", "cnn"), ("XGBoost", "xgboost")]:
            assert metrics[f"{key}_mae"] == pytest.approx(results.loc[f"{prefix} {label}", "MAE"])
            assert metrics[f"{key}_wrmsse"] == pytest.approx(
                results.loc[f"{prefix} {label}", "Demo WRMSSE"]
            )
        restored = predict(model, data, data.split.origins("test"), torch.device("cpu"))
        np.testing.assert_allclose(restored, namespace["test_forecasts"][f"{prefix} Conv1D"])
        assert (directory / "mae_by_date.csv").is_file()
        assert (directory / "wrmsse_by_level.csv").is_file()
