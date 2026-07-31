import json
from pathlib import Path
from types import SimpleNamespace

import torch

from iic.parameters import flatten_parameters
from iic.pinn.config import load_config
from iic.pinn.model import MLP
from iic.pinn.problem import PinnFunctions
import iic.pinn.train as train_module


ROOT = Path(__file__).resolve().parents[1]


def test_lbfgs_phase_uses_closure_and_records_actual_work(tmp_path, monkeypatch):
    raw = json.loads((ROOT / "configs" / "pinn-smoke.json").read_text())
    raw["training"]["phases"] = [
        {
            "optimizer": "lbfgs",
            "learning_rate": 1.0,
            "max_iter": 2,
            "max_eval": 3,
            "line_search_fn": "strong_wolfe",
        }
    ]
    config_path = tmp_path / "lbfgs.json"
    config_path.write_text(json.dumps(raw))
    config = load_config(config_path)
    model = MLP((2,)).to(dtype=torch.float64)

    def fake_functions(candidate_model, *_args, **_kwargs):
        theta = flatten_parameters(candidate_model)
        return PinnFunctions(
            theta=theta,
            constraint_fn=lambda candidate: candidate[:1],
            pde_regularizer_fn=lambda candidate: candidate.new_zeros(()),
            training_objective_fn=lambda candidate: candidate.square().sum(),
            regularizer_fn=lambda candidate: candidate.square().sum(),
            component_values_fn=lambda candidate: {},
            metadata={},
        )

    monkeypatch.setattr(train_module, "build_functions", fake_functions)
    data = SimpleNamespace(
        evaluation_coords=torch.zeros((2, 2), dtype=torch.float64),
        evaluation_values=torch.ones((2, 1), dtype=torch.float64),
    )

    result = train_module.train(model, data, config, nu=0.5, rho=1.0)

    phase = result.optimizer_phases[0]
    assert phase["optimizer"] == "lbfgs"
    assert phase["closure_calls"] >= 1
    assert phase["function_evaluations"] == phase["closure_calls"]
    assert 0 <= phase["actual_iterations"] <= phase["requested_steps"]
