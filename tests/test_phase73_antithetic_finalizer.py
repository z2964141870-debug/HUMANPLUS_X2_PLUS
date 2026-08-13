import torch

from cwi_x2.phase73_antithetic import stratified_bootstrap_projection


def test_phase73_two_level_bootstrap_is_available() -> None:
    metric = torch.ones(5, 64, 66, dtype=torch.float64)
    component = 0.5 * metric
    result = stratified_bootstrap_projection(component, metric, seed=73, draws=64)
    assert result["point"] == 0.5
    assert result["p025"] > 0.0
