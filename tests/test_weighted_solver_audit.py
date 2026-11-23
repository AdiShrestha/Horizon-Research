import numpy as np
import pytest

from trajcp.data.weighted_audit import WeightedAudit


def test_hand_oracle_with_distinct_worst_weight_and_discrepancy_ids():
    difference = np.array([[[4., 1.], [2., 0.]], [[1., 3.], [1., 2.]]])
    weights = np.array([[[8., 1.], [1., 4.]], [[.25, 3.], [2., 2.]]])
    reducer = WeightedAudit()
    reducer.update([17], difference[:1], weights[:1])
    reducer.update([22], difference[1:], weights[1:])
    report = reducer.report(2.)
    assert report["trajectories"] == [
        {"id": 17, "macro_step": 2, "cell": 0, "discrepancy": 2., "weight": 1.,
         "epsilon": 2., "epsilon_over_q": 1.},
        {"id": 22, "macro_step": 1, "cell": 0, "discrepancy": 1., "weight": .25,
         "epsilon": 4., "epsilon_over_q": 2.},
    ]
    assert report["matched_maximum"]["id"] == 22
    assert report["conservative_maximum"] == {
        "macro_step": 1, "cell": 0, "discrepancy": 4., "weight": .25, "epsilon": 16.,
        "epsilon_over_q": 8., "discrepancy_id": 17, "weight_id": 22,
    }
    assert not report["passes_strict_1e_3"]
    whole = WeightedAudit()
    whole.update([17, 22], difference, weights)
    assert whole.report(2.) == report


def test_flat_arm_equals_matched_maximum_and_strict_boundary():
    reducer = WeightedAudit()
    reducer.update([0, 1], np.array([[[.002, 0]], [[.001, .0005]]]), 1.)
    report = reducer.report(2.)
    assert report["matched_maximum"]["epsilon_over_q"] == report["conservative_maximum"]["epsilon_over_q"] == .001
    assert not report["passes_strict_1e_3"]
    assert reducer.report(2.1)["passes_strict_1e_3"]


@pytest.mark.parametrize("invalid", [0., -1., np.nan, np.inf])
def test_refuses_invalid_weights_or_calibration_thresholds(invalid):
    reducer = WeightedAudit()
    with pytest.raises(ValueError, match="positive"):
        reducer.update([0], np.zeros((1, 1, 2)), invalid)
    reducer.update([0], np.zeros((1, 1, 2)), 1.)
    with pytest.raises(ValueError, match="positive"):
        reducer.report(invalid)
