import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from trajcp.conformal.core import calibrate, field_score


@pytest.mark.parametrize("oracle", ["puncc", "mapie"])
def test_core_7_scalar_split_cp(oracle):
    if oracle == "puncc":
        from deel.puncc.api.prediction import BasePredictor
        from deel.puncc.regression import SplitCP
    else:
        from mapie.regression import SplitConformalRegressor

    model = DummyRegressor(strategy="constant", constant=0.0)
    model.fit(np.zeros((2, 1)), np.zeros(2))
    rng = np.random.default_rng(701)
    differences = []
    for n, alpha in [(27, 0.11), (49, 0.13), (63, 0.075), (99, 0.24)]:
        for ties in (False, True):
            features = np.arange(n, dtype=np.float64)[:, None]
            target = rng.normal(size=n)
            if ties:
                target = np.round(target, 1)
            prediction = model.predict(np.zeros((1, 1)))[0]
            scores = field_score((target - prediction)[:, None, None], np.array(1.0))
            expected = calibrate(scores, alpha)
            if oracle == "puncc":
                split = SplitCP(BasePredictor(model, is_trained=True), train=False)
                split.fit(X_calib=features, y_calib=target)
                center, lower, upper = split.predict(features[:1], alpha=alpha)
                center, lower, upper = (np.asarray(x).item() for x in (center, lower, upper))
                radii = np.array([center - lower, upper - center])
            else:
                split = SplitConformalRegressor(model, confidence_level=1 - alpha)
                split.conformalize(features, target)
                center, bounds = split.predict_interval(features[:1])
                radii = np.array(
                    [center[0] - bounds[0, 0, 0], bounds[0, 1, 0] - center[0]]
                )
            differences.extend(abs(radii - expected))
    max_difference = max(differences)
    print(f"{oracle} max absolute threshold difference across eight cases={max_difference:.3g}")
    assert max_difference <= 1e-12
