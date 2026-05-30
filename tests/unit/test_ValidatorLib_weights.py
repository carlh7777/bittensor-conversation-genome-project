import numpy as np
import pytest

from conversationgenome.validator.ValidatorLib import ValidatorLib


def test_update_scores_basic():
    vl = ValidatorLib()

    # 4 miners, rewards for two uids
    rewards = np.array([1.0, 0.5], dtype=np.float32)
    uids = np.array([0, 1], dtype=np.int64)
    ema_scores = np.zeros(4, dtype=np.float32)
    scores = np.zeros(4, dtype=np.float32)

    moving_average_alpha = 0.1
    device = "cpu"
    neurons = 4
    nonlinear_power = 1.0

    new_scores, new_ema = vl.update_scores(rewards, uids, ema_scores, scores, moving_average_alpha, device, neurons, nonlinear_power)

    # Basic sanity checks
    assert isinstance(new_scores, np.ndarray)
    assert isinstance(new_ema, np.ndarray)
    assert new_scores.shape[0] == neurons
    # Scores should be normalized to sum 1 (or be a valid probability vector)
    assert pytest.approx(1.0, rel=1e-3) == float(np.sum(new_scores))
    # EMA values should be >= 0
    assert np.all(new_ema >= 0)


def test_get_raw_weights_distribution_no_burn():
    vl = ValidatorLib()

    # scores with one zero entry
    scores = np.array([0.2, 0.5, 0.3, 0.0], dtype=float)

    weights = vl.get_raw_weights(scores)

    # Returned array shape and sum
    assert isinstance(weights, np.ndarray)
    assert weights.shape == scores.shape
    # zero input should remain zero in result
    assert weights[3] == pytest.approx(0.0)
    # Sum of absolute weights should be ~1.0
    assert pytest.approx(1.0, rel=1e-4) == float(np.sum(np.abs(weights)))

    # The uid with the highest original score (index 1) should receive the highest weight
    assert weights[1] == pytest.approx(np.max(weights))


def test_get_raw_weights_all_zero():
    vl = ValidatorLib()

    scores = np.zeros(5, dtype=float)
    weights = vl.get_raw_weights(scores=scores)

    # Should return zeros array of same shape
    assert isinstance(weights, np.ndarray)
    assert np.all(weights == 0)


def test_get_raw_weights_distribution_with_burn():
    vl = ValidatorLib()

    burn_uid = 2
    burn_rate = 0.9

    # scores with one zero entry
    scores = np.array([0.2, 0.5, 0.3, 0.0], dtype=float)

    weights = vl.get_raw_weights(scores=scores, burn_uid=burn_uid, burn_rate=burn_rate)
    print(f"Raw weights: {weights}")

    # Returned array shape and sum
    assert isinstance(weights, np.ndarray)
    assert weights.shape == scores.shape
    # zero input should remain zero in result
    assert weights[3] == pytest.approx(0.0)

    # burn_uid should have weight close to burn_rate
    assert weights[burn_uid] == pytest.approx(burn_rate, rel=1e-4)

    # Sum of absolute weights should be ~1.0
    assert pytest.approx(1.0, rel=1e-4) == float(np.sum(np.abs(weights)))

    # The burn burn_uid should receive the highest weight
    assert weights[burn_uid] == pytest.approx(np.max(weights))


def test_get_raw_weights_baseline_for_eligible_zero_score_uids():
    """Score-0 UIDs in the eligible set receive a small baseline weight; uneligible
    score-0 UIDs (and the burn_uid) stay at zero."""
    vl = ValidatorLib()

    burn_uid = 4
    burn_rate = 0.9

    # UIDs 0,1,2 scored; 3 and 5 are zero. Pretend 3 is serving (eligible) and 5 isn't.
    scores = np.array([0.2, 0.5, 0.3, 0.0, 0.0, 0.0], dtype=float)
    eligible_uids = [0, 1, 2, 3]  # excludes 4 (burn) and 5 (not serving)

    weights = vl.get_raw_weights(
        scores=scores,
        burn_uid=burn_uid,
        burn_rate=burn_rate,
        eligible_uids=eligible_uids,
    )

    assert weights[burn_uid] == pytest.approx(burn_rate, rel=1e-4)
    assert weights[5] == pytest.approx(0.0)  # uneligible score-0 stays at zero
    assert weights[3] > 0  # eligible score-0 gets baseline
    assert weights[3] < weights[0]  # baseline is far below any cubic-scored weight
    assert pytest.approx(1.0, rel=1e-4) == float(np.sum(np.abs(weights)))


def test_get_raw_weights_no_baseline_when_eligible_uids_none():
    """When eligible_uids is None (legacy callers), score-0 UIDs stay at zero."""
    vl = ValidatorLib()

    scores = np.array([0.2, 0.5, 0.3, 0.0], dtype=float)
    weights = vl.get_raw_weights(scores=scores, burn_uid=2, burn_rate=0.5)

    assert weights[3] == pytest.approx(0.0)


def test_get_raw_weights_distribution_with_smaller_burn():
    vl = ValidatorLib()

    burn_uid = 3
    burn_rate = 0.5

    # scores with one zero entry
    scores = np.array([0.2, 0.5, 0.3, 0.0], dtype=float)

    weights = vl.get_raw_weights(scores=scores, burn_uid=burn_uid, burn_rate=burn_rate)
    print(f"Raw weights: {weights}")

    # Returned array shape and sum
    assert isinstance(weights, np.ndarray)
    assert weights.shape == scores.shape

    # burn_uid should have weight close to burn_rate
    assert weights[burn_uid] == pytest.approx(burn_rate, rel=1e-4)

    # Sum of absolute weights should be ~1.0
    assert pytest.approx(1.0, rel=1e-4) == float(np.sum(np.abs(weights)))

    # The burn burn_uid should receive the highest weight
    assert weights[burn_uid] == pytest.approx(np.max(weights))
