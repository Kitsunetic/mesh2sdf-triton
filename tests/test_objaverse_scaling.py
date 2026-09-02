from benchmarks.objaverse_scaling import accuracy_gated_speedup, scalability_score


def test_scalability_score_prioritizes_accuracy_before_speed() -> None:
    # Given three accurate cases and a sub-unit worst-case speedup
    accuracy_passes = 3
    minimum_speedup = 0.131

    # When the autoresearch score is computed
    score = scalability_score(accuracy_passes, minimum_speedup)

    # Then accuracy occupies the integer tier and speed remains the tie-breaker
    assert score == 3000.131


def test_scalability_score_target_means_four_accurate_cases_at_ten_x() -> None:
    # Given every benchmark case is accurate with a 10x minimum speedup
    accuracy_passes = 4
    minimum_speedup = 10.0

    # When the autoresearch score is computed
    score = scalability_score(accuracy_passes, minimum_speedup)

    # Then the configured target is reached exactly
    assert score == 4010.0


def test_dense_speedup_metric_rejects_inaccurate_results() -> None:
    assert accuracy_gated_speedup(42.0, accuracy_pass=True) == 42.0
    assert accuracy_gated_speedup(42.0, accuracy_pass=False) == 0.0
