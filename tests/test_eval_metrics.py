from toposteer.evaluation import normalized_trapz_area


def test_normalized_trapz_area_constant_curve_equals_constant():
    area = normalized_trapz_area([0.0, 0.5, 1.0], [0.7, 0.7, 0.7])
    assert area == 0.7


def test_normalized_trapz_area_handles_unsorted_inputs():
    area = normalized_trapz_area([1.0, 0.0, 0.5], [1.0, 0.0, 0.5])
    assert abs(area - 0.5) < 1e-9
