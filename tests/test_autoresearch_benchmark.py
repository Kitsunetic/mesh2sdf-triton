from benchmarks.autoresearch import fixed_cases


def test_fixed_cases_are_diverse_watertight_meshes() -> None:
    # Given the fixed benchmark inputs
    cases = fixed_cases()

    # When their public properties are inspected
    names = tuple(case.name for case in cases)

    # Then the suite covers three distinct watertight meshes
    assert names == ("box", "icosphere", "torus")
    assert all(case.vertices.shape[1] == 3 for case in cases)
    assert all(case.faces.shape[1] == 3 for case in cases)
    assert all(case.is_watertight for case in cases)
