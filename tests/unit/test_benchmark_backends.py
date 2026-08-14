import pytest

from benchmarks.benchmark_backends import Measurement, print_measurement, timed, timed_batch


def test_scalar_and_batch_measurements_include_warmup():
    scalar_calls = 0

    def scalar() -> None:
        nonlocal scalar_calls
        scalar_calls += 1

    scalar_result = timed(scalar, iterations=3, repeats=2)
    assert scalar_calls == 7
    assert scalar_result.minimum_ms <= scalar_result.latency_ms <= scalar_result.maximum_ms
    assert scalar_result.rate > 0.0

    batch_calls = 0

    def batch() -> None:
        nonlocal batch_calls
        batch_calls += 1

    batch_result = timed_batch(batch, operations=4, repeats=2)
    assert batch_calls == 3
    assert batch_result.rate > 0.0


def test_measurement_output_uses_operation_specific_units(capsys):
    print_measurement(
        "cpp",
        "capsules:",
        Measurement(0.02, 0.019, 0.021),
        rate_unit="queries/s",
        latency_unit="ms/query",
    )
    output = capsys.readouterr().out
    assert "50000.0 queries/s" in output
    assert "0.0200 ms/query" in output
    assert "range=[0.0190, 0.0210] ms/query" in output


def test_measurement_helpers_reject_empty_runs():
    with pytest.raises(ValueError, match="positive"):
        timed(lambda: None, iterations=0, repeats=1)
    with pytest.raises(ValueError, match="positive"):
        timed_batch(lambda: None, operations=1, repeats=0)
