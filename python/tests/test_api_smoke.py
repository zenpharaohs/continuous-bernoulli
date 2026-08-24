import numpy as np
import pytest

from cb_sampler import CbStream, draw_streams, sample
from cb_sampler.examples import posterior_demo, streaming_demo
from cb_sampler.validation import main as validation_main, run_validation


def test_draw_shape_dtype_and_bounds():
    stream = CbStream(2.0, 5.0, seed=123)
    draws = stream.draw(32)
    assert draws.shape == (32,)
    assert draws.dtype == np.float64
    assert np.all(draws > 0.0)
    assert np.all(draws < 1.0)
    stream.close()


def test_sample_shapes():
    scalar = sample(2.0, 5.0, seed=123)
    assert isinstance(scalar, np.float64)

    vector = sample(2.0, 5.0, size=7, seed=123)
    assert vector.shape == (7,)

    matrix = sample(2.0, 5.0, size=(2, 3), seed=123)
    assert matrix.shape == (2, 3)


@pytest.mark.parametrize(
    "chi,nu,expected",
    [(0.0, 3.0, 0.0), (3.0, 3.0, 1.0)],
)
def test_closed_family_boundaries_are_point_masses(chi, nu, expected):
    stream = CbStream(chi, nu, seed=123)
    state = stream.peek()
    assert state.regime == 4
    assert state.sigma == 0.0
    assert np.all(stream.draw(257) == expected)
    assert np.all(sample(chi, nu, size=17, seed=456) == expected)
    stream.close()


@pytest.mark.parametrize("observation,expected", [(0.0, 0.0), (1.0, 1.0)])
def test_repeated_endpoint_updates_never_enter_rejection_sampler(
        observation, expected):
    stream = CbStream(0.0, 0.0, seed=1729, buf_size=64)
    for _ in range(1000):
        stream.update(observation)
        assert stream.draw(1)[0] == expected
    state = stream.peek()
    assert state.regime == 4
    stream.close()


def test_endpoint_state_can_transition_back_to_continuous_family():
    stream = CbStream(0.0, 0.0, seed=19)
    stream.update(0.0)
    assert stream.draw(1)[0] == 0.0
    stream.update(1.0)
    state = stream.peek()
    assert state.chi == 1.0
    assert state.nu == 2.0
    assert state.regime != 4
    draw = stream.draw(1)[0]
    assert 0.0 < draw < 1.0
    stream.close()


@pytest.mark.parametrize(
    "chi,nu",
    [(1e-14, 10.0), (10.0 - 1e-14, 10.0)],
)
def test_resolved_near_boundary_state_draws_without_stalling(chi, nu):
    """Interior tail statistics must not inherit the mode solver's floor."""
    stream = CbStream(chi, nu, seed=1729, buf_size=64)
    draws = stream.draw(64)
    assert np.all(draws > 0.0)
    assert np.all(draws < 1.0)
    stream.close()


def test_seed_repeatability():
    a = sample(2.0, 5.0, size=16, seed=9876)
    b = sample(2.0, 5.0, size=16, seed=9876)
    np.testing.assert_array_equal(a, b)


def test_stream_index_diversifies_explicit_seed():
    a = CbStream(2.0, 5.0, seed=9876, stream_idx=0).draw(16)
    b = CbStream(2.0, 5.0, seed=9876, stream_idx=1).draw(16)
    assert not np.array_equal(a, b)


def test_draw_streams_shape_bounds_and_accounting():
    streams = [
        CbStream(1.0 + index, 5.0, seed=123, stream_idx=index)
        for index in range(3)
    ]
    draws = draw_streams(streams)
    assert draws.shape == (3,)
    assert draws.dtype == np.float64
    assert np.all(draws > 0.0)
    assert np.all(draws < 1.0)
    assert all(stream.diagnostics()["total_draws"] == 1 for stream in streams)
    for stream in streams:
        stream.close()


def test_update_peek_and_diagnostics():
    stream = CbStream(1.0, 2.0, seed=11)
    stream.update(0.25)
    stream.update_batch(np.array([0.5, 0.75]))
    state = stream.peek()
    assert state.chi == pytest.approx(2.5)
    assert state.nu == pytest.approx(5.0)
    assert state.regime in {0, 1, 2, 3, 4}
    diagnostics = stream.diagnostics()
    assert diagnostics["enabled"] is False
    stream.draw(3)
    assert stream.diagnostics()["total_draws"] == 3
    stream.close()


def test_set_stats_replaces_history_and_preserves_stream():
    stream = CbStream(1.0, 2.0, seed=11)
    stream.update_batch(np.array([0.1, 0.2, 0.3]))
    stream.set_stats(3.0, 4.0)
    state = stream.peek()
    assert state.chi == pytest.approx(3.0)
    assert state.nu == pytest.approx(4.0)
    draws = stream.draw(8)
    assert np.all(draws > 0.0)
    assert np.all(draws < 1.0)
    stream.close()


def test_set_stats_accepts_closed_family_boundary():
    stream = CbStream(1.0, 2.0, seed=11)
    stream.set_stats(0.0, 4.0)
    assert stream.peek().regime == 4
    assert np.all(stream.draw(8) == 0.0)
    stream.set_stats(4.0, 4.0)
    assert np.all(stream.draw(8) == 1.0)
    stream.close()


def test_context_manager_closes():
    with CbStream(2.0, 5.0, seed=1) as stream:
        stream.draw(1)
    with pytest.raises(RuntimeError, match="closed"):
        stream.draw(1)


@pytest.mark.parametrize(
    "chi,nu",
    [
        (-1.0, 2.0),
        (3.0, 2.0),
        (0.0, -1.0),
        (float("nan"), 1.0),
    ],
)
def test_invalid_stats(chi, nu):
    with pytest.raises(ValueError):
        CbStream(chi, nu, seed=1)


@pytest.mark.parametrize("x", [-0.1, 1.1, float("inf"), float("nan")])
def test_invalid_observations(x):
    stream = CbStream(1.0, 2.0, seed=1)
    with pytest.raises(ValueError):
        stream.update(x)
    stream.close()


def test_python_validation_smoke():
    results = run_validation(mode="smoke", n=2_000, grid_size=2000)
    assert len(results) > 0
    assert all(abs(r.leg_snr) <= 3.0 for r in results)


def test_python_validation_json_artifact(tmp_path):
    out = tmp_path / "validation.json"
    rc = validation_main(["--mode", "smoke", "--n", "2_000", "--grid-size", "2000", "--json-out", str(out)])
    assert rc == 0
    assert out.exists()
    assert '"passed": true' in out.read_text()


def test_examples_smoke():
    posterior = posterior_demo(n=128, seed=1)
    assert 0.0 < posterior["q05"] < posterior["q95"] < 1.0
    streaming_demo(seed=1)
