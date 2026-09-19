"""Tests for PEST forward-run parameter multipliers.

These exercise code that lives inside the r''' ... ''' template returned by
_generate_forward_run_script(), i.e. the forward_run.py that every PEST++
agent executes on each iteration. The template is loaded as a real module so
the tests cover the artifact that actually ships to the agents.

Regression context: _set_rch_multiplier / _set_evt_multiplier used to do
`np.array(data) * mult` on the result of MFTransientArray.get_data(), which
is a dict keyed by STRESS PERIOD. That raised
"TypeError: unsupported operand type(s) for *: 'dict' and 'float'", which the
functions' broad `except` turned into a printed warning — so the multiplier
was silently dropped and PEST saw recharge/ET as completely insensitive.
"""

import importlib.util
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.services.pest_setup import _generate_forward_run_script


@pytest.fixture(scope="module")
def fwd():
    """Load the generated forward_run.py as an importable module."""
    src = _generate_forward_run_script()
    compile(src, "forward_run.py", "exec")  # must be valid Python

    tmpdir = tempfile.mkdtemp(prefix="fwd_run_test_")
    path = Path(tmpdir) / "forward_run.py"
    path.write_text(src, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("fwd_run_under_test", path)
    module = importlib.util.module_from_spec(spec)
    # Module name is not "__main__", so main() does not execute on import.
    spec.loader.exec_module(module)
    return module


def test_generated_script_is_valid_python(fwd):
    assert hasattr(fwd, "_scale_mf6_array")
    assert hasattr(fwd, "_set_rch_multiplier")
    assert hasattr(fwd, "_set_evt_multiplier")


# ---------------------------------------------------------------------------
# _scale_mf6_array
# ---------------------------------------------------------------------------

def test_scale_transient_dict_scales_every_stress_period(fwd):
    """The dict from get_data() is keyed by stress period — scale them all."""
    data = {
        0: np.full((3, 4), 0.001),
        1: np.full((3, 4), 0.002),
        2: np.full((3, 4), 0.003),
    }
    out = fwd._scale_mf6_array(data, 2.0)

    assert isinstance(out, dict)
    assert sorted(out.keys()) == [0, 1, 2]
    assert out[0] == pytest.approx(np.full((3, 4), 0.002))
    assert out[1] == pytest.approx(np.full((3, 4), 0.004))
    assert out[2] == pytest.approx(np.full((3, 4), 0.006))


def test_scale_plain_array(fwd):
    """A non-transient array still scales as a plain ndarray."""
    out = fwd._scale_mf6_array(np.full((2, 2), 5.0), 3.0)

    assert isinstance(out, np.ndarray)
    assert out == pytest.approx(np.full((2, 2), 15.0))


def test_scale_skips_none_periods(fwd):
    """Periods with no data are dropped rather than raising."""
    out = fwd._scale_mf6_array({0: np.ones((2, 2)), 1: None}, 4.0)

    assert sorted(out.keys()) == [0]
    assert out[0] == pytest.approx(np.full((2, 2), 4.0))


def test_scale_does_not_raise_on_dict(fwd):
    """The exact failure mode of the original bug."""
    try:
        fwd._scale_mf6_array({0: np.ones((2, 2))}, 2.0)
    except TypeError as e:  # pragma: no cover - guard against regression
        pytest.fail(f"dict input must not raise TypeError, got: {e}")


# ---------------------------------------------------------------------------
# _set_rch_multiplier / _set_evt_multiplier
# ---------------------------------------------------------------------------

def _mf6_model_with(pkg_name, attr_name, data):
    """Mock MF6 model whose package attribute returns transient dict data."""
    array_obj = MagicMock()
    array_obj.get_data.return_value = data

    pkg = MagicMock()
    setattr(pkg, attr_name, array_obj)

    model = MagicMock()
    model.get_package.side_effect = lambda n: pkg if n == pkg_name else None
    return model, array_obj


@pytest.mark.parametrize(
    "func_name,pkg_name,attr_name,mult",
    [
        ("_set_rch_multiplier", "RCHA", "recharge", 2.0),
        ("_set_evt_multiplier", "EVTA", "rate", 3.0),
    ],
)
def test_multiplier_applies_to_all_stress_periods(
    fwd, func_name, pkg_name, attr_name, mult
):
    data = {0: np.full((3, 4), 0.01), 1: np.full((3, 4), 0.02)}
    model, array_obj = _mf6_model_with(pkg_name, attr_name, data)

    getattr(fwd, func_name)(model, "mf6", mult)

    array_obj.set_data.assert_called_once()
    written = array_obj.set_data.call_args[0][0]

    assert isinstance(written, dict), (
        "must write back a stress-period dict, not a collapsed array"
    )
    assert written[0] == pytest.approx(np.full((3, 4), 0.01 * mult))
    assert written[1] == pytest.approx(np.full((3, 4), 0.02 * mult))


@pytest.mark.parametrize(
    "func_name,pkg_name,attr_name",
    [
        ("_set_rch_multiplier", "RCHA", "recharge"),
        ("_set_evt_multiplier", "EVTA", "rate"),
    ],
)
def test_multiplier_is_actually_applied_not_swallowed(
    fwd, func_name, pkg_name, attr_name
):
    """The original bug swallowed a TypeError, leaving set_data uncalled."""
    model, array_obj = _mf6_model_with(
        pkg_name, attr_name, {0: np.ones((2, 2))}
    )

    getattr(fwd, func_name)(model, "mf6", 5.0)

    assert array_obj.set_data.called, (
        "set_data was never called — the multiplier was silently dropped"
    )
