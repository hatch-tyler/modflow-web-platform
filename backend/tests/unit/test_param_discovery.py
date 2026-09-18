"""Tests for PEST parameter discovery — RCH/EVT data extraction and progress callbacks."""

import numpy as np
import pytest
from unittest.mock import MagicMock, call

from app.services.mesh import _get_rch_data, _get_evt_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mf6_model(package_name, attr_name, data, has_package=True):
    """Create a mock MF6 model with a given package and attribute."""
    model = MagicMock()
    model.__class__ = type("MockMF6", (), {"__module__": "flopy.mf6.mfmodel"})

    if not has_package:
        model.get_package.return_value = None
        return model

    pkg = MagicMock()
    getattr(pkg, attr_name).get_data = MagicMock(return_value=data)

    def _get_pkg(name):
        if name in package_name if isinstance(package_name, (list, tuple)) else name == package_name:
            return pkg
        return None

    model.get_package.side_effect = _get_pkg
    return model, pkg


def _make_mf2005_model():
    """Create a mock MF2005 model."""
    model = MagicMock()
    model.__class__ = type("MockMF2005", (), {"__module__": "flopy.modflow.mf"})
    return model


# ===========================================================================
# _get_rch_data — MF6
# ===========================================================================

class TestGetRchDataMF6:
    """MF6 recharge parameter extraction."""

    def test_calls_get_data_with_period_0(self):
        """Critical: get_data must be called with period 0, not without args."""
        arr = np.array([[1e-4, 2e-4], [3e-4, 4e-4]])
        model, pkg = _make_mf6_model(("RCHA", "RCH"), "recharge", arr)

        _get_rch_data(model, is_mf6=True, is_usg=False)

        pkg.recharge.get_data.assert_called_once_with(0)

    def test_flat_array_return(self):
        """get_data(0) returning a plain numpy array."""
        arr = np.array([1e-4, 2e-4, 3e-4])
        model, _ = _make_mf6_model(("RCHA", "RCH"), "recharge", arr)

        result = _get_rch_data(model, is_mf6=True, is_usg=False)

        assert result is not None
        assert result["count"] == 3
        assert result["stats"]["min"] == pytest.approx(1e-4, rel=1e-3)
        assert result["stats"]["max"] == pytest.approx(3e-4, rel=1e-3)

    def test_dict_keyed_by_layer(self):
        """get_data(0) returning {layer: array} dict (per-layer RCHA)."""
        data = {0: np.array([1e-4, 2e-4])}
        model, _ = _make_mf6_model(("RCHA", "RCH"), "recharge", data)

        result = _get_rch_data(model, is_mf6=True, is_usg=False)

        assert result is not None
        assert result["count"] == 2

    def test_no_package_returns_none(self):
        """No RCH package → None."""
        model = MagicMock()
        model.__class__ = type("M", (), {"__module__": "flopy.mf6.mfmodel"})
        model.get_package.return_value = None

        assert _get_rch_data(model, is_mf6=True, is_usg=False) is None

    def test_get_data_returns_none(self):
        """get_data(0) returns None → None."""
        model, _ = _make_mf6_model(("RCHA", "RCH"), "recharge", None)

        assert _get_rch_data(model, is_mf6=True, is_usg=False) is None

    def test_all_zero_values_returns_none_stats(self):
        """Array of all zeros → valid stats are None but result still returned."""
        arr = np.array([0.0, 0.0, 0.0])
        model, _ = _make_mf6_model(("RCHA", "RCH"), "recharge", arr)

        result = _get_rch_data(model, is_mf6=True, is_usg=False)
        assert result is not None
        assert result["count"] == 3
        assert result["stats"]["mean"] is None


# ===========================================================================
# _get_evt_data — MF6
# ===========================================================================

class TestGetEvtDataMF6:
    """MF6 evapotranspiration parameter extraction."""

    def test_calls_get_data_with_period_0(self):
        """Critical: get_data must be called with period 0."""
        arr = np.array([[1e-3, 2e-3]])
        model, pkg = _make_mf6_model(("EVTA", "EVT"), "rate", arr)

        _get_evt_data(model, is_mf6=True, is_usg=False)

        pkg.rate.get_data.assert_called_once_with(0)

    def test_flat_array_return(self):
        arr = np.array([1e-3, 2e-3, 3e-3])
        model, _ = _make_mf6_model(("EVTA", "EVT"), "rate", arr)

        result = _get_evt_data(model, is_mf6=True, is_usg=False)

        assert result is not None
        assert result["count"] == 3

    def test_dict_keyed_by_layer(self):
        data = {0: np.array([1e-3, 2e-3])}
        model, _ = _make_mf6_model(("EVTA", "EVT"), "rate", data)

        result = _get_evt_data(model, is_mf6=True, is_usg=False)

        assert result is not None
        assert result["count"] == 2

    def test_no_package_returns_none(self):
        model = MagicMock()
        model.__class__ = type("M", (), {"__module__": "flopy.mf6.mfmodel"})
        model.get_package.return_value = None

        assert _get_evt_data(model, is_mf6=True, is_usg=False) is None

    def test_get_data_returns_none(self):
        model, _ = _make_mf6_model(("EVTA", "EVT"), "rate", None)

        assert _get_evt_data(model, is_mf6=True, is_usg=False) is None


# ===========================================================================
# _get_rch_data / _get_evt_data — MF2005
# ===========================================================================

class TestGetRchDataMF2005:
    """MF2005 recharge path."""

    def test_rch_mf2005_with_data(self):
        model = _make_mf2005_model()
        model.rch.rech.__getitem__ = MagicMock(
            return_value=MagicMock(array=np.array([[1e-4, 2e-4]]))
        )

        result = _get_rch_data(model, is_mf6=False, is_usg=False)

        assert result is not None
        assert result["count"] == 2

    def test_rch_mf2005_no_package(self):
        model = _make_mf2005_model()
        model.rch = None

        assert _get_rch_data(model, is_mf6=False, is_usg=False) is None


class TestGetEvtDataMF2005:
    """MF2005 evapotranspiration path."""

    def test_evt_mf2005_with_data(self):
        model = _make_mf2005_model()
        model.evt.evtr.__getitem__ = MagicMock(
            return_value=MagicMock(array=np.array([[1e-3, 2e-3]]))
        )

        result = _get_evt_data(model, is_mf6=False, is_usg=False)

        assert result is not None
        assert result["count"] == 2

    def test_evt_mf2005_no_package(self):
        model = _make_mf2005_model()
        model.evt = None

        assert _get_evt_data(model, is_mf6=False, is_usg=False) is None


# ===========================================================================
# discover_parameters progress callback
# ===========================================================================

class TestDiscoverParametersProgress:
    """Verify progress_callback is invoked during discover_parameters."""

    def test_progress_callback_invoked(self, tmp_path):
        """Progress callback should be called at key checkpoints."""
        from unittest.mock import patch

        callback = MagicMock()

        # Mock load_model_from_directory to return a fake model
        mock_model = MagicMock()
        mock_model.__class__ = type("M", (), {"__module__": "flopy.mf6.mfmodel"})

        with patch("app.services.pest_setup.load_model_from_directory", return_value=mock_model), \
             patch("app.services.pest_setup.get_array_data", return_value=None), \
             patch("app.services.pest_setup.get_list_package_data", return_value=None):
            from app.services.pest_setup import discover_parameters

            discover_parameters(tmp_path, progress_callback=callback)

        # Should have been called with model loaded (55%) and scanning list packages (70%)
        pcts = [c.args[0] for c in callback.call_args_list]
        assert 55 in pcts
        assert 70 in pcts

    def test_no_callback_does_not_error(self, tmp_path):
        """Passing no callback should not raise."""
        from unittest.mock import patch

        mock_model = MagicMock()
        mock_model.__class__ = type("M", (), {"__module__": "flopy.mf6.mfmodel"})

        with patch("app.services.pest_setup.load_model_from_directory", return_value=mock_model), \
             patch("app.services.pest_setup.get_array_data", return_value=None), \
             patch("app.services.pest_setup.get_list_package_data", return_value=None):
            from app.services.pest_setup import discover_parameters

            result = discover_parameters(tmp_path)

        assert result == []
