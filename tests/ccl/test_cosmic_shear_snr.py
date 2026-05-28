"""
Tests for CosmicShearSNREvaluator.

Synthetic Gaussian n(z) and a simple nz_summary dict are used so the
test runs without any on-disk data files and completes quickly with a
coarse ell grid.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
import qp

from rail.evaluation.metrics.cosmic_shear_snr import CosmicShearSNREvaluator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ensemble(z_means: list[float], sigma: float = 0.15) -> qp.Ensemble:
    """Return a qp interp ensemble with one Gaussian n(z) row per mean."""
    z = np.linspace(0.0, 3.0, 301)
    yvals = np.array([np.exp(-0.5 * ((z - m) / sigma) ** 2) for m in z_means])
    for i in range(len(z_means)):
        yvals[i] /= np.trapezoid(yvals[i], z)
    return qp.Ensemble(qp.interp, data={"xvals": z, "yvals": yvals})


def _make_nz_summary(neff: list[float]) -> OrderedDict:
    """Return an nz_summary dict matching what tables_io.read produces."""
    return OrderedDict([
        ("nz_summary", OrderedDict([("neff", np.array(neff))]))
    ])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def four_bin_ensemble():
    return _make_ensemble([0.45, 0.75, 1.05, 1.35])


@pytest.fixture
def four_bin_summary():
    return _make_nz_summary([5.0, 5.0, 5.0, 5.0])


_STAGE_KW = dict(
    f_sky=0.01,
    sigma_e=0.26,
    ell_min=200.0,
    ell_max=3000.0,
    n_ell=5,
    ell_min_cut=300.0,
    ell_max_cut=2000.0,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_snr_positive(four_bin_ensemble, four_bin_summary, tmp_path):
    """Total SNR must be a positive finite scalar."""
    stage = CosmicShearSNREvaluator.make_stage(
        name="test_snr",
        output=str(tmp_path / "out.hdf5"),
        **_STAGE_KW,
    )
    handle = stage.evaluate(four_bin_ensemble, four_bin_summary)
    snr = float(handle.data["summary"]["total_snr"].iloc[0])

    assert np.isfinite(snr), f"SNR is not finite: {snr}"
    assert snr > 0, f"SNR is not positive: {snr}"


def test_n_bins_from_npdf(four_bin_ensemble, four_bin_summary, tmp_path):
    """n_bins is inferred from ens.npdf; output must have 10 pair rows."""
    stage = CosmicShearSNREvaluator.make_stage(
        name="test_nbins",
        output=str(tmp_path / "out.hdf5"),
        **_STAGE_KW,
    )
    handle = stage.evaluate(four_bin_ensemble, four_bin_summary)
    pairs_df = handle.data["pairs"]

    expected_pairs = 4 * (4 + 1) // 2  # 10
    assert len(pairs_df) == expected_pairs
    assert np.all(np.isfinite(pairs_df["snr_per_pair"]))
    assert np.all(pairs_df["snr_per_pair"] >= 0)


def test_larger_fsky_gives_higher_snr(four_bin_ensemble, four_bin_summary, tmp_path):
    """Increasing f_sky reduces covariance and must increase the SNR."""
    def _snr(f_sky, name):
        stage = CosmicShearSNREvaluator.make_stage(
            name=name,
            output=str(tmp_path / f"{name}.hdf5"),
            f_sky=f_sky,
            **{k: v for k, v in _STAGE_KW.items() if k != "f_sky"},
        )
        return float(stage.evaluate(four_bin_ensemble, four_bin_summary)
                     .data["summary"]["total_snr"].iloc[0])

    snr_small = _snr(0.01, "small_fsky")
    snr_large = _snr(0.10, "large_fsky")

    assert snr_large > snr_small, (
        f"Expected SNR to increase with f_sky, got {snr_small:.2f} vs {snr_large:.2f}"
    )


def test_two_bins(tmp_path):
    """Stage works for n_bins=2 (3 unique pairs)."""
    ens = _make_ensemble([0.4, 0.8])
    summary = _make_nz_summary([5.0, 5.0])

    stage = CosmicShearSNREvaluator.make_stage(
        name="test_2bins",
        output=str(tmp_path / "out_2bins.hdf5"),
        **_STAGE_KW,
    )
    handle = stage.evaluate(ens, summary)
    pairs_df = handle.data["pairs"]

    assert len(pairs_df) == 3  # (0,0), (0,1), (1,1)
    assert float(handle.data["summary"]["total_snr"].iloc[0]) > 0


def test_neff_length_mismatch(four_bin_ensemble, tmp_path):
    """Stage must raise ValueError when neff length != npdf."""
    bad_summary = _make_nz_summary([5.0, 5.0])  # 2 entries, but 4 bins
    stage = CosmicShearSNREvaluator.make_stage(
        name="test_mismatch",
        output=str(tmp_path / "out.hdf5"),
        **_STAGE_KW,
    )
    with pytest.raises(ValueError, match="n_bins mismatch"):
        stage.evaluate(four_bin_ensemble, bad_summary)
