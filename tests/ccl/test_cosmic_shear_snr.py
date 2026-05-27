"""
Tests for CosmicShearSNREvaluator.

Synthetic n(z) Gaussians are used so the test runs without any data files
and completes quickly with a coarse ell grid.
"""

import numpy as np
import pytest
import qp

from rail.evaluation.metrics.cosmic_shear_snr import CosmicShearSNREvaluator


def _make_gaussian_nz(z_mean: float, z_sigma: float = 0.15, n_z: int = 200) -> qp.Ensemble:
    """Return a qp interp ensemble with a Gaussian n(z)."""
    z = np.linspace(0.01, 3.0, n_z)
    nz = np.exp(-0.5 * ((z - z_mean) / z_sigma) ** 2)
    nz /= np.trapezoid(nz, z)
    return qp.Ensemble(qp.interp, data={"xvals": z, "yvals": nz[np.newaxis, :]})


@pytest.fixture
def four_bins():
    """Four Gaussian n(z) ensembles at z = 0.3, 0.5, 0.7, 0.9."""
    return [_make_gaussian_nz(z_mean) for z_mean in [0.3, 0.5, 0.7, 0.9]]


_STAGE_KW = dict(
    n_bins=4,
    neff=[5.0, 5.0, 5.0, 5.0],
    f_sky=0.01,
    sigma_e=0.26,
    ell_min=200.0,
    ell_max=3000.0,
    n_ell=5,
    ell_min_cut=300.0,
    ell_max_cut=2000.0,
)


def test_snr_positive(four_bins, tmp_path):
    """SNR must be a positive finite scalar."""
    stage = CosmicShearSNREvaluator.make_stage(
        name="test_snr",
        output=str(tmp_path / "test_snr_output.hdf5"),
        **_STAGE_KW,
    )
    result_handle = stage.evaluate(*four_bins)
    summary = result_handle.data["summary"]

    snr = float(summary["total_snr"].iloc[0])
    assert np.isfinite(snr), f"SNR is not finite: {snr}"
    assert snr > 0, f"SNR is not positive: {snr}"


def test_snr_per_pair_shape(four_bins, tmp_path):
    """snr_per_pair should have one entry per unique bin pair."""
    n_bins = 4
    expected_pairs = n_bins * (n_bins + 1) // 2  # 10 for 4 bins

    stage = CosmicShearSNREvaluator.make_stage(
        name="test_snr_pairs",
        output=str(tmp_path / "test_snr_pairs_output.hdf5"),
        **_STAGE_KW,
    )
    result_handle = stage.evaluate(*four_bins)
    pairs_df = result_handle.data["pairs"]

    assert len(pairs_df) == expected_pairs
    assert "snr_per_pair" in pairs_df.columns
    assert np.all(np.isfinite(pairs_df["snr_per_pair"]))
    assert np.all(pairs_df["snr_per_pair"] >= 0)


def test_larger_fsky_gives_higher_snr(four_bins, tmp_path):
    """Increasing f_sky reduces covariance and must increase the SNR."""
    common = dict(
        n_bins=4,
        neff=[5.0, 5.0, 5.0, 5.0],
        sigma_e=0.26,
        ell_min=200.0,
        ell_max=3000.0,
        n_ell=5,
        ell_min_cut=300.0,
        ell_max_cut=2000.0,
    )

    stage_small = CosmicShearSNREvaluator.make_stage(
        name="snr_small_fsky",
        f_sky=0.01,
        output=str(tmp_path / "snr_small.hdf5"),
        **common,
    )
    stage_large = CosmicShearSNREvaluator.make_stage(
        name="snr_large_fsky",
        f_sky=0.10,
        output=str(tmp_path / "snr_large.hdf5"),
        **common,
    )

    snr_small = float(stage_small.evaluate(*four_bins).data["summary"]["total_snr"].iloc[0])
    snr_large = float(stage_large.evaluate(*four_bins).data["summary"]["total_snr"].iloc[0])

    assert snr_large > snr_small, (
        f"Expected SNR to increase with f_sky, got {snr_small:.2f} vs {snr_large:.2f}"
    )


def test_two_bins(tmp_path):
    """Stage should work correctly for n_bins=2."""
    bins = [_make_gaussian_nz(0.4), _make_gaussian_nz(0.8)]

    stage = CosmicShearSNREvaluator.make_stage(
        name="test_snr_2bins",
        n_bins=2,
        neff=[5.0, 5.0],
        f_sky=0.01,
        sigma_e=0.26,
        ell_min=200.0,
        ell_max=3000.0,
        n_ell=5,
        ell_min_cut=300.0,
        ell_max_cut=2000.0,
        output=str(tmp_path / "test_snr_2bins_output.hdf5"),
    )
    result_handle = stage.evaluate(*bins)
    pairs_df = result_handle.data["pairs"]

    assert len(pairs_df) == 3  # (0,0), (0,1), (1,1)
    snr = float(result_handle.data["summary"]["total_snr"].iloc[0])
    assert snr > 0
