"""
Cosmic shear signal-to-noise evaluator using CCL.

Computes the total SNR of a tomographic cosmic shear analysis from
per-bin n(z) distributions stored as a single multi-row qp ensemble
and per-bin effective number densities stored in a companion HDF5 file.
"""

from __future__ import annotations

import collections
from typing import Any

import numpy as np
import pandas as pd
import pyccl as ccl
from ceci.config import StageParameter as Param

from rail.core.data import Hdf5Handle, QPHandle
from rail.core.stage import RailStage


class CosmicShearSNREvaluator(RailStage):
    """Evaluate the total SNR of a tomographic cosmic shear analysis.

    The number of tomographic bins is inferred from the number of rows
    (``npdf``) in the input qp ensemble.

    Inputs
    ------
    input : QPHandle
        A qp ensemble with ``n_bins`` rows.  Each row is the normalised
        n(z) distribution for one tomographic bin (bin 0 first).
        The interp parameterisation is assumed; ``dist.xvals`` is the
        shared redshift grid and ``dist.yvals`` has shape
        ``(n_bins, n_z)``.

    nz_summary : Hdf5Handle
        HDF5 file that stores survey summary statistics.  Expected
        structure (as written by :func:`tables_io.write`)::

            nz_summary/
                neff   – array of shape (n_bins,), effective number
                         density [arcmin^{-2}] per tomographic bin.

    Config
    ------
    f_sky : float
        Fraction of sky covered by the survey.
    sigma_e : float
        Per-component *total* ellipticity dispersion (intrinsic +
        measurement noise combined in quadrature).
    ell_min, ell_max, n_ell : float, float, int
        Log-spaced multipole grid (default: 100–15 800 in 17 bins,
        matching the reference notebook).
    ell_min_cut, ell_max_cut : float, float
        Multipole scale cuts applied when computing the SNR
        (default: 300–1 800).
    neff_groupname : str
        Top-level group name in the nz_summary HDF5 (default
        ``"nz_summary"``).
    neff_colname : str
        Column / dataset name for neff within that group (default
        ``"neff"``).

    Outputs
    -------
    output : Hdf5Handle
        HDF5 file with two tables:

        ``pairs``
            One row per unique (i, j) pair.  Columns:
            ``pair_i``, ``pair_j``, ``snr_per_pair``,
            ``cls_ell_0`` … ``cls_ell_{n_ell-1}``.

        ``summary``
            One-row scalar results:
            ``total_snr``, ``n_bins``, ``ell_min_cut``,
            ``ell_max_cut``, ``ell_0`` … ``ell_{n_ell-1}``.
    """

    name = "CosmicShearSNREvaluator"
    config_options = RailStage.config_options.copy()
    config_options.update(
        f_sky=Param(float, 0.01, msg="Fraction of sky covered by the survey"),
        sigma_e=Param(
            float,
            0.26,
            msg="Per-component total ellipticity dispersion (intrinsic + measurement)",
        ),
        ell_min=Param(float, 100.0, msg="Minimum multipole for the C_ell grid"),
        ell_max=Param(float, 15800.0, msg="Maximum multipole for the C_ell grid"),
        n_ell=Param(int, 17, msg="Number of log-spaced multipole bins"),
        ell_min_cut=Param(float, 300.0, msg="Lower multipole scale cut for SNR"),
        ell_max_cut=Param(float, 1800.0, msg="Upper multipole scale cut for SNR"),
        neff_groupname=Param(
            str, "nz_summary", msg="HDF5 group name in the nz_summary file"
        ),
        neff_colname=Param(
            str, "neff", msg="Column name for neff within the nz_summary group"
        ),
    )

    inputs = [
        ("input", QPHandle),        # multi-row n(z) ensemble, one row per bin
        ("nz_summary", Hdf5Handle), # effective number densities
    ]
    outputs = [("output", Hdf5Handle)]

    def evaluate(self, nz_ensemble: Any, nz_summary_data: Any) -> Hdf5Handle:
        """Compute the cosmic shear SNR.

        Parameters
        ----------
        nz_ensemble : qp.Ensemble
            Multi-row qp ensemble (``npdf == n_bins``).
        nz_summary_data : dict-like
            Data as returned by ``tables_io.read`` for the nz_summary
            HDF5 file (an OrderedDict with group ``neff_groupname``
            containing column ``neff_colname``).

        Returns
        -------
        Hdf5Handle
            Handle to the output metrics file.
        """
        self.set_data("input", nz_ensemble)
        self.set_data("nz_summary", nz_summary_data)
        self.run()
        self.finalize()
        return self.get_handle("output")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_nz(self, ens: Any) -> tuple[np.ndarray, list[np.ndarray]]:
        """Extract and normalise the per-bin n(z) from a qp ensemble.

        Parameters
        ----------
        ens : qp.Ensemble
            Multi-row interp ensemble; ``dist.xvals`` shape ``(n_z,)``,
            ``dist.yvals`` shape ``(n_bins, n_z)``.

        Returns
        -------
        z_grid : ndarray, shape (n_z,)
        nz_list : list of ndarray, length n_bins
            Normalised n(z) for each bin.
        """
        z_grid = np.asarray(ens.dist.xvals, dtype=float)
        yvals = np.asarray(ens.dist.yvals, dtype=float)
        n_bins = ens.npdf

        nz_list: list[np.ndarray] = []
        for b in range(n_bins):
            pz = np.nan_to_num(yvals[b], nan=0.0, posinf=0.0)
            norm = np.trapezoid(pz, z_grid)
            if norm > 0.0:
                pz = pz / norm
            nz_list.append(pz)
        return z_grid, nz_list

    def _read_neff(self, nz_summary_data: Any, n_bins: int) -> np.ndarray:
        """Extract neff from the nz_summary data dict.

        Parameters
        ----------
        nz_summary_data : dict-like
            As returned by ``tables_io.read``.
        n_bins : int
            Expected number of bins (used for validation).

        Returns
        -------
        neff : ndarray, shape (n_bins,)  [arcmin^{-2}]
        """
        grp = nz_summary_data[self.config.neff_groupname]
        neff = np.asarray(grp[self.config.neff_colname], dtype=float)
        if len(neff) != n_bins:
            raise ValueError(
                f"neff has {len(neff)} entries but the qp ensemble has "
                f"{n_bins} rows (n_bins mismatch)."
            )
        return neff

    @staticmethod
    def _gaussian_covariance(
        cls: np.ndarray,
        noise: np.ndarray,
        pairs: list[tuple[int, int]],
        ells: np.ndarray,
        f_sky: float,
        n_bins: int,
    ) -> np.ndarray:
        """Build the Gaussian covariance matrix of the C_ell data vector.

        The data vector ordering is [C_ell^{pair_0}, C_ell^{pair_1}, …],
        each block having ``n_ell`` entries.

        The diagonal Gaussian covariance element is::

            Cov(C_ℓ^{ij}, C_ℓ^{kl}) = δ_{ℓℓ'} / [(2ℓ+1) Δℓ f_sky]
                × [C_ℓ_tot^{ik} C_ℓ_tot^{jl} + C_ℓ_tot^{il} C_ℓ_tot^{jk}]

        where ``C_ℓ_tot^{ij} = C_ℓ^{ij} + δ_{ij} σ_e² / n_i``.
        """
        n_pairs, n_ell = cls.shape

        # Total (signal + noise) C_ell tensor: shape (n_bins, n_bins, n_ell)
        cls_tot = np.zeros((n_bins, n_bins, n_ell))
        for p_idx, (i, j) in enumerate(pairs):
            cls_tot[i, j] = cls[p_idx]
            cls_tot[j, i] = cls[p_idx]
        for i in range(n_bins):
            cls_tot[i, i] += noise[i]

        # Effective linear bin width: Δℓ = ℓ × Δ(ln ℓ)
        delta_ell = ells * np.gradient(np.log(ells))

        cov = np.zeros((n_pairs * n_ell, n_pairs * n_ell))
        for l_idx in range(n_ell):
            n_modes = (2.0 * ells[l_idx] + 1.0) * delta_ell[l_idx] * f_sky
            for p1, (i, j) in enumerate(pairs):
                for p2, (k, l) in enumerate(pairs):
                    val = (
                        cls_tot[i, k, l_idx] * cls_tot[j, l, l_idx]
                        + cls_tot[i, l, l_idx] * cls_tot[j, k, l_idx]
                    ) / n_modes
                    cov[p1 * n_ell + l_idx, p2 * n_ell + l_idx] = val
        return cov

    # ------------------------------------------------------------------
    # RailStage interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        n_ell = self.config.n_ell

        # --- n(z) and number of bins ---
        ens = self.get_data("input")
        z_grid, nz_list = self._read_nz(ens)
        n_bins = ens.npdf

        # --- neff ---
        nz_summary_data = self.get_data("nz_summary")
        neff = self._read_neff(nz_summary_data, n_bins)

        # --- multipole grid ---
        ells = np.geomspace(self.config.ell_min, self.config.ell_max, n_ell)

        # --- CCL cosmology and weak-lensing tracers ---
        cosmo = ccl.CosmologyVanillaLCDM()
        tracers = [
            ccl.WeakLensingTracer(cosmo, dndz=(z_grid, nz))
            for nz in nz_list
        ]

        # --- signal angular power spectra ---
        pairs = [(i, j) for i in range(n_bins) for j in range(i, n_bins)]
        n_pairs = len(pairs)
        cls = np.zeros((n_pairs, n_ell))
        for p_idx, (i, j) in enumerate(pairs):
            cls[p_idx] = ccl.angular_cl(cosmo, tracers[i], tracers[j], ells)

        # --- shape-noise power spectra ---
        arcmin2_per_sr = (180.0 * 60.0 / np.pi) ** 2
        neff_sr = neff * arcmin2_per_sr
        noise = self.config.sigma_e ** 2 / neff_sr  # shape (n_bins,)

        # --- Gaussian covariance ---
        cov = self._gaussian_covariance(
            cls, noise, pairs, ells, self.config.f_sky, n_bins
        )

        # --- SNR within scale cuts ---
        ell_mask = (ells >= self.config.ell_min_cut) & (ells <= self.config.ell_max_cut)
        ell_idx = np.where(ell_mask)[0]
        sel_idx = np.array(
            [p * n_ell + l for p in range(n_pairs) for l in ell_idx]
        )

        d_sel = cls.ravel()[sel_idx]
        cov_sel = cov[np.ix_(sel_idx, sel_idx)]
        snr = float(np.sqrt(max(float(d_sel @ np.linalg.inv(cov_sel) @ d_sel), 0.0)))

        # --- per-pair SNR (diagonal block approximation) ---
        n_sel_ell = len(ell_idx)
        snr_per_pair = np.zeros(n_pairs)
        for p in range(n_pairs):
            sl = slice(p * n_sel_ell, (p + 1) * n_sel_ell)
            dp = d_sel[sl]
            covp = cov_sel[sl, sl]
            snr_per_pair[p] = float(
                np.sqrt(max(float(dp @ np.linalg.inv(covp) @ dp), 0.0))
            )

        # --- build output tables ---
        pair_data: dict[str, Any] = {
            "pair_i": np.array([i for i, j in pairs]),
            "pair_j": np.array([j for i, j in pairs]),
            "snr_per_pair": snr_per_pair,
        }
        for l_idx in range(n_ell):
            pair_data[f"cls_ell_{l_idx}"] = cls[:, l_idx]

        summary_data: dict[str, Any] = {
            "total_snr": np.array([snr]),
            "n_bins": np.array([n_bins]),
            "ell_min_cut": np.array([self.config.ell_min_cut]),
            "ell_max_cut": np.array([self.config.ell_max_cut]),
        }
        for l_idx in range(n_ell):
            summary_data[f"ell_{l_idx}"] = np.array([ells[l_idx]])

        self.add_data(
            "output",
            collections.OrderedDict([
                ("pairs", pd.DataFrame(pair_data)),
                ("summary", pd.DataFrame(summary_data)),
            ]),
        )
