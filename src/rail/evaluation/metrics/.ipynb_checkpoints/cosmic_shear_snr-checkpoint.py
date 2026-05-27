"""
Cosmic shear signal-to-noise evaluator using CCL.

Computes the total SNR of a tomographic cosmic shear analysis from
per-bin n(z) distributions stored as qp ensembles.  The angular power
spectra are computed via CCL's Limber approximation, and the Gaussian
covariance matrix is assembled from the total (signal + shape-noise)
power spectra.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyccl as ccl
from ceci.config import StageParameter as Param

from rail.core.data import Hdf5Handle, QPHandle
from rail.core.stage import RailStage


class CosmicShearSNREvaluator(RailStage):
    """Evaluate the total SNR of a tomographic cosmic shear analysis.

    Inputs
    ------
    input_0 … input_3 : QPHandle
        Stacked n(z) ensemble for each tomographic bin (bin 0 first).
        Only the first ``n_bins`` inputs are read.

    Config
    ------
    n_bins : int
        Number of tomographic bins (1–4; default 4).
    neff : list[float]
        Effective galaxy number density [arcmin^{-2}] for each bin.
    f_sky : float
        Sky fraction covered by the survey.
    sigma_e : float
        Per-component *total* ellipticity dispersion (intrinsic + measurement).
    ell_min, ell_max, n_ell : float, float, int
        Log-spaced multipole grid (default: 100–15 800 in 17 bins).
    ell_min_cut, ell_max_cut : float, float
        Multipole scale cuts used when computing the SNR (default: 300–1 800).
    z_eval_min, z_eval_max, n_z_eval : float, float, int
        Redshift grid for evaluating n(z) from the qp ensemble.

    Outputs
    -------
    output : Hdf5Handle
        HDF5 file with keys:
        ``snr``          – total SNR (scalar),
        ``snr_per_pair`` – per-pair SNR in diagonal-block approximation,
        ``pair_i``, ``pair_j`` – bin indices for each pair (0-based),
        ``cls``          – signal C_ℓ array, shape (n_pairs, n_ell),
        ``ells``         – multipole centres.
    """

    name = "CosmicShearSNREvaluator"
    config_options = RailStage.config_options.copy()
    config_options.update(
        n_bins=Param(int, 4, msg="Number of tomographic bins (1–4)"),
        neff=Param(
            list,
            [5.0, 5.0, 5.0, 5.0],
            msg="Effective galaxy number density [arcmin^-2] per bin",
        ),
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
        z_eval_min=Param(float, 0.005, msg="Minimum redshift for n(z) evaluation"),
        z_eval_max=Param(float, 3.0, msg="Maximum redshift for n(z) evaluation"),
        n_z_eval=Param(int, 300, msg="Number of redshift grid points for n(z)"),
    )

    # Up to 4 tomographic bins; only the first n_bins inputs are used.
    inputs = [
        ("input_0", QPHandle),
        ("input_1", QPHandle),
        ("input_2", QPHandle),
        ("input_3", QPHandle),
    ]
    outputs = [("output", Hdf5Handle)]

    def evaluate(
        self,
        ens_0: Any,
        ens_1: Any | None = None,
        ens_2: Any | None = None,
        ens_3: Any | None = None,
    ) -> Hdf5Handle:
        """Compute the cosmic shear SNR.

        Parameters
        ----------
        ens_0 … ens_3 : qp.Ensemble
            n(z) ensembles for bins 0–3.  Pass ``None`` for bins beyond
            ``n_bins``.

        Returns
        -------
        Hdf5Handle
            Handle to the output metrics file.
        """
        n_bins = self.config.n_bins
        ensembles = [ens_0, ens_1, ens_2, ens_3]
        for i in range(n_bins):
            if ensembles[i] is None:
                raise ValueError(f"n_bins={n_bins} but ens_{i} is None")
            self.set_data(f"input_{i}", ensembles[i])
        self.run()
        self.finalize()
        return self.get_handle("output")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_nz(self, z_grid: np.ndarray) -> list[np.ndarray]:
        """Read and normalise n(z) for each bin from the data store."""
        n_bins = self.config.n_bins
        nz_list = []
        for b in range(n_bins):
            ens = self.get_data(f"input_{b}")
            pz = np.asarray(ens.pdf(z_grid), dtype=float)
            pz = np.nan_to_num(pz, nan=0.0, posinf=0.0)
            norm = np.trapezoid(pz, z_grid)
            if norm > 0.0:
                pz /= norm
            nz_list.append(pz)
        return nz_list

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

        The data vector is ordered as:
            [C_ell^{pair_0}, C_ell^{pair_1}, …]
        where each block has ``n_ell`` entries.

        The diagonal Gaussian covariance element is:

            Cov(C_ℓ^{ij}, C_ℓ^{kl}) = δ_{ℓℓ'} / [(2ℓ+1) Δℓ f_sky]
                × [C_ℓ_tot^{ik} C_ℓ_tot^{jl} + C_ℓ_tot^{il} C_ℓ_tot^{jk}]
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
        n_bins = self.config.n_bins
        n_ell = self.config.n_ell

        # --- n(z) ---
        z_grid = np.linspace(self.config.z_eval_min, self.config.z_eval_max, self.config.n_z_eval)
        nz_list = self._read_nz(z_grid)

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
        neff = np.asarray(self.config.neff[:n_bins], dtype=float)
        neff_sr = neff * arcmin2_per_sr
        noise = self.config.sigma_e ** 2 / neff_sr  # (n_bins,)

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

        # per-pair SNR (diagonal block approximation)
        n_sel_ell = len(ell_idx)
        snr_per_pair = np.zeros(n_pairs)
        for p in range(n_pairs):
            sl = slice(p * n_sel_ell, (p + 1) * n_sel_ell)
            dp = d_sel[sl]
            covp = cov_sel[sl, sl]
            snr_per_pair[p] = float(
                np.sqrt(max(float(dp @ np.linalg.inv(covp) @ dp), 0.0))
            )

        pair_i = np.array([i for i, j in pairs])
        pair_j = np.array([j for i, j in pairs])

        self.add_data(
            "output",
            {
                "snr": np.array([snr]),
                "snr_per_pair": snr_per_pair,
                "pair_i": pair_i,
                "pair_j": pair_j,
                "cls": cls,
                "ells": ells,
                "ell_min_cut": np.array([self.config.ell_min_cut]),
                "ell_max_cut": np.array([self.config.ell_max_cut]),
            },
        )
