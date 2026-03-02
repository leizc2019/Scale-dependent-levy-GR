import numpy as np
from scipy.stats import linregress, t
import matplotlib.pyplot as plt
from pathlib import Path


class LevyMSDAnalysis:
    """
    Unified p0–Δt and MSD–Δt analysis
    """

    def __init__(
        self,
        min_points=4,
        msd_fit_range=None
    ):
        """
        msd_fit_range: tuple or None
            e.g. (1, 100) -> only fit dt in this range
        """
        self.min_points = min_points
        self.msd_fit_range = msd_fit_range

    # --------------------------------------------------
    # MSD computation
    # --------------------------------------------------
    @staticmethod
    def compute_msd(delta):
        delta = delta[np.isfinite(delta)]
        if len(delta) == 0:
            return np.nan
        delta = delta - np.mean(delta)
        return np.mean(delta ** 2)

    # --------------------------------------------------
    # MSD scaling fit
    # --------------------------------------------------
    def fit_msd(self, dt_list, msd_list, ci_level=0.95):
        dt = np.asarray(dt_list)
        msd = np.asarray(msd_list)

        mask = (msd > 0) & np.isfinite(msd)
        dt = dt[mask]
        msd = msd[mask]

        if self.msd_fit_range is not None:
            dt_min, dt_max = self.msd_fit_range
            mask = (dt >= dt_min) & (dt <= dt_max)
            dt = dt[mask]
            msd = msd[mask]

        n = len(dt)
        if n < self.min_points:
            return None

        log_dt = np.log(dt)
        log_msd = np.log(msd)

        slope, intercept, r, p, stderr = linregress(log_dt, log_msd)

        # t-based confidence interval for γ
        dof = n - 2
        alpha = 1 - ci_level
        tval = t.ppf(1 - alpha / 2, dof)

        gamma_ci = (
            slope - tval * stderr,
            slope + tval * stderr
        )

        return {
            "msd_exp": slope,  # γ
            "msd_exp_ci": gamma_ci,  # 置信区间
            "intercept": intercept,
            "r2": r ** 2,
            "n_points": n
        }

    def fit_msd_running(self, dt_list, msd_list, ci_level=0.95):
        """
        Cumulative MSD scaling:
        fit [dt[0], ..., dt[i]] for i >= 1
        """

        dt = np.asarray(dt_list)
        msd = np.asarray(msd_list)

        mask = (msd > 0) & np.isfinite(msd)
        dt = dt[mask]
        msd = msd[mask]

        if self.msd_fit_range is not None:
            dt_min, dt_max = self.msd_fit_range
            mask = (dt >= dt_min) & (dt <= dt_max)
            dt = dt[mask]
            msd = msd[mask]

        results = []

        for i in range(1, len(dt)):  # 从第二个点开始
            dt_sub = dt[: i + 1]
            msd_sub = msd[: i + 1]

            if len(dt_sub) < self.min_points:
                continue

            log_dt = np.log(dt_sub)
            log_msd = np.log(msd_sub)

            slope, intercept, r, p, stderr = linregress(log_dt, log_msd)

            dof = len(dt_sub) - 2
            alpha = 1 - ci_level
            tval = t.ppf(1 - alpha / 2, dof)

            gamma_ci = (
                slope - tval * stderr,
                slope + tval * stderr
            )

            results.append({
                "dt_max": dt_sub[-1],     # 当前拟合的最大 Δt
                "msd_exp": slope,         # β
                "msd_exp_ci": gamma_ci,
                "r2": r ** 2,
                "n_points": len(dt_sub)
            })

        return results

