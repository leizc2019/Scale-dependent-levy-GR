import numpy as np
from scipy.stats import linregress, t


class LevyP0Analysis:
    """
    Physically consistent p0–Δt analysis for Lévy / truncated Lévy diagnostics
    """

    def __init__(
        self,
        eps_mode="fixed_sigma1",   # "fixed_sigma1" | "fixed_quantile"
        eps_c=0.05,
        eps_quantile=0.05,
        min_points=4
    ):
        self.eps_mode = eps_mode
        self.eps_c = eps_c
        self.eps_quantile = eps_quantile
        self.min_points = min_points
        self.eps_ = None

    # --------------------------------------------------
    # Step 1: determine epsilon from Δt = 1
    # --------------------------------------------------
    def fit_epsilon(self, delta_dt1):
        delta_dt1 = delta_dt1[np.isfinite(delta_dt1)]

        if self.eps_mode == "fixed_sigma1":
            sigma1 = np.std(delta_dt1)
            self.eps_ = self.eps_c * sigma1

        elif self.eps_mode == "fixed_quantile":
            self.eps_ = np.quantile(np.abs(delta_dt1), self.eps_quantile)

        else:
            raise ValueError("Unknown eps_mode")

        if self.eps_ <= 0:
            raise ValueError("Invalid epsilon")

        return self.eps_

    # --------------------------------------------------
    # Step 2: compute p0 components
    # --------------------------------------------------
    def compute_p0_components(self, delta):
        delta = delta[np.isfinite(delta)]
        n = len(delta)

        if n == 0:
            return np.nan, np.nan, np.nan

        p0_total = np.mean(np.abs(delta) < self.eps_)
        p0_zero = np.mean(delta == 0)
        p0_cont = np.mean((np.abs(delta) < self.eps_) & (delta != 0))

        return p0_total, p0_zero, p0_cont

    # --------------------------------------------------
    # Step 3: fit Lévy scaling using p0_cont
    def fit_levy_p0(self, dt_list, p0_cont_list, ci_level=0.95):
        dt = np.asarray(dt_list)
        p0 = np.asarray(p0_cont_list)

        mask = (p0 > 0) & np.isfinite(p0)
        dt = dt[mask]
        p0 = p0[mask]

        n = len(dt)
        if n < self.min_points:
            return None

        log_dt = np.log(dt)
        log_p0 = np.log(p0)

        slope, intercept, r, p, stderr = linregress(log_dt, log_p0)

        # slope 必须为负才有物理意义
        if slope >= 0:
            return {
                "alpha": np.nan,
                "alpha_ci": (np.nan, np.nan),
                "slope": slope,
                "r2": r ** 2,
                "n_points": n
            }

        # alpha point estimate
        alpha_hat = -1.0 / slope

        # t-based CI for slope
        dof = n - 2
        tval = t.ppf(1 - (1 - ci_level) / 2, dof)

        slope_low = slope - tval * stderr
        slope_high = slope + tval * stderr

        # 映射到 alpha（注意不对称性）
        alpha_ci = (
            -1.0 / slope_high,
            -1.0 / slope_low
        )

        # 保证 CI 顺序
        alpha_ci = tuple(sorted(alpha_ci))

        return {
            "alpha": alpha_hat,
            "alpha_ci": alpha_ci,
            "slope": slope,
            "r2": r ** 2,
            "n_points": n
        }
