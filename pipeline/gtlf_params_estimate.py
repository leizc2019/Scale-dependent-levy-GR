from joblib import Parallel, delayed
from scipy.integrate import cumulative_trapezoid
from scipy.stats import levy_stable, norm, skew
from configs.config import STATS_RESULTS, INTERMEDIATE_DIR, INCREMENTS
from pathlib import Path
import pandas as pd
import numpy as np
from utils.levy_gamma_estimate import compute_cq
from utils.estimate_gtlf_parallel import estimate_gtlf_parameters_parallel


def _invalid(x):
    return (not np.isfinite(x)) or (x <= 0)


def find_tlf_cutoff_ccdf(inc_values, gamma, alpha, dt, k=10):
    inc = inc_values - np.mean(inc_values)
    inc = np.abs(inc)
    inc = inc[inc > 0]

    # ===== 经验 CCDF =====
    x_emp = np.sort(inc)
    ccdf_emp = 1.0 - np.arange(1, len(x_emp) + 1) / len(x_emp)

    # ===== 理论 CCDF（log grid）=====
    x_grid = np.logspace(np.log10(x_emp.min()), np.log10(x_emp.max()), 1000)

    scale = (gamma * dt) ** (1.0 / alpha)
    pdf_levy = levy_stable.pdf(
        x_grid, alpha=alpha, beta=0, loc=0, scale=scale
    )

    ccdf_levy = cumulative_trapezoid(
        pdf_levy[::-1],
        x_grid[::-1],
        initial=0
    )[::-1]
    ccdf_levy /= ccdf_levy[0]

    # ===== 插值到经验点 =====
    ccdf_levy_interp = np.interp(x_emp, x_grid, ccdf_levy)

    eps = 1e-12
    log_ratio = np.log(ccdf_emp + eps) - np.log(ccdf_levy_interp + eps)

    thr = np.log(0.75)  # 25% 系统性偏离
    for i in range(len(log_ratio) - k):
        window = log_ratio[i:i + k]
        if (
                np.mean(window) < thr
                and np.sum(np.diff(window) < 0) >= 0.7 * (k - 1)
        ):
            return x_emp[i]

    return None


def refine_gtlf_params(
    inc_values,
    dt,
    q_list,
    q_core=0.2,
    alpha_init=1.0,
    half_width=0.2,
    n_alpha_local=200,
):
    # --------------------------------------------------
    # Stage 0：快速局部估计
    # --------------------------------------------------
    alpha0, gamma0, gamma_std0 = estimate_levy_params(
        inc_values,
        dt,
        q_list,
        q_core=q_core,
        alpha_init=alpha_init,
    )

    if alpha0 is None:
        return None, None, None, None

    # --------------------------------------------------
    # Stage I：α 局部加密扫描 → CI
    # --------------------------------------------------
    alpha, alpha_ci, gamma, gamma_std_log = alpha_scan(
        inc_values, q_list, dt, q_core=q_core,
        alpha_center=alpha0,
        half_width=half_width,
        n_alpha=n_alpha_local,
    )

    if alpha is None:
        # fallback：至少返回点估计
        return alpha0, None, gamma0, gamma_std0

    return alpha, alpha_ci, gamma, gamma_std_log


def gamma_computed(inc, q_list, dt, alpha,):
    if alpha <= 0 or alpha >= 2:
        return None, None

    gamma_rows_dt = []

    for q in q_list:
        # 经验分位数（绝对值）
        q_obs = np.quantile(np.abs(inc), q)
        c_q = compute_cq(alpha, cutoff=None, q=q)
        if c_q <= 0 or q_obs <= 0:
            continue
        gamma_q = (q_obs / c_q) ** alpha / dt
        gamma_rows_dt.append({
            "q": q,
            "gamma_q": gamma_q
        })

    gamma_vals = np.array([r["gamma_q"] for r in gamma_rows_dt])
    if len(gamma_vals) < 3 or np.any(gamma_vals <= 0) or not np.all(np.isfinite(gamma_vals)):
        return None, None

    weights = np.exp(-((gamma_vals - 0.5) / 0.2) ** 2)
    weights /= weights.sum()

    log_gamma = np.sum(weights * np.log(gamma_vals))
    gamma_mean = np.exp(log_gamma)
    gamma_std_log = np.std(np.log(gamma_vals))

    return gamma_mean, gamma_std_log

def data_preprocessing(inc_values, n_bins=40):
    inc = inc_values - np.mean(inc_values)
    abs_inc = np.abs(inc)

    mask = abs_inc > 0
    inc = inc[mask]
    abs_inc = abs_inc[mask]

    bins_abs = np.logspace(np.log10(abs_inc.min()), np.log10(abs_inc.max()), n_bins)
    hist_abs, bins_abs = np.histogram(abs_inc, bins=bins_abs, density=True)
    x_abs = np.sqrt(bins_abs[:-1] * bins_abs[1:])
    x_emp = np.concatenate([-x_abs[::-1], x_abs])
    hist = np.concatenate([hist_abs[::-1], hist_abs]) / 2
    bin_widths = np.concatenate([np.diff(bins_abs)[::-1], np.diff(bins_abs)])

    return inc,x_emp,hist,bin_widths


def alpha_scan(
    inc_values, q_list, dt, q_core,
    alpha_center, half_width, n_alpha,
    bins_list=(50, 45, 40)
):

    for n_bins in bins_list:

        inc, x_emp, hist, bin_widths = data_preprocessing(inc_values, n_bins)

        # --- 核心区 ---
        r_core = np.quantile(np.abs(inc), q_core)
        mask_core = np.abs(x_emp) <= r_core
        if mask_core.sum() < 3:
            continue

        m_emp = np.sum(hist[mask_core] * bin_widths[mask_core])

        alpha_grid = np.linspace(
            alpha_center - half_width,
            alpha_center + half_width,
            n_alpha
        )
        alpha_grid = alpha_grid[(alpha_grid > 0) & (alpha_grid < 2)]
        alpha_accept = []

        for alpha in alpha_grid:

            gamma, gamma_std_log = gamma_computed(inc, q_list, dt, alpha)
            if gamma is None or gamma <= 0:
                continue

            try:
                scale = (gamma * dt) ** (1.0 / alpha)
                pdf_levy = levy_stable.pdf(
                    x_emp, alpha=alpha, beta=0, loc=0, scale=scale
                )
                if not np.all(np.isfinite(pdf_levy)):
                    continue
            except (ValueError, FloatingPointError):
                continue

            m_levy = np.trapezoid(pdf_levy[mask_core], x_emp[mask_core])

            delta_m = m_levy - m_emp
            tol_center = 0.02 * m_emp

            if abs(delta_m) < tol_center:
                alpha_accept.append(alpha)

        if len(alpha_accept) > 0:
            alpha_accept = np.array(alpha_accept)
            alpha_ci = (alpha_accept.min(), alpha_accept.max())
            alpha = 0.5 * (alpha_ci[0] + alpha_ci[1])
            gamma, gamma_std_log = gamma_computed(inc, q_list, dt, alpha)

            return alpha, alpha_ci, gamma, gamma_std_log

    # 全部 bins 都失败
    return None, None, None, None


def estimate_levy_params(inc_values, dt, q_list, q_core, alpha_init=1.0, step_alpha=0.02, max_iter_alpha=100):

    # ======================================================
    # 数据预处理
    # ======================================================
    inc,x_emp,hist,bin_widths = data_preprocessing(inc_values)

    # ======================================================
    # Stage I：无 cutoff 的 Lévy 拟合（α, γ）
    # ======================================================
    #纯Levy核心区
    r_core = np.quantile(np.abs(inc), q_core)

    gamma, gamma_std_log = gamma_computed(inc,q_list,dt,alpha_init)
    converged = False
    alpha = alpha_init

    for _ in range(max_iter_alpha):

        if not np.isfinite(alpha) or alpha <= 0 or alpha >= 2:
            break

        if gamma is None or not np.isfinite(gamma) or gamma <= 0:
            break

        try:
            scale = (gamma * dt) ** (1.0 / alpha)
            if _invalid(scale):
                raise ValueError("Invalid scale")

            pdf_levy = levy_stable.pdf(
                x_emp, alpha=alpha, beta=0, loc=0, scale=scale
            )

            if not np.all(np.isfinite(pdf_levy)):
                raise FloatingPointError("Non-finite pdf")

        except (ValueError, FloatingPointError):
            converged = False
            break

        # —— 中心区，用于 α ——
        alpha = np.clip(alpha, 0.05, 1.99)
        mask_core = np.abs(x_emp) <= r_core
        if mask_core.sum() < 5:
            continue


        m_levy = np.trapezoid(pdf_levy[mask_core], x_emp[mask_core])
        m_emp = np.sum(hist[mask_core] * bin_widths[mask_core])

        delta_m = m_levy - m_emp
        tol_center = 0.02 * m_emp

        # ---- 收敛判定 ----
        if abs(delta_m) > tol_center:
            alpha += step_alpha * np.sign(delta_m)

            gamma, gamma_std_log = gamma_computed(inc, q_list, dt, alpha)
            if gamma is None:
                converged = False
                break
        else:
            # α 收敛，停止 Stage I
            converged = True
            break

    if not converged:
        return None, None, None

    return alpha, gamma, gamma_std_log



def gamma_std_from_alpha_ci(gamma, alpha_low, alpha_high, alpha, dt, ci_level=0.95):
    """
    gamma: float, 计算值
    alpha_low, alpha_high: alpha 置信区间
    alpha: alpha 均值
    dt: 当前 Δt
    ci_level: 置信水平（默认95%）
    """
    # 对应正态分布 z 值
    from scipy.stats import norm
    z = norm.ppf(0.5 + ci_level / 2)

    # α std 近似
    alpha_std = (alpha_high - alpha_low) / (2 * z)

    # γ std
    gamma_std = gamma * np.log(dt) / (alpha ** 2) * alpha_std

    return gamma_std


def gtlf_params_estimate(log_type):
    """
    对所有井：
    - 基于 μ(Δt) + 幂律尾样本数 选择有效 Δt
    - 对每个 Δt 的增量序列分别估计 α 和 γ
    - 输出长表格，每口井每个 Δt 一行
    """
    base_dir = Path(INTERMEDIATE_DIR)
    stats_dir = Path(STATS_RESULTS) / "gtlf"
    stats_dir.mkdir(parents=True, exist_ok=True)

    results = []
    nll_results = []

    for well_dir in sorted(base_dir.iterdir()):
        well_name = well_dir.name

        for dt in INCREMENTS:
            csv_path = well_dir / log_type / "delta" / f"step_{dt}.csv"
            if not csv_path.exists():
                continue

            df = pd.read_csv(csv_path)
            if f"delta_{log_type}" not in df.columns:
                continue

            inc_values = df.iloc[:, 1].values


            cutoff_init = None

            q_list = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7]

            alpha, alpha_ci, gamma, gamma_std_log = refine_gtlf_params(inc_values, dt, q_list)
            print("well:", well_name, "dt:", dt, "alpha:",alpha, "alpha_ci:", alpha_ci,)
            if alpha is not None:
                cutoff_init = find_tlf_cutoff_ccdf(inc_values, gamma, alpha, dt)

            k_gtlf = None
            c_gtlf = None
            results_estimated = None
            cutoff_ci_95 = None

            print( "cutoff_init:", cutoff_init)

            if alpha is not None and cutoff_init is not None:
                results_estimated, k_gtlf, c_gtlf = estimate_gtlf_parameters_parallel(
                    inc_values, alpha, gamma, dt, cutoff_init)

            if alpha is not None and cutoff_init is None:
                results_estimated, k_gtlf, c_gtlf = estimate_gtlf_parameters_parallel(
                    inc_values, alpha, gamma, dt)

            if results_estimated is not None:
                print("best_lc:", results_estimated["lc"])
                print("k_gtlf:", k_gtlf)
                print("c_gtlf:", c_gtlf)
                # vals: negative log-likelihood (NLL)
                vals = results_estimated["lc_vals_fine"]
                grid = results_estimated["lc_grid_fine"]
                nll_min = vals.min()

                ci68_mask = vals <= nll_min + 0.5
                ci95_mask = vals <= nll_min + 1.92

                cutoff_ci_68 = (grid[ci68_mask][0], grid[ci68_mask][-1])
                cutoff_ci_95 = (grid[ci95_mask][0], grid[ci95_mask][-1])
                print("cutoff_ci_95:", cutoff_ci_95)
                nll_results.append({
                    "well": well_name,
                    "dt": dt,
                    "lc_mle": results_estimated["lc"],
                    "lc_grid_coarse": results_estimated["lc_grid_coarse"],
                    "lc_vals_coarse": results_estimated["lc_vals_coarse"],
                    "lc_grid_fine": results_estimated["lc_grid_fine"],
                    "lc_vals_fine": results_estimated["lc_vals_fine"],
                    "nll_min": nll_min if nll_min is not None else np.nan,
                    "lc_ci68_low": cutoff_ci_68[0] if cutoff_ci_68 is not None else np.nan,
                    "lc_ci68_high": cutoff_ci_68[1] if cutoff_ci_68 is not None else np.nan,
                    "lc_ci95_low": cutoff_ci_95[0] if cutoff_ci_95 is not None else np.nan,
                    "lc_ci95_high": cutoff_ci_95[1] if cutoff_ci_95 is not None else np.nan,
                })

            results.append({
                "well": well_name,
                "dt": dt,
                "cutoff_init": cutoff_init if cutoff_init is not None else np.nan,
                "cutoff": results_estimated["lc"] if results_estimated is not None else np.nan,
                "cutoff_ci_95_l": cutoff_ci_95[0] if cutoff_ci_95 is not None else np.nan,
                "cutoff_ci_95_r": cutoff_ci_95[1] if cutoff_ci_95 is not None else np.nan,
                "alpha": alpha if alpha is not None else np.nan,
                "alpha_ci_lower":alpha_ci[0] if alpha_ci is not None else np.nan,
                "alpha_ci_upper":alpha_ci[1] if alpha_ci is not None else np.nan,
                "gamma": gamma if gamma is not None else np.nan,
                "gamma_std": gamma_std_log if gamma_std_log is not None else np.nan,
                "k_gtlf": k_gtlf if k_gtlf is not None else np.nan,
                "c_gtlf": c_gtlf if c_gtlf is not None else np.nan,
            })

            if not results:
                print("No valid wells found.")
                return

    out_df = pd.DataFrame(results)
    out_csv = stats_dir / f"{log_type}_gtlf_params.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved Levy α–γ by Δt summary to:\n{out_csv}")

    if nll_results is not None:
        nll_stats_dir = Path(STATS_RESULTS) / "gtlf" / "nll_results"
        nll_stats_dir.mkdir(parents=True, exist_ok=True)
        nll_df = pd.DataFrame(nll_results)
        nll_csv = nll_stats_dir / f"nll_for_lc.csv"
        nll_df.to_csv(nll_csv, index=False)
        print(f"\nSaved nll vals of lc to:\n{nll_csv}")


