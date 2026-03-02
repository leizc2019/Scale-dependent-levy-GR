import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import levy_stable
from joblib import Parallel, delayed

# ================= tail 指数向量化 =================
def gtlf_kernel_from_base(base_pdf, x, lc, k, beta):
    tail_factor = np.maximum(np.abs(x) - lc, 0)
    return base_pdf * np.exp(-(tail_factor / k) ** beta)

# ================= 预计算 core / tail =================
def precompute_gtlf_components(data, alpha, gamma_levy, dt, lc, n_core=2000, n_tail=3000, tail_factor=20.0):
    scale = gamma_levy * dt
    base_pdf_data = levy_stable.pdf(data, alpha, 0, scale=scale)
    base_pdf_data = np.asarray(base_pdf_data, dtype=float)

    x_core = np.linspace(-lc, lc, n_core)
    pdf_core = levy_stable.pdf(x_core, alpha, 0, scale=scale)
    pdf_core = np.asarray(pdf_core, dtype=float)
    core = np.sum(pdf_core * (x_core[1] - x_core[0]))  # 向量化积分

    x_tail = np.linspace(lc, lc + tail_factor * lc, n_tail)
    pdf_tail = levy_stable.pdf(x_tail, alpha, 0, scale=scale)
    pdf_tail = np.asarray(pdf_tail, dtype=float)

    return {"base_pdf_data": base_pdf_data, "core": core, "x_tail": x_tail, "pdf_tail": pdf_tail}

# ================= c 计算 =================
def compute_c_fast(core, x_tail, pdf_tail, lc, k, beta):
    tail_weight = np.exp(-((x_tail - lc) / k) ** beta)
    tail = np.sum(pdf_tail * tail_weight * (x_tail[1] - x_tail[0]))
    den = core + 2.0 * tail
    if not np.isfinite(den) or den <= 0:
        return np.inf
    return 1.0 / den

# ================= loglik_k =================
def neg_loglik_k_fast(logk, data, lc, base_pdf_data, core, x_tail, pdf_tail, beta):
    k = np.exp(logk)
    c = compute_c_fast(core, x_tail, pdf_tail, lc, k, beta)
    pdf = gtlf_kernel_from_base(base_pdf_data, data, lc, k, beta) * c
    if not np.all(np.isfinite(pdf)):
        return np.inf
    val = np.sum(np.log(np.maximum(pdf, 1e-300)))
    return -float(val)

# ================= profile lc 并行 =================
def profile_negloglik_lc_fast(lc, data, alpha, gamma_levy, dt):
    beta = 2.0 - alpha
    cache = precompute_gtlf_components(data, alpha, gamma_levy, dt, lc, n_core=2000, n_tail=3000)
    res = minimize_scalar(neg_loglik_k_fast, bounds=(float(np.log(1e-6)), float(np.log(1e3))),
                          args=(data, lc, cache["base_pdf_data"], cache["core"], cache["x_tail"], cache["pdf_tail"], beta),
                          method="bounded")
    k_opt = np.exp(res.x)
    c_opt = compute_c_fast(cache["core"], cache["x_tail"], cache["pdf_tail"], lc, k_opt, beta)
    return -np.sum(np.log(np.maximum(gtlf_kernel_from_base(cache["base_pdf_data"],
                                                           data, lc, k_opt, beta) * c_opt, 1e-300)))


def parallel_local_search(data, alpha, gamma, dt, best_lc, fine_range, n_candidates, n_jobs=-1):
    bounds = (best_lc*(1-fine_range), best_lc*(1+fine_range))
    grid = np.linspace(bounds[0], bounds[1], n_candidates)
    with Parallel(n_jobs) as parallel:
        vals = parallel(
            delayed(profile_negloglik_lc_fast)(lc, data, alpha, gamma, dt) for lc in grid
        )
    best_idx = np.argmin(vals)
    return grid[best_idx], grid, vals


# ================= cutoff 并行搜索 =================
def estimate_cutoff_from_raw_parallel(inc_values, alpha, gamma_levy, dt,
                                      lc0=None, fine_range=0.25, n_grid=25, n_jobs=-1):
    data = inc_values - np.mean(inc_values)
    if lc0 is not None:
        lc_min, lc_max = lc0 * 0.8, lc0 * 3
    else:
        abs_data = np.abs(data)
        lc_min, lc_max = np.quantile(abs_data, [0.8, 0.999])
    grid_coarse = np.linspace(lc_min, lc_max * 2, n_grid)

    with Parallel(n_jobs) as parallel:
        vals_coarse = parallel(
            delayed(profile_negloglik_lc_fast)(lc, data, alpha, gamma_levy, dt)
            for lc in grid_coarse
        )
    vals_coarse = np.asarray(vals_coarse)

    best_idx = np.argmin(vals_coarse)
    best_lc = grid_coarse[best_idx]

    # =========================
    # 2. 局部加密 profile likelihood
    # =========================
    lc1, grid_fine, vals_fine = parallel_local_search(
        data, alpha, gamma_levy, dt,
        best_lc, fine_range, n_grid
    )
    vals_fine = np.asarray(vals_fine)

    return {
        "lc": lc1,
        "lc_grid_coarse": grid_coarse,
        "lc_vals_coarse": vals_coarse,
        "lc_grid_fine": grid_fine,
        "lc_vals_fine": vals_fine,
    }


# ================= 一次性估计 lc, k, c =================
def estimate_gtlf_parameters_parallel(inc_values, alpha, gamma_levy, dt,
                                      lc0=None, fine_range=0.25, n_grid=25, n_jobs=-1):
    data = inc_values - np.mean(inc_values)
    beta = 2.0 - alpha

    lc_results = estimate_cutoff_from_raw_parallel(inc_values, alpha, gamma_levy, dt, lc0, fine_range, n_grid, n_jobs)
    lc_opt = lc_results["lc"]
    cache = precompute_gtlf_components(data, alpha, gamma_levy, dt, lc_opt, n_core=4000, n_tail=6000)

    res_k = minimize_scalar(
        neg_loglik_k_fast,
        bounds=(np.log(1e-6), np.log(1e3)),
        args=(data, lc_opt, cache["base_pdf_data"], cache["core"], cache["x_tail"], cache["pdf_tail"], beta),
        method='bounded'
    )
    k_opt = np.exp(res_k.x)
    c_opt = compute_c_fast(cache["core"], cache["x_tail"], cache["pdf_tail"], lc_opt, k_opt, beta)
    return lc_results, k_opt, c_opt
