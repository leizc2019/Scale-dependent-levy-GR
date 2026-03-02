import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis

from configs.config import (RAW_DIR, INTERMEDIATE_DIR, LOG_TYPES, INCREMENTS, STATS_RESULTS)
from pipeline.delta_pipeline import DeltaPipeline
from pipeline.levy_msd_pipeline import LevyMSDAnalysis
from pipeline.levy_p0_analysis import LevyP0Analysis


def compute_basic_statistics(data):
    """
    Compute basic descriptive statistics for ΔX.

    Returns
    -------
    dict
    """
    std = np.std(data, ddof=1)

    return {
        "n": len(data),
        "std": std,
        "variance": std ** 2,
        "skewness": skew(data, bias=False),
        "excess_kurtosis": kurtosis(data, fisher=True, bias=False)
    }


def run_compute_stats_all(log_type):
    pipeline = DeltaPipeline(
        raw_dir=Path(RAW_DIR),
        intermediate_dir=Path(INTERMEDIATE_DIR),
        log_types=LOG_TYPES,
        increments=INCREMENTS,
    )
    pipeline.run()

    base_dir = Path(INTERMEDIATE_DIR)
    stats_base = Path(STATS_RESULTS)

    msd_analyzer = LevyMSDAnalysis(
        min_points=2,
        msd_fit_range=(1, 1000)
    )
    # 使用物理一致的 p0 分析器
    analyzer = LevyP0Analysis(
        eps_mode="fixed_sigma1",  # 或 "fixed_quantile"
        eps_c=0.05,
        min_points=4
    )

    basic_stats = []
    p0_records = []
    levy_results = []

    for well_dir in sorted(base_dir.iterdir()):
        well_name = well_dir.name

        delta_dir = well_dir / log_type / "delta"
        if not delta_dir.exists():
            continue

        # --------------------------------------------------
        # Step 1: 用 Δt=1 拟合 epsilon（只做一次）
        # --------------------------------------------------
        dt1_path = delta_dir / "step_1.csv"
        if not dt1_path.exists():
            continue

        delta_dt1 = pd.read_csv(dt1_path).iloc[:, 1].values
        delta_dt1 = delta_dt1[np.isfinite(delta_dt1)]

        if len(delta_dt1) == 0:
            continue

        try:
            eps = analyzer.fit_epsilon(delta_dt1)
        except ValueError:
            continue

        dt_msd = []
        msd_vals = []
        p0_cont_list = []
        dt_list = []

        for dt in INCREMENTS:
            csv_path = delta_dir / f"step_{dt}.csv"
            if not csv_path.exists():
                continue

            data = pd.read_csv(csv_path).iloc[:, 1].values
            data = data[np.isfinite(data)]

            if len(data) == 0:
                continue

            msd_value = msd_analyzer.compute_msd(data)

            dt_msd.append(dt)
            msd_vals.append(msd_value)
            stats_dt = compute_basic_statistics(data)

            basic_stats.append({
                "well": well_name,
                "dt": dt,
                "std": stats_dt["std"],
                "variance": stats_dt["variance"],
                "skewness": stats_dt["skewness"],
                "excess_kurtosis": stats_dt["excess_kurtosis"],
                "msd": msd_value,
                "msd_exp": np.nan,
                "r2": np.nan,
                "n_samples": stats_dt["n"]
            })

            p0_total, p0_zero, p0_cont = analyzer.compute_p0_components(data)
            p0_records.append({
                "well": well_name,
                "log_type": log_type,
                "dt": dt,
                "p0_total": p0_total,
                "p0_zero": p0_zero,
                "p0_cont": p0_cont,
                "eps": eps,
                "n_samples": len(data)
            })
            if p0_cont > 0 and np.isfinite(msd_value):
                dt_list.append(dt)
                p0_cont_list.append(p0_cont)

        running_msd_results = msd_analyzer.fit_msd_running(dt_msd, msd_vals)
        if running_msd_results:
            for r in running_msd_results:
                dt_max = r["dt_max"]
                # 在 basic_stats 中找到 dt == dt_max 的那一行
                for row in basic_stats:
                    if row["well"] == well_name and row["dt"] == dt_max:
                        row["msd_exp"] = r["msd_exp"]
                        row["r2"] = r["r2"]
                        break

        # 如果没有levy有效数据，跳过
        if len(dt_list) < 2:
            continue
        fit = analyzer.fit_levy_p0(dt_list, p0_cont_list)
        if fit is None:
            continue
        levy_results.append({
            "well": well_name,
            "log_type": log_type,
            "alpha_levy": fit["alpha"],
            "alpha_ci_low": fit["alpha_ci"][0],
            "alpha_ci_high": fit["alpha_ci"][1],
            "r2": fit["r2"],
            "n_dt": fit["n_points"],
            "eps": eps
        })

    out_basic_stats_dir = stats_base / "all_stats_results"
    out_basic_stats_dir.mkdir(parents=True, exist_ok=True)
    basic_stats_df = pd.DataFrame(basic_stats)
    basic_stats_path = out_basic_stats_dir / f"{log_type}_basic_stats.csv"
    basic_stats_df.to_csv(basic_stats_path, index=False)

    p0_df = pd.DataFrame(p0_records)
    p0_path = out_basic_stats_dir / f"{log_type}_p0_stats.csv"
    p0_df.to_csv(p0_path, index=False)

    levy_df = pd.DataFrame(levy_results)
    levy_path = out_basic_stats_dir / f"{log_type}_levy_alpha_from_p0_stats.csv"
    levy_df.to_csv(levy_path, index=False)

    print("all statistics are saved")






