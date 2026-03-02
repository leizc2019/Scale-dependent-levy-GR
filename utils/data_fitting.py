from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from configs.config import STATS_RESULTS
from utils.well_id import WellID


wid = WellID("data/well_id_mapping.csv")


# ===============================
# 拟合函数定义
# ===============================
def linear_func(x, a, b):
    return a * x + b


def exp_func(x, a, b):
    return a * np.exp(b * x)


def log_func(x, a, b):
    return a * np.log(x) + b


def power_func(x, a, b):
    return a * x ** b


# ===============================
# 拟合主函数
# ===============================
def fit_relationship_per_well(df,  x_col, y_col, r2_threshold_diff=0.05, well_col='well'):
    """
    对每口井的数据进行多种关系拟合（线性、指数、对数、幂函数）。

    df: pd.DataFrame, 原始数据
    well_col: str, 井名列
    x_col, y_col: str, 拟合的 x、y 列
    r2_threshold_diff: float, 不同拟合R²差异小于该阈值时，输出全部，否则只输出最优拟合
    """

    stats_base = Path(STATS_RESULTS)
    out_stats_dir = stats_base / "gtlf" / "params_relationship"
    out_stats_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for well, group in df.groupby(well_col):
        x = group[x_col].values
        y = group[y_col].values

        # 排除空值和非正数（对数和幂函数要求）
        mask = (~np.isnan(x)) & (~np.isnan(y))
        x, y = x[mask], y[mask]
        if len(x) < 3:
            continue  # 数据太少无法拟合

        # 获取对应 well ID
        well_id = wid.id(well)
        min_idx = np.argmin(x)
        x_min_val = x[min_idx]
        y_at_x_min = y[min_idx]
        x_max_val = np.max(x)
        y_at_x_max = y[np.argmax(x)]
        y_min_val = np.min(y)
        y_max_val = np.max(y)

        funcs = {
            'linear': linear_func,
            'exponential': exp_func,
            'logarithmic': log_func,
            'power': power_func
        }

        fit_info = []

        for name, func in funcs.items():
            try:
                # 对 log/power 函数 x 必须 >0
                if name in ['logarithmic', 'power']:
                    x_fit = x[x > 0]
                    y_fit = y[x > 0]
                    if len(x_fit) < 3:
                        continue
                else:
                    x_fit = x
                    y_fit = y

                # 初始猜测
                p0 = [1, 1]

                popt = np.array(curve_fit(func, x_fit, y_fit, p0=p0, maxfev=10000)[0])

                y_pred = func(x_fit, *popt)

                # Pearson 相关系数
                pearson_r, _ = pearsonr(y_fit, y_pred)

                # R² 拟合优度
                ss_res = np.sum((y_fit - y_pred) ** 2)
                ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
                r2 = 1 - ss_res / ss_tot
                expr = None

                # 数学表达式字符串
                if name == 'linear':
                    expr = f"y = {popt[0]:.2f}*x + {popt[1]:.2f}"
                elif name == 'exponential':
                    expr = f"y = {popt[0]:.2f}*exp({popt[1]:.2f}*x)"
                elif name == 'logarithmic':
                    expr = f"y = {popt[0]:.2f}*ln(x) + {popt[1]:.2f}"
                elif name == 'power':
                    expr = f"y = {popt[0]:.2f}*x^{popt[1]:.2f}"

                fit_info.append({
                    'well': well,
                    "well_id": well_id,
                    'function': name,
                    'expression': expr,
                    'r2': round(r2, 2),
                    'x_min': round(x_min_val, 2),
                    "x_max": round(x_max_val, 2),
                    "y_min": round(y_min_val, 2),
                    "y_max": round(y_max_val, 2),
                    'y_at_x_min': round(y_at_x_min, 2),
                    "y_at_x_max": round(y_at_x_max, 2),
                })

            except (RuntimeError, ValueError, TypeError) as e:
                print(f"Fit failed for well {well}, function {name}: {e}")
                # 拟合失败则跳过
                continue

        # =====================
        # 根据 R² 判断输出
        # =====================
        if not fit_info:
            continue

        r2_values = [f['r2'] for f in fit_info]
        r2_best = max(r2_values)

        for f in fit_info:
            if r2_best - f['r2'] <= r2_threshold_diff:
                results.append(f)

    df_out = pd.DataFrame(results)
    out_csv = out_stats_dir / f"{x_col}_vs_{y_col}.csv"
    df_out.to_csv(out_csv, index=False)
