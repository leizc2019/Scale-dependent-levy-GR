import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import levy_stable
import hashlib
from functools import lru_cache


def sample_gtl(alpha, lc, size=200000):
    """
    简化 GTL 采样：
    - 先生成 α-stable
    - 对大跳进行指数截断（accept–reject）
    """
    samples = []
    batch = size

    while len(samples) < size:
        x = levy_stable.rvs(alpha, 0, size=batch)  # 对称 α-stable
        accept_prob = np.exp(-np.abs(x) / lc)
        u = np.random.rand(batch)
        accepted = x[u < accept_prob]
        samples.extend(accepted.tolist())

    return np.array(samples[:size])



@lru_cache(maxsize=4096)
def compute_cq_cached(alpha, cutoff, q):
    if cutoff is None:
        return levy_stable.ppf((1 + q) / 2, alpha, 0)
    else:
        samples = sample_gtl(alpha, cutoff, size=200000)
        return np.quantile(np.abs(samples), q)


def compute_cq(alpha, cutoff=None, q=0.75, n_mc=200000):
    if alpha <= 0 or alpha >= 2:
        raise ValueError(f"Invalid alpha for Levy: {alpha}")
    if cutoff is None:
        # ===== 无截断 Lévy：解析分位数 =====
        return levy_stable.ppf((1 + q) / 2, alpha, 0)
        # 用标准对称 alpha-stable，scale=1
        # samples = levy_stable.rvs(alpha=alpha, beta=0, loc=0, scale=1, size=n_mc)
    else:
        # ===== 截断 Lévy：Monte Carlo =====
        samples = sample_gtl(alpha, cutoff, size=n_mc)
        return np.quantile(np.abs(samples), q)




