import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Literal


def merge_csv_files(
    csv_paths: List[Path],
    on: List[str],
    how: Literal["left", "right", "inner", "outer"] = "left",
    usecols_list: Optional[List[Optional[List[str]]]] = None,
    rename_list: Optional[List[Optional[Dict[str, str]]]] = None
) -> pd.DataFrame:
    """
    通用 CSV 文件拼接函数。

    参数：
    ----------
    csv_paths : List[Path]
        需要拼接的 CSV 文件路径列表。
    on : List[str]
        拼接参考列。
    how : str, default "left"
        拼接方式，可选 "left", "right", "inner", "outer"。
    usecols_list : List[List[str]] or None
        每个 CSV 需要保留的列列表，None 表示保留全部列。
    rename_list : List[Dict[str, str]] or None
        每个 CSV 的列重命名字典，None 表示不重命名。

    返回：
    ----------
    pd.DataFrame
        多个 CSV 文件合并后的 DataFrame。
    """
    if usecols_list is None:
        usecols_list = [None] * len(csv_paths)
    if rename_list is None:
        rename_list = [None] * len(csv_paths)

    df_merged = None
    for i, csv_path in enumerate(csv_paths):
        if not csv_path.exists():
            raise FileNotFoundError(f"{csv_path} not found.")

        df = pd.read_csv(csv_path, usecols=usecols_list[i])
        if rename_list[i]:
            df = df.rename(columns=rename_list[i])

        if df_merged is None:
            df_merged = df
        else:
            df_merged = pd.merge(df_merged, df, on=on, how=how)

    return df_merged
