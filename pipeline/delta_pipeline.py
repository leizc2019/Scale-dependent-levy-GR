from pathlib import Path
import pandas as pd
from utils.io import DeltaIO


def is_empty(dir_path: Path) -> bool:
    if not dir_path.exists():
        return True
    return not any(dir_path.iterdir())


def compute_delta(values, step):
    return values[step:] - values[:-step]


class DeltaPipeline:
    def __init__(
        self,
        raw_dir: Path,
        intermediate_dir: Path,
        log_types: dict,
        increments: list[int],
    ):
        self.raw_dir = raw_dir
        self.io = DeltaIO(intermediate_dir)
        self.log_types = log_types
        self.increments = increments

    def process_single_log(
        self,
        well_name: str,
        log_type: str,
        file_path: Path,
    ):
        column = self.log_types[log_type]["column"]
        depth_col = self.log_types[log_type]['depth']

        delta_dir = self.io.delta_dir(well_name, log_type)

        if not is_empty(delta_dir):
            return

        df = self.io.read_log(file_path)

        if column not in df.columns:
            raise ValueError(
                f"{file_path} missing required column '{column}'"
            )
        if depth_col not in df.columns:
            raise ValueError(
                f"{file_path} missing required depth column '{depth_col}'"
            )

        values = df[column].values
        depth = df[depth_col].values

        for step in self.increments:
            if step >= len(values):
                print(f"    [WARN] step={step} too large, skipped.")
                continue

            delta = compute_delta(values, step)

            depth_delta = depth[step:]

            self.io.write_csv(
                pd.DataFrame({"depth": depth_delta, f"delta_{log_type}": delta}),
                delta_dir / f"step_{step}.csv",
            )

            print(
                f"    saved {log_type}, step={step}, N={len(delta)}"
            )

    def run(self):
        for log_type in self.log_types:
            raw_log_dir = self.raw_dir / log_type

            if not raw_log_dir.exists():
                print(f"[WARN] raw directory not found: {raw_log_dir}")
                continue

            for file_path in raw_log_dir.iterdir():
                if file_path.suffix.lower() not in (".xlsx", ".xls"):
                    continue

                well_name = file_path.stem
                self.process_single_log(
                    well_name, log_type, file_path
                )
