from pathlib import Path
import pandas as pd


class DeltaIO:
    def __init__(self, intermediate_dir: Path):
        self.intermediate_dir = intermediate_dir

    def base_dir(self, well_name: str, log_type: str) -> Path:
        return self.intermediate_dir / well_name / log_type

    def delta_dir(self, well_name: str, log_type: str) -> Path:
        return self.base_dir(well_name, log_type) / "delta"

    def abs_delta_dir(self, well_name: str, log_type: str) -> Path:
        return self.base_dir(well_name, log_type) / "abs_delta"

    @staticmethod
    def read_log(file_path: Path) -> pd.DataFrame:
        return pd.read_excel(file_path)

    @staticmethod
    def write_csv(df: pd.DataFrame, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
