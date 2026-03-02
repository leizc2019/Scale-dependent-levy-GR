import pandas as pd
from pathlib import Path


class WellID:
    def __init__(self, csv_path):
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        df = pd.read_csv(csv_path)
        self.well_to_id = dict(zip(df["well"], df["well_id"]))
        self.id_to_well = dict(zip(df["well_id"], df["well"]))

    def id(self, well):
        return self.well_to_id.get(well)

    def add_to_df(self, df, well_col="well", id_col="well_id"):
        df[id_col] = df[well_col].map(self.well_to_id)
        return df
