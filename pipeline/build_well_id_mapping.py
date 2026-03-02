from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/GR")
OUT_CSV = Path("data/well_id_mapping.csv")


def estimate_dt_from_gr(file_path):
    df = pd.read_excel(file_path)
    depth = df.iloc[:, 0].dropna().values

    if len(depth) < 2:
        raise ValueError(f"{file_path.name}: insufficient depth samples")

    return float(depth[1] - depth[0])


def build_well_id_mapping(raw_dir=RAW_DIR, out_csv=OUT_CSV):
    wells = sorted([
        f for f in raw_dir.iterdir()
        if f.is_file() and f.suffix.lower() == ".xlsx" and not f.name.startswith("~$")
    ])

    if len(wells) == 0:
        raise RuntimeError("No wells found in raw directory")

    records = []
    for i, f in enumerate(wells, start=1):
        records.append({
            "well_id": i,
            "well": f.stem,
            "dt": estimate_dt_from_gr(f)
        })

    df = pd.DataFrame(records)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"[OK] Generated {len(df)} well IDs → {out_csv}")



