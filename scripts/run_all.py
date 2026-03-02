from pathlib import Path
from configs.config import STATS_RESULTS
from pipeline.stats_results import run_compute_stats_all
from utils.merge_csv_files import merge_csv_files
from pipeline.build_well_id_mapping import build_well_id_mapping
from pipeline.gtlf_params_estimate import gtlf_params_estimate


def main():
    log_type = "GR"
    run_compute_stats_all(log_type)
    build_well_id_mapping()
    gtlf_params_estimate("GR")
    gtlf_csv_path = Path(STATS_RESULTS) / "gtlf" / f"{log_type}_gtlf_params.csv"
    basic_stats_path = Path(STATS_RESULTS) / "all_stats_results" / f"{log_type}_basic_stats.csv"
    csv_paths = [gtlf_csv_path, basic_stats_path]
    on = ["well", "dt"]
    df_merged = merge_csv_files(csv_paths, on)
    df_merged.to_csv(Path(STATS_RESULTS) / "gtlf" / f"{log_type}_merged.csv", index=False)


if __name__ == "__main__":
    main()



