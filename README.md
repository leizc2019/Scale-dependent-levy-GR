Scale-Dependent GTLF Analysis of Gamma-Ray Well Logs
Overview

This repository provides a reproducible implementation of a scale-dependent gradually truncated Lévy (GTLF) framework for analyzing gamma-ray (GR) well-log increments.

The framework allows the stability index (α) and truncation scale (l_c) to evolve explicitly with observational scale (Δt), enabling quantitative characterization of nonstationary heavy-tailed behavior in geological records.

The workflow includes:

Multi-scale increment computation

GTLF parameter estimation

Mean squared displacement (MSD) scaling analysis

Automated merging of statistical outputs

This implementation corresponds to the methods described in:

Scale-dependent Lévy statistics indicate environment-modulated transition scales in geological systems
(Manuscript submitted)

Scientific Background

Classical Lévy models assume stationary increments.
This framework extends truncated Lévy modeling by explicitly estimating scale-evolving parameters α(Δt) and l_c(Δt), allowing statistical transitions to be quantified across multiple observational scales.

The approach integrates:

Gradually truncated Lévy fitting

Complementary cumulative distribution function (CCDF) diagnostics

Running MSD scaling analysis

The resulting parameter trajectories define operational transition scales in heterogeneous stratigraphic systems.

Repository Structure
configs/        Global configuration settings  
pipeline/       Core workflow (statistics + GTLF estimation)  
scripts/        Entry script (run_all.py)  
utils/          Utility functions (I/O, fitting, merging)  
data/           Input data (user-supplied)  
Requirements

Python 3.12 or higher

numpy

scipy

pandas

matplotlib

Install dependencies:

pip install -r requirements.txt
Running the Workflow

From the project root directory:

python -m scripts.run_all

Before execution, configure input and output paths in:

configs/config.py
Input Data

This repository does not redistribute proprietary datasets.

Gamma-ray well-log data analyzed in the associated study are publicly available from the International Ocean Discovery Program (IODP):

https://iodp.tamu.edu/

Users must download the relevant datasets and configure paths accordingly.

Output

Results are saved according to configuration settings and include:

Basic statistical summaries

Estimated GTLF parameters (α, l_c, and related metrics)

Merged multi-scale result tables

Reproducibility

All analyses are deterministic given fixed input data.

Parameter estimation procedures follow the methodology described in the Supplementary Material of the associated publication.

For reproducible research purposes, a tagged release corresponding to the manuscript submission is archived.

Citation

If you use this code, please cite:

[Full paper citation, once published]

License

This project is released under the MIT License
