# Code for manuscript "Efficient coding characterizes altered neural representations elicited by subtle sensory lesions"

Code and artifacts to reproduce the figures and statistical tables for the associated analysis.

The repository ships the derived analysis artifacts under `data/artifacts/`, including rate-level functions, threshold/gain fits, mutual-information summaries, canonical 3000-replicate optimization-prior bootstraps, and additional 5000-replicate bootstrap sidecars for sensitivity checks. The raw recordings are not stored in git; they are archived on Zenodo under DOI `10.5281/zenodo.20394990`.

## Setup

Requires Python >= 3.13.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

All commands are driven by the `Makefile`. Adjust parallelism for heavy steps with `NJOBS`, for example `make bootstrap NJOBS=30`.

## Fast track: reproduce from shipped artifacts

This path uses the shipped artifacts and does not require downloading the raw recordings.

```bash
make tables    # regenerate data/derived/tables/ from shipped artifacts
make figures   # render figures 1-5 to figures/svg/
make all       # tables + figures
```

The prior grid `df_mi_pdf.parquet` is derived and not tracked directly. If absent, `make figures` triggers `make mi` once to regenerate it under `cache/` and symlink it into `data/artifacts/`.

Canonical table outputs use the 3000-replicate bootstrap files. The `*_bootstrap_5000.parquet` files in `data/artifacts/` are included as additional sensitivity artifacts and are not used by `make tables`.

## Heavy track: regenerate artifacts from raw recordings

Download and extract the raw data archive from Zenodo:

```bash
make download-raw
make extract-raw
```

This downloads:

```text
https://zenodo.org/api/records/20394990/files/raw-data_HHL-CHL-EfficientCoding.zip/content
```

and extracts it to `data/raw/`. To rebuild the shipped artifacts from those raw recordings:

```bash
make rlfs       # build rate-level functions from data/raw/
make stg        # fit threshold/gain sigmoid summaries
make mi         # rebuild mutual-information optimization-prior summaries
make bootstrap  # rebuild canonical 3000-replicate bootstrap artifacts and table profiles
make artifacts  # extract raw data, then run rlfs + stg + mi + bootstrap
```

## Layout

```text
config/reproduction.yaml   grid, bootstrap, and split parameters
data/artifacts/            shipped analysis artifacts and bootstrap outputs
data/derived/tables/       generated statistical table CSVs and rendered text
scripts/                   artifact, statistics, bootstrap, and table-generation code
scripts/panels/            one module per figure (figure1..5)
figures/svg/               generated figure output from make figures
```
