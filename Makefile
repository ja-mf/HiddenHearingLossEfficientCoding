# Reproducibility pipeline for the HHL/CHL efficient-coding analysis.
#
# Two tracks (see README):
#   Track 1 (fast, from shipped artifacts):
#       make all       = figures tables
#   Track 2 (heavy, regenerates artifacts; needs raw data):
#       make artifacts = download/extract raw -> rlfs -> stg -> mi -> bootstrap
#
# df_mi_pdf.parquet (~310 MB) is not shipped; the figures/* targets depend on
# it and trigger `make mi` once if the cache is absent. compute_mi.py uses
# joblib.Memory under .cache/, so repeated `make mi` calls re-emit the parquet
# from cached Monte-Carlo MI in seconds.
#
# Wall-clock on Apple M4 Pro, NJOBS=10 (measured):
#   empirical          ~6 s,    260 MB peak RSS
#   tables             ~4 s,    280 MB
#   mi                ~45 s,    7.5 GB  (cold or warm; bulk cost is the 31.25M-row expand+parquet write, not the MC)
#   figures           ~50 s,    fig4/5 peak ~5.3 GB; fig1-3 ~0.4 GB
#   bootstrap         ~5 min,   ~5 GB per cohort (parent + workers); 4 cohorts ~75 s each
#   quintile profile  ~5 min    (5 quintile bootstraps; same memory shape as bootstrap)
#
# Override parallelism with: make bootstrap NJOBS=30

SHELL    := /bin/bash
PY       := python
PYPATH   := PYTHONPATH=scripts
NJOBS    ?= 10
ART      := data/artifacts
COHORTS  := HHL CHL
VARIANTS := HHL HHL_low HHL_high CHL
RAW_ZIP  := cache/raw-data_HHL-CHL-EfficientCoding.zip
RAW_URL  := https://zenodo.org/api/records/20394990/files/raw-data_HHL-CHL-EfficientCoding.zip/content

.PHONY: all figures figure1 figure2 figure3 figure4 figure5 \
        tables empirical \
        artifacts download-raw extract-raw rlfs stg mi bootstrap \
        clean-derived help

# ---------- Track 1: figures and tables from shipped artifacts ------------
all: figures tables

figures: figure1 figure2 figure3 figure4 figure5
figure1 figure2 figure3 figure4 figure5: $(ART)/df_mi_pdf.parquet
	$(PYPATH) $(PY) -c "import panels.$@"

# df_mi_pdf.parquet is derived; rebuild on demand if absent.
$(ART)/df_mi_pdf.parquet:
	$(MAKE) mi

empirical:
	$(PYPATH) $(PY) scripts/empirical_lme.py

# Regenerate the canonical CSVs (empirical_tests, bootstrap_contrasts) and
# print Tables S1/S2/S3 to stdout.
tables: empirical
	@mkdir -p data/derived/tables
	$(PYPATH) $(PY) scripts/reproduce_tables.py >/dev/null
	$(PYPATH) $(PY) scripts/render_tables.py | tee data/derived/tables/rendered_tables.txt

# ---------- Track 2: regenerate shipped artifacts -------------------------
artifacts: extract-raw rlfs stg mi bootstrap

download-raw: $(RAW_ZIP)

$(RAW_ZIP):
	mkdir -p cache
	curl -L "$(RAW_URL)" -o "$(RAW_ZIP)"

extract-raw: $(RAW_ZIP)
	mkdir -p data/raw
	unzip -q -o "$(RAW_ZIP)" -d data -x "__MACOSX/*" "*/.DS_Store"

rlfs:
	@for c in $(COHORTS); do echo ">> rlfs $$c"; $(PYPATH) $(PY) scripts/build_rlfs.py $$c; done

stg:
	@for c in $(COHORTS); do echo ">> stg $$c"; $(PYPATH) $(PY) scripts/fit_stg.py $$c --n-jobs $(NJOBS); done

mi:
	mkdir -p cache
	$(PYPATH) $(PY) scripts/compute_mi.py --out-dir cache
	cp cache/df_mi_stats.parquet $(ART)/df_mi_stats.parquet
	ln -sf ../../cache/df_mi_pdf.parquet $(ART)/df_mi_pdf.parquet

bootstrap: $(ART)/df_mi_pdf.parquet
	@for v in $(VARIANTS); do echo ">> bootstrap $$v"; $(PYPATH) $(PY) scripts/run_bootstrap.py --cohort $$v --n-jobs $(NJOBS); done
	@echo ">> threshold quintile profile"; $(PYPATH) $(PY) scripts/threshold_quantile_profile.py --n-jobs $(NJOBS)

clean-derived:
	rm -rf data/derived/tables figures/svg

help:
	@grep -E '^[a-z][a-z0-9-]*:' Makefile | sed 's/:.*//' | sort -u
