# Rebuild every result in this repository, then check it.
PY := python3

.PHONY: all analysis figures verify clean distclean help

all: verify

## analysis  : fits, statistics, result tables and the numerical report
analysis:
	$(PY) analysis/run_analysis.py

## figures   : the five figures
figures:
	$(PY) analysis/make_figures.py

## verify    : rebuild everything, then check it against the source data
verify: analysis figures
	$(PY) analysis/verify_data.py

## clean     : remove Python scratch, keep the results
clean:
	rm -rf analysis/__pycache__

## distclean : also remove every generated result and figure
distclean: clean
	rm -f results/* figures/*

## help      : list the targets
help:
	@grep -E '^##' $(MAKEFILE_LIST) | sed 's/## //'
