#!/usr/bin/env bash
# Reproduce every number in the pilot. Total about 20 minutes on one CPU core.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../results
python uav_train.py                       | tee ../results/uav_train_log.txt
python uav_run.py 0.79 0 30 plain79       | tee ../results/uav_run_plain79_log.txt
python uav_run.py 0.79 1 30 norm79        | tee ../results/uav_run_norm79_log.txt
python make_figure.py
python converter_train.py                 | tee ../results/converter_train_log.txt
python converter_run.py 30 0.6 derate     | tee ../results/converter_run_derate_log.txt
python converter_run.py 20 1.5 weak       | tee ../results/converter_run_weak_log.txt
python converter_normalization.py         | tee ../results/converter_normalization_log.txt
python make_figure_converter.py
# python converter_safety_filter_exploratory.py 20 0.6 t1   # exploratory, inconclusive; see README
