#!/bin/bash
export HYDRA_FULL_ERROR=1
set -euxo pipefail

split=${SPLIT:-test}
TRAIN_TEST_SPLIT=nav${split}

pred_dir=${PRED_DIR:-/path/to/your/planning_results_dir}
metric_cache_path=${METRIC_CACHE_PATH:-${NAVSIM_EXP_ROOT}}

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score_one_stage.py" \
    train_test_split=$TRAIN_TEST_SPLIT \
    agent=human_agent \
    experiment_name=drivedreamer-policy \
    metric_cache_path=$metric_cache_path \
    pred_dir="$pred_dir" \
    split=$split
