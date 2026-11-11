#!/usr/bin/env bash
# Per-task launcher for FLOWER LIBERO eval.
# Workaround for LIBERO OffScreenRenderEnv GL-context corruption across tasks:
# each task_id runs in fresh Python process.
#
# Args: <suite> <seed> <mode> <q_path>
# mode: baseline | soft | hard
set -eu

SUITE=${1:-libero_object}
SEED=${2:-42}
MODE=${3:-baseline}
Q_PATH=${4:-/workspace/tea_q.pt}

TS=$(date +%Y%m%d_%H%M%S)
TAG="baseline" ; [ "$MODE" = "soft" ] && TAG="tea_K4_soft" ; [ "$MODE" = "hard" ] && TAG="tea_K4_hard"
OUT=/workspace/algorithm/results/libero_${SUITE}_flower_${TAG}_s${SEED}_${TS}
mkdir -p $OUT

echo "OUT=$OUT"
echo "suite=$SUITE seed=$SEED mode=$MODE"

for task_id in 0 1 2 3 4 5 6 7 8 9; do
    echo ""
    echo "=========================================="
    echo "Task $task_id / 9"
    echo "=========================================="
    if [ "$MODE" = "baseline" ]; then
        CMD="--baseline --seed $SEED"
    elif [ "$MODE" = "soft" ]; then
        CMD="--tea_q $Q_PATH --K 4 --rerank_mode soft --alpha 2.0 --sigma 0.2 --seed $SEED"
    else
        CMD="--tea_q $Q_PATH --K 4 --rerank_mode hard --seed $SEED"
    fi
    python -u /workspace/algorithm/run_libero_flower_tea.py \
        --task_suite libero_$SUITE \
        --flower_ckpt /workspace/flower_ckpt.safetensors \
        --flower_config /workspace/flower_cfg.yaml \
        --task_ids $task_id \
        --num_trials_per_task 10 \
        --num_steps_wait 10 \
        --output_dir /workspace/algorithm/results \
        --append_dir $OUT \
        $CMD 2>&1 | tail -20
done

echo ""
echo "=========================================="
echo "All tasks done. Aggregating..."
echo "=========================================="
python -u /workspace/algorithm/aggregate_flower_pertask.py --run_dir $OUT
echo "=========================================="
echo "FINAL $OUT"
echo "=========================================="
