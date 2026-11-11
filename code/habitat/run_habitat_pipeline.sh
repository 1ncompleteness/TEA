#!/usr/bin/env bash
# TEA Habitat pipeline: parallel collection (5 procs), merged 50k Cal-QL training
# (full + neighbor-blind from same buffer), parallel per-seed evals, trust.
set -uo pipefail

cd "$(dirname "$0")"
PY="${PY:-python}"  # run inside the habitat env (README: Environment)
export HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet
export PYTHONPATH="$PWD/training:$PWD/testing:${PYTHONPATH:-}"
CKPT=ckpt/social_nav_latest.pth
TS=run
mkdir -p results logs

echo "=== [A] Parallel demo collection: 5 x 100 episodes (shuffled iterators) ==="
pids=()
for hs in 101 102 103 104 105; do
  $PY training/train_tea_habitat.py --ckpt $CKPT --collect_only \
      --num_demo_episodes 100 --max_episode_steps 500 \
      --habitat_seed $hs --shuffle_episodes --seed $hs \
      --output_dir results/demo_collection_$hs \
      > logs/${TS}_collect_$hs.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || echo "collector $p failed"; done
echo "collection done"

DEMOS=$(ls results/demo_collection_*/demos.npz | paste -sd,)
echo "buffers: $DEMOS"

echo "=== [B] Train full + blind Q (50k each, parallel) ==="
$PY training/train_tea_habitat.py --ckpt $CKPT --demos_npz "$DEMOS" \
    --train_steps 50000 --seed 42 --output_dir results/training_full \
    > logs/${TS}_train_full.log 2>&1 &
PF=$!
$PY training/train_tea_habitat.py --ckpt $CKPT --demos_npz "$DEMOS" \
    --neighbor_blind --train_steps 50000 --seed 42 \
    --output_dir results/training_neighbor_blind \
    > logs/${TS}_train_blind.log 2>&1 &
PB=$!
wait $PF || { echo "full train failed"; exit 1; }
wait $PB || { echo "blind train failed"; exit 1; }

QFULL=results/training_full/tea_q.pt
QBLIND=results/training_neighbor_blind/tea_q.pt
SD=$($PY -c "import json;print(json.load(open('results/training_full/config.json'))['state_dim'])")
AD=$($PY -c "import json;print(json.load(open('results/training_full/config.json'))['action_dim'])")
echo "state_dim=$SD action_dim=$AD"

echo "=== [C] Parallel evals: K1 x5 seeds, TEA-K4 x5, blind x2 ==="
pids=()
for s in 42 43 44 45 46; do
  $PY testing/tea_habitat_eval.py --ckpt $CKPT --K 1 --split train \
      --num_episodes 30 --max_episode_steps 1500 --seeds $s \
      --output_dir results/${TS}_eval_baseline_K1_s$s \
      > logs/${TS}_eval_baseline_K1_s$s.log 2>&1 &
  pids+=($!)
done
for s in 42 43 44 45 46; do
  $PY testing/tea_habitat_eval.py --ckpt $CKPT --q_path $QFULL --K 4 \
      --soft_bon --alpha 2.0 --sigma 0.2 --split train \
      --state_dim $SD --action_dim $AD \
      --num_episodes 30 --max_episode_steps 1500 --seeds $s \
      --output_dir results/${TS}_eval_tea_K4_s$s \
      > logs/${TS}_eval_K4_s$s.log 2>&1 &
  pids+=($!)
done
for s in 42 43; do
  TEA_NEIGHBOR_BLIND=1 $PY testing/tea_habitat_eval.py --ckpt $CKPT \
      --q_path $QBLIND --K 4 --soft_bon --alpha 2.0 --sigma 0.2 \
      --split train --state_dim $SD --action_dim $AD \
      --num_episodes 30 --max_episode_steps 1500 --seeds $s \
      --output_dir results/${TS}_eval_neighbor_blind_K4_s$s \
      > logs/${TS}_eval_blind_s$s.log 2>&1 &
  pids+=($!)
done
fails=0
for p in "${pids[@]}"; do wait "$p" || fails=$((fails+1)); done
echo "evals done (failures: $fails)"

echo "=== [D] Merge + trust calibration ==="
$PY testing/merge_evals.py "results/${TS}_eval_baseline_K1_s*" "results/${TS}_eval_tea_K4_s*" \
    "results/${TS}_eval_neighbor_blind_K4_s*" results/${TS}_summary.json
$PY testing/habitat_trust_calib.py --run_dir results/training_full \
    --demos_npz "$DEMOS" 2>&1 | tail -12
echo "=== DONE ==="
