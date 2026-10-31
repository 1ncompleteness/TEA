#!/usr/bin/env bash
# CALVIN 34-task MTLC eval: MDT baseline + MDT+TEA K=4 rerank.
# Waits for LIBERO orchestrator (PID set via LIBERO_ORCH_PID) to exit before launching.
# Each eval prints per-task SR (mdt_evaluate.py line 93) + writes results.json to train_folder.
#
# Usage:
#   LIBERO_ORCH_PID=2823 bash algorithm/run_mdt_calvin_34task.sh
#   (or bash algorithm/run_mdt_calvin_34task.sh  # runs immediately)

set -u

REPO_ROOT="${REPO_ROOT:-$(pwd)}"  # set to your repository root
LOG_ROOT="${REPO_ROOT}/algorithm/results"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG_BASE="${LOG_ROOT}/mdt_34task_${STAMP}"
mkdir -p "${LOG_BASE}"

# Wait for LIBERO orchestrator to exit (GPU contention)
if [ -n "${LIBERO_ORCH_PID:-}" ]; then
    echo "[34task] waiting for LIBERO orchestrator PID=${LIBERO_ORCH_PID} to exit..."
    while kill -0 "${LIBERO_ORCH_PID}" 2>/dev/null; do sleep 60; done
    echo "[34task] orchestrator exited. waiting 30s for GPU drain..."
    sleep 30
fi

MDT_SRC="${REPO_ROOT}/environments/mdt_policy"
CALVIN_DATA="${REPO_ROOT}/environments/calvin/dataset"
MDT_PRE="${REPO_ROOT}/algorithm/results/mdt_pretrained"
VOLTRON="${REPO_ROOT}/algorithm/results/voltron_cache"
TEA_Q="${REPO_ROOT}/algorithm/results/tea_mdt_s42_20260418_125901/tea_q.pt"

COMMON_DOCKER=(
    docker run -d --gpus all --cpus=6 --memory=12g --runtime=nvidia
    --mount type=bind,source=/mnt/wslg,target=/mnt/wslg
    --mount type=bind,source=/usr/lib/wsl,target=/usr/lib/wsl
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix
    -v "${MDT_SRC}:/workspace/mdt_src"
    -v "${CALVIN_DATA}:/workspace/calvin_dataset"
    -v "${REPO_ROOT}/algorithm:/workspace/algorithm"
    -v "${MDT_PRE}:/workspace/mdt_pretrained"
    -v "${VOLTRON}:/workspace/voltron_cache"
    -v "${TEA_Q}:/workspace/tea_q.pt"
    -e DISPLAY=:0 -e WAYLAND_DISPLAY=wayland-0
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu
    -e PYTHONUNBUFFERED=1
    -w /workspace/mdt_policy
    mdt:latest
)

run_eval() {
    local tag=$1 tea_env=$2 script=$3
    local name="mdt_34task_${tag}_${STAMP}"
    local log="${LOG_BASE}/${tag}.log"
    docker rm -f "${name}" 2>/dev/null || true
    "${COMMON_DOCKER[@]:0:${#COMMON_DOCKER[@]}-1}" \
        --name "${name}" ${tea_env} \
        mdt:latest \
        bash -c "find /workspace/mdt_policy -name '__pycache__' -exec rm -rf {} + 2>/dev/null; cp -rf /workspace/mdt_src/mdt/* /workspace/mdt_policy/mdt/; cp -rf /workspace/mdt_src/calvin_env/calvin_env/* /workspace/mdt_policy/calvin_env/calvin_env/; touch /workspace/mdt_policy/calvin_env/calvin_env/__init__.py; bash /workspace/algorithm/${script} 2>&1 | tee /workspace/algorithm/results/${log##*/algorithm/results/}"
    echo "[34task] launched ${tag} -> ${name}"
}

echo "[34task] Phase 1: MDT baseline eval (num_sequences=100, per-task SR)"
docker rm -f mdt_34task_base_${STAMP} 2>/dev/null || true
docker run -d --name "mdt_34task_base_${STAMP}" --gpus all --cpus=6 --memory=12g --runtime=nvidia \
    --mount type=bind,source=/mnt/wslg,target=/mnt/wslg \
    --mount type=bind,source=/usr/lib/wsl,target=/usr/lib/wsl \
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
    -v "${MDT_SRC}:/workspace/mdt_src" \
    -v "${CALVIN_DATA}:/workspace/calvin_dataset" \
    -v "${REPO_ROOT}/algorithm:/workspace/algorithm" \
    -v "${MDT_PRE}:/workspace/mdt_pretrained" \
    -v "${VOLTRON}:/workspace/voltron_cache" \
    -e DISPLAY=:0 -e WAYLAND_DISPLAY=wayland-0 \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu \
    -e PYTHONUNBUFFERED=1 \
    -w /workspace/mdt_policy \
    mdt:latest \
    bash -c "find /workspace/mdt_policy -name '__pycache__' -exec rm -rf {} + 2>/dev/null; cp -rf /workspace/mdt_src/mdt/* /workspace/mdt_policy/mdt/; cp -rf /workspace/mdt_src/calvin_env/calvin_env/* /workspace/mdt_policy/calvin_env/calvin_env/; touch /workspace/mdt_policy/calvin_env/calvin_env/__init__.py; bash /workspace/algorithm/run_mdt_eval.sh 2>&1 | tee /workspace/algorithm/results/mdt_34task_${STAMP}/base.log"

echo "[34task] baseline launched. waiting for completion..."
while docker ps --format '{{.Names}}' | grep -q "^mdt_34task_base_${STAMP}$"; do sleep 60; done
echo "[34task] baseline done at $(date). logs: ${LOG_BASE}/base.log"

echo "[34task] Phase 2: MDT+TEA K=4 rerank eval"
docker rm -f mdt_34task_tea_${STAMP} 2>/dev/null || true
docker run -d --name "mdt_34task_tea_${STAMP}" --gpus all --cpus=6 --memory=12g --runtime=nvidia \
    --mount type=bind,source=/mnt/wslg,target=/mnt/wslg \
    --mount type=bind,source=/usr/lib/wsl,target=/usr/lib/wsl \
    --mount type=bind,source=/tmp/.X11-unix,target=/tmp/.X11-unix \
    -v "${MDT_SRC}:/workspace/mdt_src" \
    -v "${CALVIN_DATA}:/workspace/calvin_dataset" \
    -v "${REPO_ROOT}/algorithm:/workspace/algorithm" \
    -v "${MDT_PRE}:/workspace/mdt_pretrained" \
    -v "${VOLTRON}:/workspace/voltron_cache" \
    -v "${TEA_Q}:/workspace/tea_q.pt" \
    -e DISPLAY=:0 -e WAYLAND_DISPLAY=wayland-0 \
    -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu \
    -e TEA_Q_PATH=/workspace/tea_q.pt -e TEA_K=4 -e TEA_LOCAL_STEPS=0 \
    -e PYTHONUNBUFFERED=1 \
    -w /workspace/mdt_policy \
    mdt:latest \
    bash -c "find /workspace/mdt_policy -name '__pycache__' -exec rm -rf {} + 2>/dev/null; cp -rf /workspace/mdt_src/mdt/* /workspace/mdt_policy/mdt/; cp -rf /workspace/mdt_src/calvin_env/calvin_env/* /workspace/mdt_policy/calvin_env/calvin_env/; touch /workspace/mdt_policy/calvin_env/calvin_env/__init__.py; bash /workspace/algorithm/run_mdt_tea_eval.sh /workspace/tea_q.pt 2>&1 | tee /workspace/algorithm/results/mdt_34task_${STAMP}/tea.log"

echo "[34task] tea launched. waiting for completion..."
while docker ps --format '{{.Names}}' | grep -q "^mdt_34task_tea_${STAMP}$"; do sleep 60; done
echo "[34task] tea done at $(date). logs: ${LOG_BASE}/tea.log"

echo "[34task] Phase 3: aggregate 34-task SR (parse stdout logs)"
python3 "${REPO_ROOT}/algorithm/aggregate_calvin_34task.py" \
    --base "${LOG_BASE}/base.log" \
    --tea "${LOG_BASE}/tea.log" \
    --output "${LOG_BASE}/aggregate.json" \
    2>&1 | tee "${LOG_BASE}/aggregate.log"

echo "[34task] ALL DONE. output: ${LOG_BASE}/"
