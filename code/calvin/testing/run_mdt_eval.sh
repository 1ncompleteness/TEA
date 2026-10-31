#!/bin/bash
# Run MDT evaluation on pretrained mdtv-1-d weights (D→D).
# Expected: 3.67-3.75 avg chain length.

set -e

# Runtime fixes
pip install "moviepy==1.0.3" "numpy<2" wandb termcolor gdown -q 2>/dev/null || true

# Stub pyhash with Python built-in hash (only fnv1_32 used in utils.py)
mkdir -p /tmp/pyhash_shim
cat > /tmp/pyhash_shim/pyhash.py << 'EOF'
import hashlib as _h
class fnv1_32:
    def __call__(self, s):
        return int(_h.md5(str(s).encode()).hexdigest()[:8], 16)
EOF
export PYTHONPATH=/tmp/pyhash_shim:$PYTHONPATH

PRETRAINED=/workspace/mdt_pretrained/mdtv-1-d
DATASET=/workspace/calvin_dataset/task_D_D
VOLTRON_CACHE=/workspace/voltron_cache
MDT_ROOT=/workspace/mdt_policy

mkdir -p $VOLTRON_CACHE

cd $MDT_ROOT
python mdt/evaluation/mdt_evaluate.py \
    dataset_path=$DATASET \
    train_folder=$PRETRAINED \
    model_name=mdtv_1_d_reproduce \
    num_sequences=100 \
    num_videos=0 \
    render=False \
    log_wandb=False \
    sampler_type=ddim \
    multistep=10 \
    num_sampling_steps=10 \
    sigma_min=0.001 \
    sigma_max=80 \
    noise_scheduler=exponential \
    device=0 \
    eval_cfg_overwrite.overwrite_module_cfg.voltron_cache=$VOLTRON_CACHE
