#!/usr/bin/env bash
# Train a baseline with the settings of the papers. Run from the project root.
#
#   MODEL=resnet56 DATA=cifar10 ./scripts/baselines.sh                # 1 GPU
#   MODEL=resnet110 DATA=cifar100 GPU=3 ./scripts/baselines.sh        # 1 GPU, the 4th one
#   MODEL=resnet50 DATA=imagenet ./scripts/baselines.sh               # 8 GPUs
#   MODEL=transformer_base DATA=wmt14_en_de ./scripts/baselines.sh    # 8 GPUs
#
# MODEL: resnet20|32|44|56|110 (CIFAR), resnet18|34|50|101|152 (ImageNet),
#        transformer_base|transformer_big (WMT14)
# DATA:  cifar10 | cifar100 | imagenet | wmt14_en_de
# Optional: SEED (default 0), GPUS (default 1 for CIFAR, 8 otherwise),
#           GPU (which GPU a single-GPU run uses, default 0),
#           NORM (replaces the model's norm layers: dyt|dysoftsign|none|batchnorm|layernorm,
#                 see config/norm/; default: the model's own),
#           OUT (run directory, default outputs/baselines/<data>_<model>[_<norm>]_seed<seed>).
#
#   MODEL=resnet56 DATA=cifar10 NORM=dyt ./scripts/baselines.sh
#
# Extra arguments go to Hydra, e.g. trainer.precision=bf16, wandb.mode=offline,
# model.norm_layer.alpha_init=1.0 or resume_from=<checkpoint>.
# Runs of every seed share the wandb group <data>_<model>[_<norm>].
set -euo pipefail

: "${MODEL:?set MODEL, e.g. MODEL=resnet56}"
: "${DATA:?set DATA, e.g. DATA=cifar10}"
SEED=${SEED:-0}

overrides=()
case "$DATA:$MODEL" in
  cifar10:resnet* | cifar100:resnet*)
    experiment=${DATA}_resnet
    depth=${MODEL#resnet}
    overrides+=("model.depth=$depth")
    # He et al.: ResNet-110 starts at lr 0.01 for about 400 iterations.
    [[ "$depth" == 110 ]] && overrides+=("trainer.scheduler.warmup_steps=400")
    default_gpus=1
    ;;
  imagenet:resnet*)
    experiment=imagenet_resnet
    overrides+=("model.depth=${MODEL#resnet}")
    default_gpus=8
    ;;
  wmt14_en_de:transformer_base | wmt14_en_de:transformer_big)
    experiment=wmt14_${MODEL}
    default_gpus=8
    ;;
  *)
    echo "unsupported MODEL=$MODEL with DATA=$DATA" >&2
    exit 1
    ;;
esac

GPUS=${GPUS:-$default_gpus}
setting=${DATA}_${MODEL}${NORM:+_$NORM}
[[ -n "${NORM:-}" ]] && overrides+=("norm=$NORM")
OUT=${OUT:-outputs/baselines/${setting}_seed${SEED}}
args=("+experiment=$experiment" ${overrides[@]+"${overrides[@]}"} "seed=$SEED" "hydra.run.dir=$OUT"
      "wandb.group=$setting" "$@")

echo "training $MODEL on $DATA with $GPUS GPU(s), output in $OUT"
if [[ "$GPUS" == 1 ]]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU:-0}} python nstudy/main.py "${args[@]}"
else
  torchrun --standalone --nproc_per_node="$GPUS" nstudy/main.py "${args[@]}"
fi
