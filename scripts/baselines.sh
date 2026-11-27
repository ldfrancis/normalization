#!/usr/bin/env bash
# Train a baseline with the settings of the papers. Run from the project root.
#
#   MODEL=resnet56 DATA=cifar10 ./scripts/baselines.sh                # 1 GPU
#   MODEL=resnet110 DATA=cifar100 GPU=3 ./scripts/baselines.sh        # 1 GPU, the 4th one
#   MODEL=resnet50 DATA=imagenet ./scripts/baselines.sh               # 8 GPUs
#   MODEL=plain56 DATA=cifar10 ./scripts/baselines.sh                 # plain net, no shortcuts
#   MODEL=transformer_base DATA=wmt14_en_de ./scripts/baselines.sh    # 8 GPUs
#
# MODEL: resnet20|32|44|56|110 (CIFAR), resnet18|34|50|101|152 (ImageNet),
#        plain<depth> for the plain net of a ResNet depth (same layers, no shortcuts),
#        transformer_base|transformer_big (WMT14)
# DATA:  cifar10 | cifar100 | imagenet | wmt14_en_de
# Optional: SEED (default 0), GPUS (default 1 for CIFAR, 8 otherwise),
#           GPU (which GPU a single-GPU run uses, default 0),
#           NORM (replaces the model's norm layers: dyt|dyt_channel|dysoftsign|none|batchnorm|layernorm|layernorm2d,
#                 see config/norm/; default: the model's own),
#           FIRST_LAYER_LR_MULT (ResNets: multiplier of the learning rate for layer 1,
#                     rising linearly with depth to 1 at the last layer; e.g. 0.1.
#                     Default: one rate for all layers),
#           DROPOUT (NORM=dyt|dyt_channel|dysoftsign|dyscale: dropout on the norm's output, e.g. 0.1),
#           DROPOUT_CHANNELS (as DROPOUT, but zeroing whole channels: Dropout2d; ResNets),
#           OUT (run directory, default outputs/baselines/<data>_<model>[_<norm>][_lrmult<m>][_drop<p>|_drop2d<p>]_seed<seed>).
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
# A plain net trains with the settings of the ResNet of the same depth.
if [[ "$MODEL" == plain* ]]; then
  overrides+=("model.residual=false")
fi
case "$DATA:$MODEL" in
  cifar10:resnet* | cifar100:resnet* | cifar10:plain* | cifar100:plain*)
    experiment=${DATA}_resnet
    depth=${MODEL#resnet}
    depth=${depth#plain}
    overrides+=("model.depth=$depth")
    # He et al.: ResNet-110 starts at lr 0.01 for about 400 iterations (plain-110 too).
    [[ "$depth" == 110 ]] && overrides+=("trainer.scheduler.warmup_steps=400")
    default_gpus=1
    ;;
  imagenet:resnet* | imagenet:plain*)
    experiment=imagenet_resnet
    depth=${MODEL#resnet}
    overrides+=("model.depth=${depth#plain}")
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
setting=${DATA}_${MODEL}${NORM:+_$NORM}${FIRST_LAYER_LR_MULT:+_lrmult$FIRST_LAYER_LR_MULT}${DROPOUT:+_drop$DROPOUT}${DROPOUT_CHANNELS:+_drop2d$DROPOUT_CHANNELS}
[[ -n "${NORM:-}" ]] && overrides+=("norm=$NORM")
[[ -n "${DROPOUT:-}" ]] && overrides+=("model.norm_layer.dropout=$DROPOUT")
if [[ -n "${DROPOUT_CHANNELS:-}" ]]; then
  [[ -n "${DROPOUT:-}" ]] && { echo "set DROPOUT or DROPOUT_CHANNELS, not both" >&2; exit 1; }
  overrides+=("model.norm_layer.dropout=$DROPOUT_CHANNELS" "model.norm_layer.dropout_channels=true")
fi
[[ -n "${FIRST_LAYER_LR_MULT:-}" ]] &&
  overrides+=("trainer.first_layer_lr_multiplier=$FIRST_LAYER_LR_MULT")
OUT=${OUT:-outputs/baselines/${setting}_seed${SEED}}
args=("+experiment=$experiment" ${overrides[@]+"${overrides[@]}"} "seed=$SEED" "hydra.run.dir=$OUT"
      "wandb.group=$setting" "$@")

echo "training $MODEL on $DATA with $GPUS GPU(s), output in $OUT"
if [[ "$GPUS" == 1 ]]; then
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${GPU:-0}} python nstudy/main.py "${args[@]}"
else
  torchrun --standalone --nproc_per_node="$GPUS" nstudy/main.py "${args[@]}"
fi
