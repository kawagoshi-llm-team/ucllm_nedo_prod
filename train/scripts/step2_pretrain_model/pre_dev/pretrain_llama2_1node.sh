#!/bin/bash
# This example script is contributed by external user https://github.com/nrailgun
set -ex

######################################
# Change the below configurations here
ucllm_nedo_dev_train_dir="../.."
megatron_deepspeed_dir="${ucllm_nedo_dev_train_dir}/Megatron-DeepSpeed"
echo "ucllm_nedo_dev_train_dir = ${ucllm_nedo_dev_train_dir}"
echo "megatron_deepspeed_dir = ${megatron_deepspeed_dir}"
echo ""

output_model_dir=""
save_interval=1000

# Parses the arguments.
while [[ ${#} -gt 0 ]]; do
    case ${1} in
        # Shifts twice for option that takes an argument.
        --output_model_dir) output_model_dir=${2}; shift ;;
        --save_interval) save_interval=${2}; shift ;;
        *) echo "Unknown parameter passed: ${1}"; exit 1 ;;
    esac
    # Shifts once per loop to move to the next key/value.
    shift
done

# Modifies the arguments.
output_model_dir="${output_model_dir%/}"  # Removes a trailing slash "/" if it exists.

DATASET_1=../../../../../../persistentshare/storage/team_kawagoshi/nishijima/datasets/small_data_text_document
DATASET="1 ${DATASET_1}"
TOKENIZER_PATH="${ucllm_nedo_dev_train_dir}/dataset/code10k_en20k_ja30k.ver2.1.model"
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=106

TP=1
PP=1
ZERO_STAGE=0
no_pp="false"
## Total number of GPUs.
num_gpus_pernode=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
#$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
num_node=1
#"${NHOSTS}"
num_gpus=$((${num_gpus_pernode} * ${num_node}))
## Data parallel size.
dp_size=$(( ${num_gpus} / ${pp_size} / ${mp_size} ))

## Whether or not log optimizer states (norms, max abs values) to tensorboard.
## This is not required for training and might save GPU memory when turned off.
log_optimizer_state="true"
###############################################################################
### Output and data configs
current_time=$(date "+%Y.%m.%d_%H.%M.%S")
host="${HOSTNAME}"
seed=1234
#num_workers=0
GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -1)
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

##Llama 0.3B
HIDDEN_SIZE=1024 # e.g. llama-13b: 5120: 2048
FFN_HIDDEN_SIZE=4096 # e.g. llama-13b: 13824 : 11008
NUM_LAYERS=12 # e.g. llama-13b: 40 : 32 
NUM_HEADS=16 # e.g. llama-13b: 40 : 32
SEQ_LENGTH=2048 #: 2048
NUM_KV_HEADS=4 # llama2 70B uses GQA :32


##Llama 7B
#HIDDEN_SIZE=2048 # e.g. llama-13b: 5120: 2048
#FFN_HIDDEN_SIZE=11008 # e.g. llama-13b: 13824 : 11008
#NUM_LAYERS=32 # e.g. llama-13b: 40 : 32 
#NUM_HEADS=32 # e.g. llama-13b: 40 : 32
#SEQ_LENGTH=4096 #: 2048
#NUM_KV_HEADS=32 # llama2 70B uses GQA :32

init_std=0.02
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=256 # e.g. llama: 4M tokens
TRAIN_STEPS=25000 # e.g. llama: 1T tokens / 4M tokens_per_batch = 250000 steps
LR=3e-4
MIN_LR=3e-6
LR_WARMUP_STEPS=500
WEIGHT_DECAY=0.1
GRAD_CLIP=1

## Activation checkpointing saves GPU memory, but reduces training speed
# activation_checkpoint="true"
activation_checkpoint="false"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################

prescale_grad="true"
jobname="llama2_${model_size}B_tok${train_tokens_in_billion}B"
jobname="${jobname}_lr${lr}_min${min_lr}_w${lr_warmup_tokens_in_million}M_d${lr_decay_tokens_in_billion}B_${lr_decay_style}"
jobname="${jobname}_gbs${global_batch_size}_mbs${batch_size}_g${num_gpus}"
if [[ $zero_stage -gt 0 ]]; then
    jobname="${jobname}_z${zero_stage}"
    prescale_grad="false"
fi
if [[ $mp_size -gt 1 ]]; then
    jobname="${jobname}_mp${mp_size}"
fi
if [ "${no_pp}" = "false" ]; then
    jobname="${jobname}_pp${pp_size}"
fi
jobname="${jobname}_seed${seed}_rebase"

username=$(whoami)
log_path="${output_model_dir}/log"
CHECKPOINT_PATH="${output_model_dir}/checkpoint/${jobname}"
tensorboard_path="${output_model_dir}/tensorboard/${jobname}_${host}_${current_time}"
deepspeed_config_dir="${output_model_dir}/deepspeed_config"
mkdir -p ${log_path}
mkdir -p ${CHECKPOINT_PATH}
mkdir -p ${tensorboard_path}
mkdir -p ${deepspeed_config_dir}
###############################################################################

DS_CONFIG="${deepspeed_config_dir}/ds_config_gbs${global_batch_size}_mbs${batch_size}_log${log_interval}_zero${zero_stage}.json"
template_json="${megatron_deepspeed_dir}/examples_deepspeed/rebase/ds_config_gpt_TEMPLATE.json"
sed "s/GBSIZE/${global_batch_size}/" ${template_json} \
    | sed "s/MBSIZE/${batch_size}/" \
    | sed "s/LOG_INTERVAL/${log_interval}/" \
    | sed "s/ZERO_STAGE/${zero_stage}/" \
    | sed "s/PRESCALE_GRAD/${prescale_grad}/" \
      > ${DS_CONFIG}


### change project your wandb project name
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 10,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "bf16": {
    "enabled": true
  },
  "wall_clock_breakdown": false,
  "wandb": {
    "enabled": true,
    "project": "llama2"
  }
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"

if [ "${activation_checkpoint}" = "true" ]; then
  ds_args="--deepspeed-activation-checkpointing ${ds_args}"

  ## old argument for recomputing the transformer layer
  # ds_args="--checkpoint-activations ${ds_args}"

  ## new argument for recomputing the transformer layer
  ds_args="--recompute-granularity full --recompute-method uniform ${ds_args}"
  ## new argument for recomputing only the attention layer
  # ds_args="--recompute-granularity selective ${ds_args}"
fi

if [[ "${no_pp}" = "true" ]]; then
  ds_args=="--no-pipeline-parallel ${ds_args}"
fi

iteration_file="$checkpoint_path/latest_checkpointed_iteration.txt"
iteration_file_2="$checkpoint_path/latest"
iteration=0
for (( node = 0; node <= num_node-1; node++ ))
do
    if $(ssh -q worker-"$node" "test -f \"$iteration_file\""); then
        local_iteration=$(ssh -q worker-"$node" cat $iteration_file)
        iteration=$(( ${local_iteration} > ${iteration} ? ${local_iteration} :  ${iteration} ))
    fi
done
if [[ $iteration -gt 0 ]]; then
    iteration_2="global_step${iteration}"
    ds_ssh "echo $iteration > $iteration_file"
    ds_ssh "echo $iteration_2 > $iteration_file_2"
fi

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


torchrun $DISTRIBUTED_ARGS \
       ${megatron_deepspeed_dir}/pretrain_gpt.py \
       --tensor-model-parallel-size $TP \
       --pipeline-model-parallel-size $PP \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --ffn-hidden-size $FFN_HIDDEN_SIZE \
       --num-attention-heads $NUM_HEADS \
       --micro-batch-size $MICRO_BATCH_SIZE \
       --global-batch-size $GLOBAL_BATCH_SIZE \
       --seq-length $SEQ_LENGTH \
       --max-position-embeddings $SEQ_LENGTH \
       --train-iters $TRAIN_STEPS \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --init-method-std ${init_std} \
       --data-path $DATASET \
       --data-impl mmap \
       --tokenizer-type SentencePieceTokenizer \
       --tokenizer-model $TOKENIZER_PATH \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr $LR \
       --lr-decay-style cosine \
       --min-lr $MIN_LR \
       --weight-decay $WEIGHT_DECAY \
       --clip-grad $GRAD_CLIP \
       --lr-warmup-iters $LR_WARMUP_STEPS \
       --optimizer adam \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 10 \
       --save-interval $save_interval \
       --eval-interval 100 \
       --eval-iters 10 \
       --bf16 \
       --no-query-key-layer-scaling \
       --attention-dropout 0 \
       --hidden-dropout 0 \
       --use-rotary-position-embeddings \
       --untie-embeddings-and-output-weights \
       --swiglu \
       --normalization rmsnorm \
       --disable-bias-linear \
       --num-key-value-heads $NUM_KV_HEADS \
       --use-flash-attn-v2 \
       --seed $seed \
       --tensorboard-queue-size 1 \
       --log-timers-to-tensorboard \
       --log-batch-size-to-tensorboard \
       --log-validation-ppl-to-tensorboard \
       --tensorboard-dir $tensorboard_path \
       $ds_args
