#! /bin/bash
# Change for multinode config
CHECKPOINT_PATH="/save/my_base_ckpt"

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=4
MP_SIZE=4


DATA_NUM_WORKERS=1

script_path=$(realpath $0)
script_dir=$(dirname $script_path)
main_dir=$(dirname $script_dir)

OPTIONS_NCCL=""
HOST_FILE_PATH="hostfile"

train_data_weights="1 1"
train_data="example/train_1.bin example/train_2.bin"
dev_data="example/dev.bin"
test_data="example/test.bin"

config_json="$script_dir/ds_config.json"

SAMPLE_LENGTH=128
gpt_options=" \
       --experiment-name train_on_special_dusincr_user \
       --model-parallel-size ${MP_SIZE} \
       --mode finetune \
       --train-iters 800 \
       --ckpt-path ${CHECKPOINT_PATH} \
       --my-sample-length ${SAMPLE_LENGTH}\
       --train-data-weights ${train_data_weights}\
       --train-data ${train_data} \
       --valid-data ${dev_data} \
       --test-data ${test_data} \
       --num-workers ${DATA_NUM_WORKERS} \
       --distributed-backend nccl \
       --lr-decay-style cosine \
       --warmup .1 \
       --fp16 \
       --save-interval 100 \
       --eval-interval 50 \
       --eval-iters 40 \
       --eval-batch-size 8 \
       --save /save \
"

gpt_options="${gpt_options}
       --deepspeed \
       --deepspeed_config ${config_json} \
"
              

run_cmd="${OPTIONS_NCCL} deepspeed --hostfile ${HOST_FILE_PATH}  --include=localhost:0,1,2,3 train.py $@ ${gpt_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
