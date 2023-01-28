#!/bin/bash
MPSIZE=1
MAXSEQLEN=512
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
MY_MODEL_CKPT="/data/glm_finetune/ckpt_5" # 此处设置CKPT

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=40
TOPP=0

script_path=$(realpath $0)
script_dir=$(dirname $script_path)

# 此处设置CUDA_DEVICE
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$MPSIZE --master_port $MASTER_PORT inference_glm_e2e.py \
       --mode inference \
       --outfile output_me.txt \
       --model-parallel-size $MPSIZE \
       --my-ckpt-path $MY_MODEL_CKPT \
       $MODEL_ARGS \
       --num-beams 4 \
       --no-repeat-ngram-size 3 \
       --length-penalty 0.7 \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --output-path samples_glm_e2e_2 \
       --batch-size 1 \
       --out-seq-length 500 \
       --mode inference \
       --input-source ./e2e_test.txt \
       --outfile output_dialog_e2e.txt \
       --sampling-strategy BaseStrategy
