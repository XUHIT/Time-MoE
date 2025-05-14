#!/bin/bash

# 设置使用的 GPU（根据你机器情况修改）
export CUDA_VISIBLE_DEVICES=0

# 通用参数
BATCH_SIZE=512
GPUS=1  # 使用的 GPU 数量

# 模型列表（可以加入更多模型路径）
MODEL_LIST=(
  # "pretrained_model/TimeMoE-50M"
  "pretrained_model/TimeMoE-50M"
)

# 数据集列表
DATA_LIST=(
  "eval_data/ETTh1.csv"
  # "eval_data/ETTh2.csv"
  # "eval_data/ETTm1.csv"
  # "eval_data/ETTm2.csv"
  # "eval_data/Flight.csv"
  # "eval_data/Weather.csv"

)

# 预测长度列表
PRED_LEN_LIST=(96)

# 三重循环：模型 × 数据集 × 预测长度
for MODEL_PATH in "${MODEL_LIST[@]}"; do
  for DATA_PATH in "${DATA_LIST[@]}"; do
    for PRED_LEN in "${PRED_LEN_LIST[@]}"; do

      # 自动匹配 context_length（根据 run_eval.py 的逻辑）
      if [ "$PRED_LEN" -eq 96 ]; then
        CONTEXT_LEN=512
      elif [ "$PRED_LEN" -eq 192 ]; then
        CONTEXT_LEN=1024
      elif [ "$PRED_LEN" -eq 336 ]; then
        CONTEXT_LEN=2048
      elif [ "$PRED_LEN" -eq 720 ]; then
        CONTEXT_LEN=3072
      else
        CONTEXT_LEN=$((PRED_LEN * 4))
      fi

      echo ">>> MODEL: $MODEL_PATH | DATA: $DATA_PATH | PRED_LEN=$PRED_LEN | CONTEXT_LEN=$CONTEXT_LEN"

      # 启动多卡推理
      python -m torch.distributed.run \
        --nproc_per_node=$GPUS \
        run_eval.py \
        --model $MODEL_PATH \
        --data $DATA_PATH \
        --prediction_length $PRED_LEN \
        --context_length $CONTEXT_LEN \
        --batch_size $BATCH_SIZE

      echo ">>> DONE ✅ MODEL: $MODEL_PATH | DATA: $DATA_PATH | PRED_LEN=$PRED_LEN"
      echo ""
    done
  done
done
