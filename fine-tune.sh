
export WANDB_DISABLED=true

python torch_dist_run.py main.py \
  -d train_data/ETTh1_train.jsonl \
  --precision bf16 \
  --learning_rate 1e-4 \
  --min_learning_rate 5e-5 \
  --lr_scheduler_type cosine \
  --attn_implementation flash_attention_2 \
  --model pretrained_model/TimeMoE-50M \
  --global_batch_size 128 \
  --normalization none \
  --micro_batch_size 4 \
  --stride 1 \
  --num_train_epochs 1.0 \
  --logging_steps 1 \
