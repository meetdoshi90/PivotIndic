BASE="$PWD"
#export LOCAL_RANK='0,1,2,3'
export CUDA_VISIBLE_DEVICES='2,3,4,7'

METHOD=""

OUT_DIR="$BASE/model-4m-2e1d$METHOD-en-hi-ks/"
mkdir -p $OUT_DIR

SRC_1_FILE="${BASE}/../../data/train/ks/train.$1"
SRC_2_FILE="${BASE}/../../data/train/ks/train.$2"
TGT_FILE="${BASE}/../../data/train/ks/train.$3"

# SRC_1_FILE="${BASE}/../dummy_data/test.$1"
# SRC_2_FILE="${BASE}/../dummy_data/test.$2"
# TGT_FILE="${BASE}/../dummy_data/test.$3"

TEST_SRC_1_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$1"
TEST_SRC_2_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$2"
TEST_TGT_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$3"

BATCH_SIZE=12
GRAD_STEPS=2
LEARNING_RATE=3e-5
WARMUP=1000
SAVE_STEPS=500
EVAL_STEPS=500
NUM_EPOCHS=1

if [ ! -d $OUT_DIR ] 
then
  mkdir -p $OUT_DIR
fi

python3 $BASE/ft_ensemble_2e1d$METHOD.py \
    --output_dir $OUT_DIR \
    --do_train \
    --do_eval \
    --logging_strategy 'steps'\
    --logging_dir './logs-test/'\
    --logging_steps 1 \
    --seed 42 \
    --overwrite_output_dir \
    --dropout_rate 0.1  \
    --warmup_steps $WARMUP \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type 'constant_with_warmup' \
    --weight_decay 0.001 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --remove_unused_columns False \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --evaluation_strategy 'steps'\
    --prediction_loss_only False \
    --do_predict True \
    --wandb_project "indictrans2-2e1d$METHOD" \
    --src_1_path $SRC_1_FILE \
    --src_2_path $SRC_2_FILE \
    --tgt_path $TGT_FILE \
    --test_src_1_path $TEST_SRC_1_FILE \
    --test_src_2_path $TEST_SRC_2_FILE \
    --test_tgt_path $TEST_TGT_FILE \
    --save_total_limit 2 \
    --load_best_model_at_end \
    --gradient_accumulation_steps $GRAD_STEPS \
    --report_to wandb \
    --predict_with_generate True \
    --generation_max_length 512 \
    --generation_num_beams 1 \
    --max_grad_norm 1.0 \
    --alpha $4 \
    --beta $5 \
    --num_train_epochs $NUM_EPOCHS \
    --save_safetensors True \
    --metric_for_best_model 'eval_bleu' \
    --greater_is_better True \
    --bf16 True \
    --bf16_full_eval True \
    --dataloader_num_workers 64 \
    # --replace 'late' \
    # --replace_type 'random' \
    # --replace_prob 0.2 \
    # --max_steps $MAX_STEPS \