BASE="$PWD"
#export LOCAL_RANK='0,1,2,3'
export CUDA_VISIBLE_DEVICES='4'
lang1=eng_Latn 
lang2=hin_Deva 
lang3=ben_Beng 
lang4=asm_Beng
lang5=brx_Deva
alpha=0.5 
beta=0.5

METHOD="conditional"
EXP="multi-pivot"
PAIR="en-hi-bn-as-bo"

OUT_DIR="$BASE/model-$EXP-$METHOD-$PAIR/"
mkdir -p $OUT_DIR

# For Multi-sourcing
SRC_1_FILE="${BASE}/../../data/train/bo/train.$lang1"
SRC_2_FILE="${BASE}/../../data/train/bo/train.$lang2"
SRC_3_FILE="${BASE}/../../data/train/bo/train.$lang3"
SRC_4_FILE="${BASE}/../../data/train/bo/train.$lang4"
TGT_FILE="${BASE}/../../data/train/bo/train.$lang5"
TEST_SRC_1_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$lang1"
TEST_SRC_2_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$lang2"
TEST_SRC_3_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$lang3"
TEST_SRC_4_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$lang4"
TEST_TGT_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$lang5"

# For 1e-1d-ft
# SRC_1_FILE="${BASE}/../../data/train/go/train.$1"
# TGT_FILE="${BASE}/../../data/train/go/train.$2"
# TEST_SRC_1_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$1"
# TEST_TGT_FILE="${BASE}/../../data/IN22/IN22-Gen/test.$2"

# SRC_1_FILE="${BASE}/../dummy_data/test.$1"
# SRC_2_FILE="${BASE}/../dummy_data/test.$2"
# TGT_FILE="${BASE}/../dummy_data/test.$3"


BATCH_SIZE=1
GRAD_STEPS=6
LEARNING_RATE=3e-5
WARMUP=1000
SAVE_STEPS=5000
EVAL_STEPS=5000
NUM_EPOCHS=1

if [ ! -d $OUT_DIR ] 
then
  mkdir -p $OUT_DIR
fi

python3 $BASE/ft_ensemble_4e1d-$METHOD.py \
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
    --wandb_project "indictrans2-$EXP-$METHOD-$PAIR" \
    --src_1_path $SRC_1_FILE \
    --src_2_path $SRC_2_FILE \
    --src_3_path $SRC_3_FILE \
    --src_4_path $SRC_4_FILE \
    --tgt_path $TGT_FILE \
    --test_src_1_path $TEST_SRC_1_FILE \
    --test_src_2_path $TEST_SRC_2_FILE \
    --test_src_3_path $TEST_SRC_3_FILE \
    --test_src_4_path $TEST_SRC_4_FILE \
    --test_tgt_path $TEST_TGT_FILE \
    --save_total_limit 4 \
    --load_best_model_at_end \
    --gradient_accumulation_steps $GRAD_STEPS \
    --report_to wandb \
    --predict_with_generate True \
    --generation_max_length 512 \
    --generation_num_beams 1 \
    --max_grad_norm 1.0 \
    --alpha $alpha \
    --beta $beta \
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