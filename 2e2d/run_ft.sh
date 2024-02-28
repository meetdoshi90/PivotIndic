BASE="$PWD"
export CUDA_VISIBLE_DEVICES='3'
OUT_DIR_NAME=$1

OUT_DIR="$BASE/model/${OUT_DIR_NAME}"
mkdir -p $OUT_DIR

SRC_1_FILE="${BASE}/../dummy_data/test.eng_Latn"
SRC_2_FILE="${BASE}/../dummy_data/test.hin_Deva"
TGT_FILE="${BASE}/../dummy_data/test.brx_Deva"

BATCH_SIZE=16
GRAD_STEPS=2
LEARNING_RATE=3e-6
WARMUP=1000
SAVE_STEPS=5000
EVAL_STEPS=5000
NUM_EPOCHS=1

if [ ! -d $OUT_DIR ] 
then
  mkdir -p $OUT_DIR
fi

python $BASE/ft_ensemble_2e2d.py \
    --output_dir $OUT_DIR \
    --do_train \
    --do_eval \
    --logging_strategy 'steps'\
    --logging_dir './logs/'\
    --logging_steps 1 \
    --seed 42 \
    --overwrite_output_dir \
    --dropout_rate 0.1  \
    --warmup_steps $WARMUP \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type 'constant_with_warmup' \
    --weight_decay 0.0001 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --remove_unused_columns False \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --evaluation_strategy 'steps'\
    --prediction_loss_only False \
    --do_predict True \
    --wandb_project 'indictrans2-2e2d-en-hi-brx' \
    --src_1_path $SRC_1_FILE \
    --src_2_path $SRC_2_FILE \
    --tgt_path $TGT_FILE \
    --save_total_limit 3 \
    --load_best_model_at_end \
    --gradient_accumulation_steps $GRAD_STEPS \
    --report_to wandb \
    --max_grad_norm 5.0 \
    --num_train_epochs $NUM_EPOCHS \
    # --max_steps $MAX_STEPS \