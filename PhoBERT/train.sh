export lr=3e-5
export s=3
echo "${lr}"
export MODEL_DIR="./phobert/"$s
echo "${MODEL_DIR}"
python3 main.py \
        --model_type phobert \
        --model_dir $MODEL_DIR \
        --data_dir ../data/word-level \
        --seed $s \
        --do_train \
        --do_eval \
        --train_batch_size 128 \
        --eval_batch_size 128 \
        --save_steps  25\
        --logging_steps 25 \
        --num_train_epochs 1000 \
        --gpu_id 0 \
        --learning_rate $lr \
        --tuning_metric intent_acc \
