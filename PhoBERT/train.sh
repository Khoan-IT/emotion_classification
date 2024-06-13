export lr=3e-5
export s=10
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
        --save_steps  71 \
        --logging_steps 71 \
        --num_train_epochs 1000 \
        --gpu_id 0 \
        --learning_rate $lr \
        --tuning_metric intent_acc \
        --num_sample 128 \
        --head_layer_dim 384 \
        # --additional_loss contrastiveloss \
        # --pretrained \
        # --pretrained_path phobert/7 \