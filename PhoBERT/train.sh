export lr=3e-5
export s=2
echo "${lr}"
export MODEL_DIR="./gpt_augment/cnn/"$s
echo "${MODEL_DIR}"
python3 main.py --token_level word-level \
                  --model_type phobert \
                  --model_dir $MODEL_DIR \
                  --data_dir data/nor_augment_not_accent \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 138 \
                  --logging_steps 138 \
                  --num_train_epochs 100 \
                  --gpu_id 0 \
                  --learning_rate $lr \
                  --tuning_metric intent_acc \
