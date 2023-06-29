export lr=3e-5
export s=1
export model="lstm"
export MODEL_DIR="./checkpoint/"$model

python3 main.py --token_level word-level \
                  --model_type $model \
                  --model_dir $MODEL_DIR \
                  --data_dir ../PhoBERT/data/nor_augment_not_accent \
                  --seed $s \
                  --do_train \
                  --do_eval \
                  --save_steps 100 \
                  --logging_steps 100 \
                  --num_train_epochs 30 \
                  --gpu_id 0 \
                  --learning_rate $lr \
                  --embedding_dim 400 \
                  --max_vocab_size 10000 \
                  --hidden_size 768 \
                  --tuning_metric intent_acc
