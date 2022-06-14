for seed in 42
do
MODEL_PATH=result/my-unsup-simcse-z4adapt-mixup-V100-temp0.05-lambda0.2-bert-base-uncased_seed${seed}_test

CUDA_VISIBLE_DEVICES=2 python3 -u train.py \
    --model_name_or_path '/home/LAB/zhangyz/code/pretrained_models/bert-base-uncased' \
    --train_file data/wiki1m_for_simcse.txt \
    --eval_path data/sts-dev.tsv \
    --output_dir $MODEL_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --seed ${seed} \
    --lambdas 0.2 \
    "$@"

python3 simcse_to_huggingface.py --path=$MODEL_PATH
python evaluation.py --model_name_or_path $MODEL_PATH --pooler cls_before_pooler --task_set full --mode test
done
