# to fill in the following path to run the first stage of our GraphGPT!
model_path=/data/LPJ/new_CodeLlama-7b-Instruct-hf
instruct_ds=/data/LPJ/ICML25/graphgpt_dataset/test_pretraining_dataset/embedding_text/conversations.json
graph_data_path=/data/LPJ/ICML25/graphgpt_dataset/test_pretraining_dataset/embedding_text/graph.jsonl
pretra_gnn=clip_gt_arxiv
output_model=/data/LPJ/ICML25/GraphGPT/checkpoints/pretraining_stage/v0
bert_path=/data/LPJ/bert/bert-L12-H128-uncased


python graphgpt/train/train_light.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end True \
    --bf16 False \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --real_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --fp16 False \
    --bert_path ${bert_path} \
    --bert_gpu 3 \
    --bert_tokenizer_max_length 15 \
    --gpus '0,1,2,3' \
    --freeze_backbone True \
    --lora_enable True \
