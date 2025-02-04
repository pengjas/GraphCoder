# to fill in the following path to run the first stage of our GraphGPT!
model_path=/data/LPJ/new_CodeLlama-7b-Instruct-hf
# model_path=/data/LPJ/new_CodeLlama-7b-Instruct-hf
instruct_ds=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/conversations.json
# instruct_ds=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/conversations.json
# graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/graph_output.jsonl
graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/graph_output.jsonl
pretra_gnn=clip_gt_arxiv
output_model=/data/LPJ/ICML25/all_checkpoints/fine_tuning_with_eval_dataset_pretrained_separate_lr_gnn_prj_freeze_gnn_tuning_proj_lora/lr_3e4_batch2_70epoch
bert_path=/data/LPJ/bert/bert-L12-H128-uncased
model_save_name=fine_tuning_with_eval_dataset_pretrained_separate_lr_gnn_prj_freeze_gnn_tuning_proj_lora_lr_3e4_batch2_70epoch
tuned_proj_path=/data/LPJ/ICML25/all_checkpoints/projector/pretrain_separate_lr_gnn8e3_projector3e4_without_lora/projector.bin

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
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 70 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --real_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 3e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 3072 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to wandb \
    --fp16 False \
    --bert_path ${bert_path} \
    --bert_gpu 3 \
    --bert_tokenizer_max_length 25 \
    --gpus '0,1,2,3' \
    --freeze_backbone True \
    --lora_enable True \
    --model_save_name ${model_save_name} \
    --freeze_gnn True \
    --use_seperate_lr False \
    --gnn_lr 8e-3 \
    --projector_lr 3e-4 \
    --llm_lr 3e-5 \
    --freeze_graph_mlp_adapter False \
