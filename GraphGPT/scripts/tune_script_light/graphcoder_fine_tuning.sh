# to fill in the following path to run the first stage of our GraphGPT!
model_path=/data/LPJ/haven_codellama
# model_path=/data/LPJ/new_CodeLlama-7b-Instruct-hf
instruct_ds=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/gpt_dataset_construction/more200_gpt4/specific_task/instr_reg/v0/with_head/conversations.json
# instruct_ds=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/conversations.json
# graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/graph_output.jsonl
graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/gpt_dataset_construction/more200_gpt4/specific_task/instr_reg/v0/with_head/graph.jsonl
pretra_gnn=clip_gt_arxiv
output_model=/data/LPJ/ICML25/all_checkpoints/fine_tuning_5layers_havenllama_specific_tasks_with_module_head/instr_reg/v0_seed0/v0_epoch4
bert_path=/data/LPJ/bert/bert-L12-H128-uncased
model_save_name=haven_llama_5layers_instr_reg_v0_epoch4
resume='/data/LPJ/ICML25/all_checkpoints/fine_tuning_5layers_havenllama_gnn_projector_with_lora_using_shuffled_acl25_gpt4_with_module_head/resume_rank64_separate_lr_gnn1e4_projector1e4_lora5e5_batch2_epoch20/haven_llama_5layers_shuffled_acl25_gpt4_with_module_head_resume_rank64_separate_gnn1e4_projector1e4_lora5e5_batch2_epoch20.ckpt'
if_resume=True
val_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/conversations.json
val_graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/graph_output.jsonl
val_early_stop_threshold=0.1
if_val=False

# tuned_proj_path=/data/LPJ/ICML25/all_checkpoints/projector/pretrain_unified_lr_8e3_gnn_projector_without_lora/projector.bin
python graphgpt/train/train_light.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --val_data_path ${val_data_path} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --val_graph_data_path ${val_graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end True \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 24 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --real_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 9216 \
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
    --freeze_gnn False \
    --use_seperate_lr True \
    --gnn_lr 1e-4 \
    --projector_lr 1e-4 \
    --llm_lr 5e-5 \
    --freeze_graph_mlp_adapter False \
    --lora_r 64 \
    --if_resume ${if_resume} \
    --resume ${resume}\
    --val_early_stop_threshold ${val_early_stop_threshold} \
    --if_val ${if_val} \
