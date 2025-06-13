# to fill in the following path to run the first stage of our GraphGPT!
model_path=/data/LPJ/haven_codellama
# model_path=/data/LPJ/new_CodeLlama-7b-Instruct-hf
instruct_ds=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/gpt_dataset_construction/stage2_1989_57/with_head/cleaned_graph/conversations.json
# instruct_ds=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/conversations.json
# graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/graph_output.jsonl
graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/gpt_dataset_construction/stage2_1989_57/with_head/cleaned_graph/graph.jsonl
pretra_gnn=clip_gt_arxiv
output_model=/data/LPJ/ICML25/all_checkpoints/fine_tuning_gnn_1layer_havenllama_using_1989_57_with_lora/v1_cleaned_graph_3epoch_separate_lr_gnn8e4_proj2e4_lora2e5_rank64
bert_path=/data/LPJ/bert/bert-L12-H128-uncased
model_save_name=fine_tuning_gnn_1layer_havenllama_using_1989_57_with_lora_v1_cleaned_graph_3epoch_separate_lr_gnn8e4_proj2e4_lora2e5_rank64
if_val=False
resume='/data/LPJ/ICML25/all_checkpoints/fine_tuning_5layers_havenllama_similar_logic_with_module_head/instr_reg/v1/epoch10/haven_llama_5layers_similar_logic_instr_reg_v1_epoch10.ckpt'
if_resume=False
val_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/conversations.json
val_graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/train_with_eval_dataset/with_module_head/graph_as_prefix/availiable_for_graphcoder/graph_output.jsonl
val_early_stop_threshold=0.1

tuned_proj_path=/data/LPJ/ICML25/all_checkpoints/projector/pretrain_gnn_1layers_havenllama_using_1989_57_without_lora/v1_cleaned_graph_20epoch_separate_lr_gnn2e3_proj5e4/projector.bin
pretrain_input_embedding_path=/data/LPJ/ICML25/all_checkpoints/pretrain_1layer_havenllama_using_1989_57_without_lora/v1_cleaned_graph_20epoch_separate_lr_gnn2e3_proj5e4/pretrain_1layer_havenllama_using_1989_57_without_lora_v1_cleaned_graph_20epoch_separate_lr_gnn2e3_proj5e4.ckpt



python graphgpt/train/train_light.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --pretrain_graph_mlp_adapter ${tuned_proj_path} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end True \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
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
    --gnn_lr 8e-4 \
    --projector_lr 2e-4 \
    --llm_lr 2e-5 \
    --freeze_graph_mlp_adapter False \
    --lora_r 64 \
    --if_resume ${if_resume} \
    --resume ${resume}\
    --val_early_stop_threshold ${val_early_stop_threshold} \
    --if_val ${if_val} \
    --pretrain_input_embedding_path ${pretrain_input_embedding_path} \
