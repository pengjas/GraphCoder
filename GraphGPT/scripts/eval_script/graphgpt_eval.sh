# to fill in the following path to extract projector for the second tuning stage!
output_model=/data/LPJ/ICML25/all_checkpoints/fine_tuning_with_eval_dataset_pretrained_gnn_prj_freeze_gnn_tuning_proj_lora/lr_3e4_batch2_70epoch/fine_tuning_with_eval_dataset_pretrained_gnn_prj_freeze_gnn_tuning_proj_lora_lr_3e4_batch2_70epoch.ckpt
tokenizer_path=/data/LPJ/new_CodeLlama-7b-Instruct-hf
datapath=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/HiVerilog_Eval/graph_as_prefix/with_module_head/availiable_for_graphcoder/conversations.json
graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/HiVerilog_Eval/graph_as_prefix/with_module_head/availiable_for_graphcoder/graph.jsonl
res_path=/data/LPJ/ICML25/GraphCoder/HiVerilog_eval_result/fine_tune_using_my_gnn_tuning_projetor_without_lora_unified_lr_8e3/v0_lr_3e4_70epoch_2batch
num_gpus=4
bert_path='/data/LPJ/bert/bert-L12-H128-uncased'
bert_tokenizer_max_length=25
conv_mode=graphchat_v1
bf16=True
f16=False
output_file_name='eval_res'
model_max_length=3072
n_pass_k=2
use_trained_gnn=True
lora_enable=True
python ./graphgpt/eval/run_graphgpt.py \
    --model_max_length ${model_max_length} \
    --output_file_name ${output_file_name} \
    --bf16 ${bf16} \
    --f16 ${f16} \
    --tokenizer_path ${tokenizer_path} \
    --conv_mode ${conv_mode} \
    --bert_tokenizer_max_length ${bert_tokenizer_max_length} \
    --bert_path ${bert_path} \
    --model_name ${output_model}  \
    --prompting_file ${datapath} \
    --graph_data_path ${graph_data_path} \
    --output_res_path ${res_path} \
    --num_gpus ${num_gpus}\
    --n_pass_k ${n_pass_k} \
    --use_trained_gnn ${use_trained_gnn} \
    --lora_enable ${lora_enable}