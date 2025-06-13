# to fill in the following path to extract projector for the second tuning stage!
output_model=/data/LPJ/ICML25/all_checkpoints/fine_tuning_gnn_1layer_havenllama_using_1989_57_with_lora/v1_cleaned_graph_3epoch_separate_lr_gnn8e4_proj2e4_lora2e5_rank64/fine_tuning_gnn_1layer_havenllama_using_1989_57_with_lora_v1_cleaned_graph_3epoch_separate_lr_gnn8e4_proj2e4_lora2e5_rank64.ckpt
tokenizer_path=/data/LPJ/haven_codellama
datapath=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/hiverilog_eval_expansion/with_head/clean_graph/conversations.json
graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/hiverilog_eval_expansion/with_head/clean_graph/graph.jsonl
res_path=/data/LPJ/ICML25/GraphCoder/eval_result/HiVerilog_eval_expansion_result/fine_tune_1layer_havenllama_using_1989_57_with_lora/v1_cleaned_graph_3epoch_separate_lr_gnn8e4_proj2e4_lora2e5_rank64/tmp0.2
num_gpus=4
bert_path='/data/LPJ/bert/bert-L12-H128-uncased'
bert_tokenizer_max_length=25
# conv_mode=qwen
conv_mode=graphchat_v1
bf16=True
f16=False
output_file_name='eval_res'
model_max_length=6912
n_pass_k=10
use_trained_gnn=True
lora_enable=True
lora_r=64
lora_alpha=128

temperature=0.2
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
    --lora_enable ${lora_enable} \
    --lora_r ${lora_r}\
    --temperature ${temperature}