# to fill in the following path to extract projector for the second tuning stage!
output_model=/data/LPJ/ICML25/all_checkpoints/fine_tune_qformer_havenllama_using_1989_57_with_lora/v0_3epoch_separate_lr_gnn1e3_qformer5e4_lora4e5_rank32/haven_llama_qformer_1989_57_finetune_with_lora_3epoch_separate_lr_gnn1e3_qformer5e4_lora4e5_rank32.ckpt
tokenizer_path=/data/LPJ/haven_codellama
datapath=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/HiVerilog_expansion_eval/with_head/conversations.json
graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/HiVerilog_expansion_eval/with_head/graph.jsonl
res_path=/data/LPJ/ICML25/GraphCoder/eval_result/HiVerilog_eval_result/fine_tune_qformer_havenllama_using_1989_57_with_lora/v0_3epoch_separate_lr_gnn1e3_qformer5e4_lora4e5_rank32/tmp0.8
num_gpus=4
bert_path='/data/LPJ/bert/bert-L12-H128-uncased'
bert_tokenizer_max_length=25
# conv_mode=qwen
conv_mode=graphchat_v1
bf16=True
f16=False
output_file_name='eval_res'
model_max_length=9216
n_pass_k=15
use_trained_gnn=True
lora_enable=True
lora_r=32
temperature=0.8
num_query_tokens=24
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
    --temperature ${temperature}\
    --num_query_tokens ${num_query_tokens}