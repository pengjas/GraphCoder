# to fill in the following path to extract projector for the second tuning stage!
output_model=/data/LPJ/ICML25/all_checkpoints/pretrain_qformer_havenqwen_using_1989_57_without_lora/v1_cleaned_graph_20epoch_separate_lr_gnn2e3_qformer5e4/pretrain_qformer_havenqwen_using_1989_57_without_lora_v1_cleaned_graph_20epoch_separate_lr_gnn2e3_qformer5e4.ckpt
tokenizer_path=/data/LPJ/haven_qwen
datapath=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/HiVerilog_expansion_eval/with_head/clean_graph/conversations.json
graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/HiVerilog_expansion_eval/with_head/clean_graph/graph.jsonl
res_path=/data/LPJ/ICML25/GraphCoder/eval_result/HiVerilog_expansion_result/pretrain_qformer_havenqwen_using_1989_57_without_lora/v1_cleaned_graph_20epoch_separate_lr_gnn2e3_qformer5e4/tmp0.2
num_gpus=4
bert_path='/data/LPJ/bert/bert-L12-H128-uncased'
bert_tokenizer_max_length=25
conv_mode=qwen
# conv_mode=graphchat_v1
bf16=True
f16=False
output_file_name='eval_res'
# model_max_length=9216
model_max_length=6912
n_pass_k=10
use_trained_gnn=True
lora_enable=False
lora_r=64
lora_alpha=128
temperature=0.2
num_query_tokens=24
load_from_ckpt=True
pretrain_input_embedding_path="/data/LPJ/ICML25/all_checkpoints/pretrain_gnn_qformer_havenllama_using_1989_57_without_lora/v0_50epoch_separate_lr_gnn1e3_qformer_5e4/haven_llama_qformer_1989_57_pretrain_without_lora_50epoch_separate_lr_gnn1e3_qformer_5e4.ckpt"
pretrain_graph_mlp_adapter=/data/LPJ/ICML25/all_checkpoints/projector/pretrain_gnn_qformer_havenllama_using_1989_57_without_lora/v0_50epoch_separate_lr_gnn1e3_qformer_5e4/projector.bin

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
    --num_query_tokens ${num_query_tokens} \
    --load_from_ckpt ${load_from_ckpt} \
    --pretrain_input_embedding_path ${pretrain_input_embedding_path} \
    --pretrain_graph_mlp_adapter ${pretrain_graph_mlp_adapter}




