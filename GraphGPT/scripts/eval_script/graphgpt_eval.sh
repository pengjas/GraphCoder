# to fill in the following path to extract projector for the second tuning stage!
output_model=/data/LPJ/ICML25/all_checkpoints/pretrain_gnn_with_tuning_projector_lora_separate_lr/v0_lr_gnn_3e2_prj_3e4_lora_3e5_2epoch_batch2/last.ckpt
tokenizer_path=/data/LPJ/Llama-2-7b-chat-hf
datapath=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/pretraining_eval/graph_as_prefix/available_for_graphcoder/conversations.json
graph_data_path=/data/LPJ/ICML25/GraphCoder/graphgpt_dataset/pretraining_eval/graph_as_prefix/available_for_graphcoder/graph.jsonl
res_path=/data/LPJ/ICML25/GraphCoder/pretraining_eval_result/train_unfreeze_gnn_with_tune_projector_lora_separate_lr_10k/v0_lr_gnn_3e2_projector_3e4_lora_3e5_2epoch_2batch
num_gpus=1
bert_path='/data/LPJ/bert/bert-L12-H128-uncased'
bert_tokenizer_max_length=25
conv_mode=graphchat_v1
bf16=True
f16=False
output_file_name='eval_res'
model_max_length=3072
n_pass_k=1
use_trained_gnn=True
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
    --use_trained_gnn ${use_trained_gnn}