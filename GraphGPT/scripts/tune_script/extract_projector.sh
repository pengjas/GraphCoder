# to fill in the following path to extract projector for the second tuning stage!
src_model=/data/LPJ/ICML25/all_checkpoints/pretrain_gnn_with_tuning_projector_without_lora_unified_lr/v0_better_balanced_lr_8e3_2epoch_batch2/better_balanced_lr_8e3_2epoch_batch2.ckpt
output_proj=/data/LPJ/ICML25/all_checkpoints/projector/pretrain_unified_lr_8e3_gnn_projector_without_lora/projector.bin

python ./scripts/extract_graph_projector.py \
  --model_name_or_path ${src_model} \
  --output ${output_proj}