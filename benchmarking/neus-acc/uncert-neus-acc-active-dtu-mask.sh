#!/bin/bash
DATASETS=("scan24" "scan37" "scan40" "scan55" "scan63" "scan65" "scan69" "scan83" "scan97" "scan105" "scan106" "scan110" "scan114" "scan118" "scan122")
for dataset in "${DATASETS[@]}"; do
ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 60001 --pipeline.topk_iter 1000 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 3000 --trainer.steps_per_eval_batch 3000 --trainer.steps_per_save 15000 \
    --pipeline.model.steps_warmup 512 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 60000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 60001 \
 --pipeline.model.acquisition active --pipeline.model.choose_multi_cam topk \
 --pipeline.model.uncert_rgb_loss_mult 0.001 --pipeline.model.uncert_beta_loss_mult 0.01 \
 --pipeline.model.overwrite_near_far_plane True \
      --pipeline.model.background_color="black" --pipeline.model.scene_contraction_norm l2 \
      --pipeline.model.cone_angle 0.0 --pipeline.model.alpha_sample_thre 0.0 \
      --pipeline.model.near_plane 0.01 --pipeline.model.grid_levels 1 \
      --pipeline.model.pdb_debug False --pipeline.model.background-model none \
      --pipeline.datamanager.num_topk_images 2 \
      --pipeline.model.sdf-field.freq_reg_end 30000 \
      --pipeline.model.sdf-field.frequency_regularizer True \
    --experiment-name "uncert_neus-acc-active-topk-60k1k-Freq30k-total10Mask-${dataset}" \
    sdfstudio-data --data "data/dtu/${dataset}" --include_foreground_mask True \
    --indices_file "outputs/uncert_neus-acc-active-topk-60k1k-Freq30k-total10Mask-${dataset}/indices.txt"
done
wait
echo "Done."

for dataset in "${DATASETS[@]}"; do
ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 120001 --pipeline.topk_iter 2000 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 3000 --trainer.steps_per_eval_batch 3000 --trainer.steps_per_save 40000 \
    --pipeline.model.steps_warmup 1024 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 120000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 120001 \
    --optimizers.fields.scheduler.warm_up_end 1000 \
 --pipeline.model.acquisition active --pipeline.model.choose_multi_cam topk \
 --pipeline.model.uncert_rgb_loss_mult 0.001 --pipeline.model.uncert_beta_loss_mult 0.01 \
 --pipeline.model.overwrite_near_far_plane True \
      --pipeline.model.background_color="black" --pipeline.model.scene_contraction_norm l2 \
      --pipeline.model.cone_angle 0.0 --pipeline.model.alpha_sample_thre 0.0 \
      --pipeline.model.near_plane 0.01 --pipeline.model.grid_levels 1 \
      --pipeline.model.pdb_debug False --pipeline.model.background-model none \
      --pipeline.datamanager.num_topk_images 4 \
    #   --pipeline.model.sdf-field.freq_reg_end 30000 \
    #   --pipeline.model.sdf-field.frequency_regularizer True \
    --experiment-name "uncert_neus-acc-active-topk-120k2k-Freq30k-total20Mask-${dataset}" \
    sdfstudio-data --data "data/dtu/${dataset}" --include_foreground_mask True \
    --indices_file "outputs/uncert_neus-acc-active-topk-120k2k-Freq30k-total20Mask-${dataset}/indices.txt"
done
wait
echo "Done."