#!/bin/bash
DATASETS=("scan24" "scan37" "scan40" "scan55" "scan63" "scan65" "scan69" "scan83" "scan97" "scan105" "scan106" "scan110" "scan114" "scan118" "scan122")
for dataset in "${DATASETS[@]}"; do
    ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 20001 --pipeline.topk_iter 500 \
    --pipeline.datamanager.eval_num_rays_per_batch 256 --pipeline.model.eval_num_rays_per_chunk 128 \
    --trainer.steps_per_eval_image 1000 --trainer.steps_per_eval_batch 1000 --trainer.steps_per_save 4000 \
    --pipeline.model.steps_warmup 256 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 20000 --pipeline.datamanager.train_num_rays_per_batch 512 \
 --pipeline.model.acquisition active --pipeline.model.choose_multi_cam topk \
 --pipeline.model.uncert_rgb_loss_mult 0.001 --pipeline.model.uncert_beta_loss_mult 0.01 \
      --pipeline.model.cone_angle 0.004 --pipeline.model.alpha_sample_thre 0.01 \
      --pipeline.model.near_plane 1.5 --pipeline.model.grid_levels 4 \
      --pipeline.model.background_color "random" --pipeline.model.disable_scene_contraction False \
    --experiment-name "uncert_neus-acc-active-topk-5e5-${dataset}" \
    sdfstudio-data --data "data/dtu/${dataset}" \
    --train_val_no_overlap True --skip_every_for_val_split 10
done
wait
echo "Done."

#       --optimizers.field_background.optimizer.lr 5e-5 \

DATASETS=("scan24" "scan37" "scan40" "scan63" "scan65" "scan122")
for dataset in "${DATASETS[@]}"; do
    ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 40001 --pipeline.topk_iter 8000 \
    --trainer.steps_per_save 8000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --pipeline.model.acquisition active --pipeline.model.choose_multi_cam dist \
    --trainer.steps_per_eval_all_images 40000 \
    --pipeline.model.uncert_rgb_loss_mult 0.001 --pipeline.model.uncert_beta_loss_mult 0.01 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 4000 --trainer.steps_per_eval_batch 4000 \
    --optimizers.field_background.optimizer.lr 5e-5 \
    --optimizers.fields.scheduler.warm_up_end 1000 \
    --optimizers.fields.scheduler.max_steps 40001 \
    --optimizers.field_background.scheduler.warm_up_end 1000 \
    --optimizers.field_background.scheduler.max_steps 40001 \
    --pipeline.model.steps_warmup 512 --pipeline.model.steps_per_grid_update 16 \
    --experiment-name "uncert_neus-acc-active-dist-40k-5e5-${dataset}" \
    sdfstudio-data --data "data/dtu/${dataset}" \
    --train_val_no_overlap True --skip_every_for_val_split 10
done
wait
echo "Done."

DATASETS=("scan24" "scan110" "scan114" "scan118" "scan37" "scan40" "scan55" "scan63" "scan65" "scan69" "scan83" "scan97" "scan105" "scan106" "scan122")
for dataset in "${DATASETS[@]}"; do
    ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 20001 --pipeline.topk_iter 1000 \
    --pipeline.datamanager.eval_num_rays_per_batch 256 --pipeline.model.eval_num_rays_per_chunk 128 \
    --trainer.steps_per_eval_image 2000 --trainer.steps_per_eval_batch 2000 --trainer.steps_per_save 4000 \
    --pipeline.model.steps_warmup 256 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 20000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 20001 \
 --pipeline.model.acquisition active --pipeline.model.choose_multi_cam topk \
 --pipeline.model.uncert_rgb_loss_mult 0.001 --pipeline.model.uncert_beta_loss_mult 0.01 \
 --pipeline.model.uncert_sigma_loss_mult 0.0 --pipeline.model.uncertainty_net True \
 --pipeline.model.overwrite_near_far_plane True \
      --pipeline.model.background_color="black" --pipeline.model.scene_contraction_norm inf \
      --pipeline.model.cone_angle 0.11 --pipeline.model.alpha_sample_thre 0.001 \
      --pipeline.model.near_plane 0.05 --pipeline.model.grid_levels 2 \
      --pipeline.model.sphere_masking True --pipeline.model.maintain_aabb False \
      --pipeline.model.grid_bg_resolution 64 --optimizers.field_background.optimizer.lr 5e-5 \
      --optimizers.field_background.scheduler.max_steps 20001 \
      --pipeline.datamanager.num_topk_images 2 --pipeline.model.fg_mask_loss_mult 0.0 \
    --experiment-name "uncert_neus-acc-active-topk-20k1k-total10-2Labsph-5e5-${dataset}" \
    sdfstudio-data --data "data/dtu/${dataset}" --include_foreground_mask True \
    --indices_file "outputs/uncert_neus-acc-active-topk-20k1k-total10-2Labsph-5e5-${dataset}/indices.txt"
done
wait
echo "Done."

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
 --pipeline.model.entropy_type ent --pipeline.model.grid_sampling True \
 --pipeline.model.kimera_type none --pipeline.model.overwrite_near_far_plane True \
      --pipeline.model.background_color="black" --pipeline.model.scene_contraction_norm inf \
      --pipeline.model.cone_angle 0.11 --pipeline.model.alpha_sample_thre 0.001 \
      --pipeline.model.near_plane 0.05 --pipeline.model.grid_levels 2 \
      --pipeline.model.sphere_masking True --pipeline.model.maintain_aabb False \
      --pipeline.model.grid_bg_resolution 64 --optimizers.field_background.optimizer.lr 5e-5 \
      --optimizers.field_background.scheduler.max_steps 60001 \
      --pipeline.datamanager.num_topk_images 2 \
      --pipeline.model.sdf-field.freq_reg_end 30000 \
      --pipeline.model.sdf-field.frequency_regularizer True \
    --experiment-name "uncert_neus-acc-active-topk-60k1k-Freq30k-diffinit-total10-2Labsph-5e5-${dataset}" \
    sdfstudio-data --data "data/dtu/${dataset}" \
    --indices_file "outputs/uncert_neus-acc-active-topk-60k1k-Freq30k-diffinit-total10-2Labsph-5e5-${dataset}/indices.txt"
done
wait
echo "Done."