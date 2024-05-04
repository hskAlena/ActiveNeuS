DATASETS=( "mic" "ficus" "chair" "hotdog" "materials" "drums" "ship" "lego")

for dataset in "${DATASETS[@]}"; do
ns-train active_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 100001 --pipeline.topk_iter 1000 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 2500 --trainer.steps_per_eval_batch 2500 --trainer.steps_per_save 20000 \
    --pipeline.model.steps_warmup 512 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 100000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 100001 \
 --pipeline.model.acquisition entropy --pipeline.model.choose_multi_cam topk \
 --pipeline.model.entropy_type no_surface --pipeline.model.grid_sampling True \
 --pipeline.model.overwrite_near_far_plane True \
      --pipeline.model.background_color="white" --pipeline.model.scene_contraction_norm l2 \
      --pipeline.model.cone_angle 0.0 --pipeline.model.alpha_sample_thre 0.0 \
      --pipeline.model.near_plane 0.01 --pipeline.model.grid_levels 1 \
      --pipeline.model.background-model none \
      --pipeline.datamanager.num_topk_images 2 \
      --pipeline.model.sdf-field.freq_reg_end 40000 \
      --pipeline.model.sdf-field.frequency_regularizer True \
      --pipeline.datamanager.precrop-iters 750 --pipeline.datamanager.precrop-frac 0.6 \
    --experiment-name "neus-acc-entropy-topk-100k1k-precrop6-Freq40k-total10-${dataset}" \
    blender-data --data "data/blender/${dataset}" --skip_every_for_val_split 8 \
    --precrop-iters 750 --precrop-frac 0.6
done
wait
echo "Done."

for dataset in "${DATASETS[@]}"; do
ns-train active_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 100001 --pipeline.topk_iter 1000 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 2500 --trainer.steps_per_eval_batch 2500 --trainer.steps_per_save 20000 \
    --pipeline.model.steps_warmup 512 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 100000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 100001 \
 --pipeline.model.acquisition random \
 --pipeline.model.overwrite_near_far_plane True \
      --pipeline.model.background_color="white" --pipeline.model.scene_contraction_norm l2 \
      --pipeline.model.cone_angle 0.0 --pipeline.model.alpha_sample_thre 0.0 \
      --pipeline.model.near_plane 0.01 --pipeline.model.grid_levels 1 \
      --pipeline.model.background-model none \
      --pipeline.datamanager.num_topk_images 2 \
      --pipeline.model.sdf-field.freq_reg_end 40000 \
      --pipeline.model.sdf-field.frequency_regularizer True \
      --pipeline.datamanager.precrop-iters 750 --pipeline.datamanager.precrop-frac 0.6 \
    --experiment-name "neus-acc-random-100k1k-precrop6-Freq40k-total10-${dataset}" \
    blender-data --data "data/blender/${dataset}" --skip_every_for_val_split 8 \
    --precrop-iters 750 --precrop-frac 0.6
done
wait
echo "Done."

######################################################################################
for dataset in "${DATASETS[@]}"; do
ns-train active_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 200001 --pipeline.topk_iter 2000 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 5000 --trainer.steps_per_eval_batch 5000 --trainer.steps_per_save 40000 \
    --pipeline.model.steps_warmup 1024 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 200000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 200001 \
    --optimizers.fields.scheduler.warm_up_end 1000 \
 --pipeline.model.acquisition entropy --pipeline.model.choose_multi_cam topk \
 --pipeline.model.entropy_type no_surface --pipeline.model.grid_sampling True \
 --pipeline.model.overwrite_near_far_plane True \
      --pipeline.model.background_color="white" --pipeline.model.scene_contraction_norm l2 \
      --pipeline.model.cone_angle 0.0 --pipeline.model.alpha_sample_thre 0.0 \
      --pipeline.model.near_plane 0.01 --pipeline.model.grid_levels 1 \
      --pipeline.model.background-model none \
      --pipeline.datamanager.num_topk_images 4 \
      --pipeline.datamanager.precrop-iters 1200 --pipeline.datamanager.precrop-frac 0.6 \
    --experiment-name "neus-acc-entropy-topk-200k2k-precropW12-total20-${dataset}" \
    blender-data --data "data/blender/${dataset}" --skip_every_for_val_split 8 \
    --precrop-iters 1200 --precrop-frac 0.6
done
wait
echo "Done."

for dataset in "${DATASETS[@]}"; do
ns-train active_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 200001 --pipeline.topk_iter 2000 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 5000 --trainer.steps_per_eval_batch 5000 --trainer.steps_per_save 40000 \
    --pipeline.model.steps_warmup 1024 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 200000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 200001 \
    --optimizers.fields.scheduler.warm_up_end 1000 \
 --pipeline.model.acquisition random \
 --pipeline.model.overwrite_near_far_plane True \
      --pipeline.model.background_color="white" --pipeline.model.scene_contraction_norm l2 \
      --pipeline.model.cone_angle 0.0 --pipeline.model.alpha_sample_thre 0.0 \
      --pipeline.model.near_plane 0.01 --pipeline.model.grid_levels 1 \
      --pipeline.model.background-model none \
      --pipeline.datamanager.num_topk_images 4 \
      --pipeline.datamanager.precrop-iters 1200 --pipeline.datamanager.precrop-frac 0.6 \
    --experiment-name "neus-acc-random-200k2k-precropW12-total20-${dataset}" \
    blender-data --data "data/blender/${dataset}" --skip_every_for_val_split 8 \
    --precrop-iters 1200 --precrop-frac 0.6
done
wait
echo "Done."