DATASETS=("mic" "ficus" "chair" "hotdog" "materials" "drums" "ship" "lego")

for dataset in "${DATASETS[@]}"; do
ns-train active-nerf-acc \
    --vis tensorboard --trainer.max_num_iterations 100001 --pipeline.topk_iter 1000 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 2500 --trainer.steps_per_eval_batch 2500 --trainer.steps_per_save 20000 \
    --pipeline.model.steps_warmup 512 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 100000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 100001 \
 --pipeline.model.acquisition random \
      --pipeline.model.background_color="white" --pipeline.model.disable_scene_contraction True \
      --pipeline.model.cone_angle 0.0 --pipeline.model.alpha_sample_thre 0.0 \
      --pipeline.model.near_plane 0.01 --pipeline.model.grid_levels 1 \
      --pipeline.datamanager.num_topk_images 2 \
      --pipeline.model.freq_reg_end 40000 \
      --pipeline.model.frequency_regularizer True --pipeline.model.posenc_len 63 \
      --pipeline.datamanager.precrop-iters 750 --pipeline.datamanager.precrop-frac 0.6 \
    --experiment-name "nerf-acc-random-100k1k-precrop6-Freq40k-total10-${dataset}" \
    blender-data --data "data/blender/${dataset}" --skip_every_for_val_split 8 \
    --precrop-iters 750 --precrop-frac 0.6
done
wait
echo "Done."

# for dataset in "${DATASETS[@]}"; do
# ns-train active-nerf-acc \
#     --vis tensorboard --trainer.max_num_iterations 100001 --pipeline.topk_iter 1000 \
#     --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
#     --trainer.steps_per_eval_image 2500 --trainer.steps_per_eval_batch 2500 --trainer.steps_per_save 20000 \
#     --pipeline.model.steps_warmup 512 --pipeline.model.steps_per_grid_update 16 \
#     --trainer.steps_per_eval_all_images 100000 --pipeline.datamanager.train_num_rays_per_batch 512 \
#     --optimizers.fields.scheduler.max_steps 100001 \
#  --pipeline.model.acquisition random \
#       --pipeline.model.background_color="white" --pipeline.model.disable_scene_contraction True \
#       --pipeline.model.cone_angle 0.0 --pipeline.model.alpha_sample_thre 0.0 \
#       --pipeline.model.near_plane 0.01 --pipeline.model.grid_levels 1 \
#       --pipeline.datamanager.num_topk_images 2 \
#       --pipeline.model.freq_reg_end 40000 \
#       --pipeline.model.frequency_regularizer True --pipeline.model.posenc_len 63 \
#       --pipeline.datamanager.precrop-iters 750 --pipeline.datamanager.precrop-frac 0.6 \
#       --pipeline.model.occ_reg_loss_mult 0.01 --pipeline.model.occ_wb_prior False \
#       --pipeline.model.occ_reg_range 20 \
#     --experiment-name "nerf-acc-random-100k1k-precrop6-Freq40k-Rtotal10-${dataset}" \
#     blender-data --data "data/blender/${dataset}" --skip_every_for_val_split 8 \
#     --precrop-iters 750 --precrop-frac 0.6
# done
# wait
# echo "Done."

for dataset in "${DATASETS[@]}"; do
ns-train active-nerf-acc \
    --vis tensorboard --trainer.max_num_iterations 200001 --pipeline.topk_iter 2000 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 5000 --trainer.steps_per_eval_batch 5000 --trainer.steps_per_save 40000 \
    --pipeline.model.steps_warmup 1024 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 200000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 200001 \
 --pipeline.model.acquisition random \
      --pipeline.model.background_color="white" --pipeline.model.disable_scene_contraction True \
      --pipeline.model.cone_angle 0.0 --pipeline.model.alpha_sample_thre 0.0 \
      --pipeline.model.near_plane 0.01 --pipeline.model.grid_levels 1 \
      --pipeline.datamanager.num_topk_images 4 \
      --pipeline.datamanager.precrop-iters 1200 --pipeline.datamanager.precrop-frac 0.6 \
    --experiment-name "nerf-acc-random-200k2k-precropW12-total20-${dataset}" \
    blender-data --data "data/blender/${dataset}" --skip_every_for_val_split 8 \
    --precrop-iters 1200 --precrop-frac 0.6
done
wait
echo "Done."