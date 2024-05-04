#!/bin/bash
# scan24, scan37, scan40, scan55, scan63, scan65, scan69, scan83, scan97, scan105, scan106, scan110, scan114, scan118, scan122
DATASETS=("scan24" "scan37" "scan40" "scan55" "scan63" "scan65" "scan69" "scan83" "scan97" "scan105" "scan106" "scan110" "scan114" "scan118" "scan122")
for dataset in "${DATASETS[@]}"; do
    ns-train active-nerf-acc \
    --vis tensorboard --trainer.max_num_iterations 20001 --pipeline.topk_iter 4000 \
    --pipeline.datamanager.eval_num_rays_per_batch 256 --pipeline.model.eval_num_rays_per_chunk 128 \
    --trainer.steps_per_eval_image 2000 --trainer.steps_per_eval_batch 2000 --trainer.steps_per_save 4000 \
    --pipeline.model.steps_warmup 256 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 20000 --pipeline.datamanager.train_num_rays_per_batch 512 \
 --pipeline.model.acquisition entropy --pipeline.model.choose_multi_cam topk \
 --pipeline.model.grid_sampling False \
      --pipeline.model.background_color="random" --pipeline.model.disable_scene_contraction False \
      --pipeline.model.cone_angle 0.004 --pipeline.model.alpha_sample_thre 0.01 \
      --pipeline.model.near_plane 0.05 --pipeline.model.grid_levels 4 \
    --experiment-name "nerf-acc-entropy-topk-xgrid-${dataset}" \
    sdfstudio-data --data "data/dtu/${dataset}" \
    --train_val_no_overlap True --skip_every_for_val_split 10
done
wait
echo "Done."

