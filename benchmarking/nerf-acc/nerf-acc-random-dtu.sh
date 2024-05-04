#!/bin/bash
# scan24, scan37, scan40, scan55, scan63, scan65, scan69, scan83, scan97, scan105, scan106, scan110, scan114, scan118, scan122
DATASETS=("scan24" "scan37" "scan40" "scan55" "scan63" "scan65" "scan69" "scan83" "scan97" "scan105" "scan106" "scan110" "scan114" "scan118" "scan122")
for dataset in "${DATASETS[@]}"; do
ns-train active-nerf-acc \
    --vis tensorboard --trainer.max_num_iterations 60001 --pipeline.topk_iter 1000 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 3000 --trainer.steps_per_eval_batch 3000 --trainer.steps_per_save 15000 \
    --pipeline.model.steps_warmup 512 --pipeline.model.steps_per_grid_update 16 \
    --trainer.steps_per_eval_all_images 60000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --optimizers.fields.scheduler.max_steps 60001 \
    --optimizers.fields.scheduler.warmup_steps 500 \
 --pipeline.model.acquisition random \
      --pipeline.model.background_color="black" --pipeline.model.disable_scene_contraction False \
      --pipeline.model.cone_angle 0.11 --pipeline.model.alpha_sample_thre 0.001 \
      --pipeline.model.near_plane 0.05 --pipeline.model.grid_levels 4 \
      --pipeline.datamanager.num_topk_images 2 \
      --pipeline.model.freq_reg_end 30000 \
      --pipeline.model.frequency_regularizer True --pipeline.model.posenc_len 63 \
      --pipeline.model.occ_reg_loss_mult 0.01 --pipeline.model.occ_wb_prior True \
    --experiment-name "nerf-acc-random-60k1k-Freq30k-WBtotal10-2Labsph-5e5-${dataset}" \
    sdfstudio-data --data "data/dtu/${dataset}" \
    --indices_file "outputs/nerf-acc-random-60k1k-Freq30k-WBtotal10-2Labsph-5e5-${dataset}/indices.txt"
done
wait
echo "Done."