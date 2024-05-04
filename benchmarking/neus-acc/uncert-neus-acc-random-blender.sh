DATASETS=("mic" "ficus" "chair" "hotdog" "materials" "drums" "ship" "lego")
for dataset in "${DATASETS[@]}"; do
    ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 20001 --pipeline.topk_iter 4000 \
    --pipeline.model.background_color="white" \
    --trainer.steps_per_save 4000 --pipeline.model.background-model none \
    --pipeline.datamanager.train_num_rays_per_batch 512 \
    --pipeline.model.acquisition random \
    --pipeline.model.uncert_rgb_loss_mult 0.001 --pipeline.model.uncert_beta_loss_mult 0.01 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 2000 --trainer.steps_per_eval_batch 2000 \
    --experiment-name "uncert_neus-acc-random-${dataset}" \
    blender-data --data "data/blender/${dataset}"
done
wait
echo "Done."

ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
 --vis tensorboard --pipeline.model.background_color="white" \
 --trainer.steps_per_save 4000 --pipeline.model.background-model none \
 --pipeline.model.acquisition random \
 --pipeline.model.uncert_rgb_loss_mult 0.001 --pipeline.model.uncert_beta_loss_mult 0.01 \
 --trainer.steps_per_eval_image 2000 --trainer.steps_per_eval_batch 2000 \
 --experiment-name uncert_neus-acc-random-lr3e4-lego \
 blender-data --data data/blender/lego

