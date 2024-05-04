DATASETS=("scan24" "scan37" "scan40" "scan55" "scan63" "scan65" "scan69" "scan83" "scan97" "scan105" "scan106" "scan110" "scan114" "scan118" "scan122")
for dataset in "${DATASETS[@]}"; do
    ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 20001 --pipeline.topk_iter 4000 \
    --trainer.steps_per_save 4000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --pipeline.model.acquisition random --trainer.steps_per_eval_all_images 20000 \
    --pipeline.model.uncert_rgb_loss_mult 0.001 --pipeline.model.uncert_beta_loss_mult 0.01 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 2000 --trainer.steps_per_eval_batch 2000 \
    --optimizers.field_background.optimizer.lr 5e-5 \
    --experiment-name "uncert_neus-acc-random-dtu-slow5e5-${dataset}" \
    sdfstudio-data --data "data/dtu/${dataset}" \
    --train_val_no_overlap True --skip_every_for_val_split 10
done
wait
echo "Done."

ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
    --vis tensorboard --trainer.max_num_iterations 20001 --pipeline.topk_iter 4000 \
    --trainer.steps_per_save 4000 --pipeline.datamanager.train_num_rays_per_batch 512 \
    --pipeline.model.acquisition random --trainer.steps_per_eval_all_images 20000 \
    --pipeline.model.uncert_rgb_loss_mult 0.001 --pipeline.model.uncert_beta_loss_mult 0.01 \
    --pipeline.datamanager.eval_num_rays_per_batch 512 --pipeline.model.eval_num_rays_per_chunk 256 \
    --trainer.steps_per_eval_image 2000 --trainer.steps_per_eval_batch 2000 \
    --experiment-name uncert_neus-acc-random-dtu-83 \
    sdfstudio-data --data "data/dtu/scan83" \
     --train_val_no_overlap True --skip_every_for_val_split 10