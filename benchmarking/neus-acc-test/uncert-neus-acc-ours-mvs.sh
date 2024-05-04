ns-train active_uncert_neus-acc --pipeline.model.sdf-field.inside-outside False \
 --vis tensorboard --pipeline.model.background_color="white" \
 --trainer.steps_per_save 4000 --pipeline.model.background-model none \
 --pipeline.model.uncert_rgb_loss_mult 0.01 --pipeline.model.uncert_beta_loss_mult 0.01 \
 --trainer.steps_per_eval_image 2000 --trainer.steps_per_eval_batch 2000 \
 --experiment-name active_uncert_neus-acc-BgNone-noDetach-2-2-angle-lego \
 blender-data --data data/blender/lego
