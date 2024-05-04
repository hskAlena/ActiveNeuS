  ns-train active_neus-acc --pipeline.model.sdf-field.inside-outside False \
 --vis tensorboard --pipeline.model.background_color="white" \
 --trainer.steps_per_save 4000 --pipeline.model.background-model none \
 --trainer.steps_per_eval_image 2000 --trainer.steps_per_eval_batch 2000 \
 --experiment-name active_neus-acc-BgNone-angle-highLr2e3-lego \
 blender-data --data data/blender/lego
