#!/bin/bash

date
tag=$(date +'%Y-%m-%d')

echo "DATASET NAME is $3"
echo "VISUALIZE in $2"
echo "Method name is $1"

method_opts=()
if [ "$1" = "nerfacto" ]; then
    # https://github.com/nerfstudio-project/nerfstudio/issues/806#issuecomment-1284327844
    method_opts=(--pipeline.model.near-plane 2. --pipeline.model.far-plane 6. --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-average-appearance-embedding False)
fi

ns-train "$1" "${method_opts[@]}" --data="data/blender/$3" --experiment-name="blender_$3_${tag}" \
             --trainer.relative-model-dir=nerfstudio_models/ \
             --pipeline.model.sdf-field.inside-outside=False \
             --pipeline.model.background_color="white" \
             --logging.local-writer.enable=True  \
             --pipeline.model.near_plane 2.0 --pipeline.model.far_plane 6.0 \
             --pipeline.model.overwrite_near_far_plane True \
            --trainer.steps_per_eval_image 5000 --pipeline.model.background_model none \
             --logging.enable-profiler=False \
             --vis tensorboard \
             --timestamp "$timestamp" \
             blender-data & GPU_PID[$idx]=$!