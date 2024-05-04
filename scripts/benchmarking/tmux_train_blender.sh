#!/bin/bash

helpFunction_launch_train()
{
   echo "Usage: $0 -m <method_name> [-v <vis>] [-s] [<gpu_list>]"
   echo -e "\t-m name of config to benchmark (e.g. mipnerf, instant_ngp)"
   echo -e "\t-v <vis>: Visualization method. <vis> can be wandb or tensorboard. Default is wandb."
   echo -e "\t-s: Launch a single training job per gpu."
   echo -e "\t<gpu_list> [OPTIONAL] list of space-separated gpu numbers to launch train on (e.g. 0 2 4 5)"
   exit 1 # Exit program after printing help
}

vis="wandb"
single=false
while getopts "m:v:s" opt; do
    case "$opt" in
        m ) method_name="$OPTARG" ;;
        v ) vis="$OPTARG" ;;
        s ) single=true ;;
        ? ) helpFunction ;; 
    esac
done

if [ -z "${method_name+x}" ]; then
    echo "Missing method name"
    helpFunction_launch_train
fi
method_opts=()
if [ "$method_name" = "nerfacto" ]; then
    # https://github.com/nerfstudio-project/nerfstudio/issues/806#issuecomment-1284327844
    method_opts=(--pipeline.model.near-plane 2. --pipeline.model.far-plane 6. --pipeline.datamanager.camera-optimizer.mode off --pipeline.model.use-average-appearance-embedding False)
fi

shift $((OPTIND-1))

# Deal with gpu's. If passed in, use those.
GPU_IDX=("$@")
if [ -z "${GPU_IDX[0]+x}" ]; then
    echo "no gpus set... finding available gpus"
    # Find available devices
    num_device=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    START=0
    END=${num_device}-1
    GPU_IDX=()

    for (( id=START; id<=END; id++ )); do
        free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i $id | grep -Eo '[0-9]+')
        if [[ $free_mem -gt 10000 ]]; then
            GPU_IDX+=( "$id" )
        fi
    done
fi
echo "available gpus... ${GPU_IDX[*]}"

DATASETS=("mic" "ficus" "chair" "hotdog" "materials" "drums" "ship" "lego")
date
tag=$(date +'%Y-%m-%d')

COUNT=0
ENDCOUNT=3
len=${#GPU_IDX[@]}
GPU_PID=()
timestamp=$(date "+%Y-%m-%d_%H%M%S")
# kill all the background jobs if terminated:
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

tmux kill-session
sleep 2

tmux new-session -d
tmux set -g mouse on

tmux split-window -h
tmux select-pane -L
tmux split-window -v
tmux select-pane -R
tmux split-window -v

tmux_log() {
    tmux pipe-pane -o "exec cat >> $HOME/tmux_logs/'tmux-%Y%m%d_%H%M%S-#P.log'"
}

idx=0

# for (( id=COUNT; id<=ENDCOUNT; id++ )); do
for dataset in "${DATASETS[@]}"; do
    if "$single" && [ -n "${GPU_PID[$idx]+x}" ]; then
        echo "Waiting for GPU ${GPU_IDX[$idx]}"
        wait "${GPU_PID[$idx]}"
        echo "GPU ${GPU_IDX[$idx]} is available"
    fi
    tmux select-pane -t "${GPU_IDX[$idx]}"
    tmux_log

    tmux send "export CUDA_VISIBLE_DEVICES=${GPU_IDX[$idx]}" C-m
    tmux send "conda activate sdfstudio" C-m
    tmux send "eval ./scripts/benchmarking/inside_tmux_train.sh '${method_name}' '${vis}' '${dataset}'" C-m
    tmux send "echo 'Launched ${method_name} ${dataset} on gpu ${GPU_IDX[$idx]}, ${tag}'" C-m

    # ((count=count+1))
    # echo "$count"
    # if [[$count=3]]; then
    #     tmux attach-session -d
    # fi
    
    # update gpu
    ((idx=(idx+1)%len))
    sleep 3
done
wait
echo "Done."
echo "Launch eval with:"
s=""
$single && s="-s"
echo "$(dirname "$0")/launch_eval_blender.sh -m $method_name -o outputs/ -t $timestamp $s"
