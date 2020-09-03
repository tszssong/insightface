#!/usr/bin/env bash

DEVKIT="//home/ubuntu/zms/wkspace/FR/devkit/experiments"
ALGO="py1_r50" #ms1mv2
ROOT=$(dirname `which $0`)
ROOT=/home/ubuntu/zms/wkspace/FR/myInsightface/Evaluation/Megaface/
echo $ROOT
#python -u py_megaface.py --gpu 0 --algo "$ALGO" --model './IR_50/Backbone_IR_50_Epoch_24_Batch_855319_Time_2019-12-23-06-51_checkpoint.pth'
#python -u remove_noises.py --algo "$ALGO"

cd "$DEVKIT"
LD_LIBRARY_PATH="/usr/local/lib64:$LD_LIBRARY_PATH" python -u run_experiment.py "$ROOT/feature_out_clean/megaface" "$ROOT/feature_out_clean/facescrub" _"$ALGO".bin ../../mx_results/ -s 1000000 -p ../templatelists/facescrub_features_list.json
cd -

