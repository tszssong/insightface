#!/usr/bin/env bash

DEVKIT="../../../megaface-devkit/devkit/experiments"
ALGO="r100ii" #ms1mv2
# ROOT=$(dirname `which $0`)
ROOT=/cloud_data01/zhengmeisong/hpc36/myInsightface/Evaluation/Megaface/
echo $ROOT
python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model '../../../mx-preTrain/model-r50-am-lfw/model,0'
python -u remove_noises.py --algo "$ALGO"

cd "$DEVKIT"
# LD_LIBRARY_PATH="/usr/local/lib64:$LD_LIBRARY_PATH" 
python -u run_experiment.py "$ROOT/fea_out_clean/megaface" "$ROOT/fea_out_clean/facescrub" _"$ALGO".bin ../../mx_results/ -s 100 -p ../templatelists/facescrub_features_list.json
cd -
