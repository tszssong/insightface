#!/usr/bin/env bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

DEVKIT="//home/ubuntu/zms/wkspace/FR/devkit/experiments"
ALGO="r100_10_07" #ms1mv2
ROOT=$(dirname `which $0`)
ROOT=/home/ubuntu/zms/wkspace/FR/myInsightface/Evaluation/Megaface/
echo $ROOT
#python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model './mxnet/model-r100-ii/model,0'
python -u gen_megaface.py --gpu 0 --algo "$ALGO" --model './mxnet/model_r100_10_07/model,15'
python -u remove_noises.py --algo "$ALGO"

cd "$DEVKIT"
LD_LIBRARY_PATH="/usr/local/lib64:$LD_LIBRARY_PATH" python -u run_experiment.py "$ROOT/feature_out_clean/megaface" "$ROOT/feature_out_clean/facescrub" _"$ALGO".bin ../../mx_results/ -s 1000000 -p ../templatelists/facescrub_features_list.json
cd -

