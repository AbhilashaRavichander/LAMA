#!/usr/bin/env bash

lm=$1
merge_rel_file=$2
feature_root_dir=$3
temp=$4
cuda1=$5
cuda2=$6

#set -e

optimize_on_the_fly() {  # only use for log features
    head_tail_dir=$1
    weight_file=$2
    num_feat=$3
    temperature=$4
    bt_obj=0
    if [ $num_feat == 2 ]; then
        bt_obj=5
    fi
    mkdir -p $(dirname "$weight_file")
    python scripts/run_experiments.py \
        --lm_model ${lm} \
        --rel_file ${merge_rel_file} \
        --prefix ${head_tail_dir} \
        --suffix .jsonl \
        --temp_model mixture_optimize \
        --batch_size 32 \
        --save ${weight_file} \
        --num_feat ${num_feat} \
        --bt_obj ${bt_obj} \
        --temperature ${temperature}
}

predict() {
    head_tail_dir=$1
    weight_file=$2
    num_feat=$3
    more=""
    if [ $# -gt 3 ]; then
        more=--enforce_prob
    fi
    bt_obj=0
    if [ $num_feat == 2 ]; then
        bt_obj=5
    fi
    python scripts/run_experiments.py \
        --lm_model ${lm} \
        --rel_file ${merge_rel_file} \
        --prefix ${head_tail_dir} \
        --suffix .jsonl \
        --temp_model mixture_predict \
        --batch_size 32 \
        --load ${weight_file} \
        --bt_obj ${bt_obj} \
        --num_feat ${num_feat} ${more} &> ${weight_file}.out
}

# optimize with softmax
for feat_type in feature_train feature_test
do
    head_tail_dir=""
    if [ $feat_type == feature_train ]; then
        head_tail_dir=data/TREx_train_train
    elif [ $feat_type == feature_test ]; then
        head_tail_dir=data/TREx
    fi
    (CUDA_VISIBLE_DEVICES=$cuda1 optimize_on_the_fly ${head_tail_dir} ${feature_root_dir}/${feat_type}/weight_${temp}/feat1_log_sm.pt 1 ${temp} ; CUDA_VISIBLE_DEVICES=$cuda1 predict data/TREx ${feature_root_dir}/${feat_type}/weight_${temp}/feat1_log_sm.pt 1) &
    (CUDA_VISIBLE_DEVICES=$cuda2 optimize_on_the_fly ${head_tail_dir} ${feature_root_dir}/${feat_type}/weight_${temp}/feat2_log_sm.pt 2 ${temp} ; CUDA_VISIBLE_DEVICES=$cuda2 predict data/TREx ${feature_root_dir}/${feat_type}/weight_${temp}/feat2_log_sm.pt 2) &
    wait
done
