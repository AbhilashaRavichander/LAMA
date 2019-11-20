#!/usr/bin/env bash

lm=bert_large
merge_rel_file=$1
feature_root_dir=$2
cuda1=$3
cuda2=$4

#set -e

precompute() {
    head_tail_dir=$1
    feature_dir=$2
    mkdir -p ${feature_dir}
    python scripts/run_experiments.py \
        --lm_model ${lm} \
        --rel_file ${merge_rel_file} \
        --prefix ${head_tail_dir} \
        --suffix .jsonl \
        --temp_model mixture_precompute \
        --batch_size 32 \
        --save ${feature_dir} \
        --num_feat 2 \
        --bt_obj 1
}

optimize() {
    head_tail_dir=$1
    feature_dir=$2
    weight_file=$3
    num_feat=$4
    more=""
    if [ $# -gt 4 ]; then
        more=--enforce_prob
    fi
    mkdir -p $(dirname "$weight_file")
    python scripts/run_experiments.py \
        --lm_model ${lm} \
        --rel_file ${merge_rel_file} \
        --prefix ${head_tail_dir} \
        --suffix .jsonl \
        --temp_model mixture_optimize \
        --batch_size 32 \
        --feature_dir ${feature_dir} \
        --save ${weight_file} \
        --num_feat ${num_feat} ${more}
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

# extract features
CUDA_VISIBLE_DEVICES=$cuda1 precompute data/TREx_train_train ${feature_root_dir}/feature_train &
CUDA_VISIBLE_DEVICES=$cuda1 precompute data/TREx_train_train_dev ${feature_root_dir}/feature_train_dev &
CUDA_VISIBLE_DEVICES=$cuda2 precompute data/TREx ${feature_root_dir}/feature_test &
wait

# optimize
for feat_type in feature_train feature_test
do
    head_tail_dir=""
    if [ $feat_type == feature_train ]; then
        head_tail_dir=data/TREx_train_train
    elif [ $feat_type == feature_test ]; then
        head_tail_dir=data/TREx
    fi
    #(CUDA_VISIBLE_DEVICES=$cuda1 optimize ${head_tail_dir} ${feature_root_dir}/${feat_type} ${feature_root_dir}/${feat_type}/weight/feat1_log.pt 1 ; CUDA_VISIBLE_DEVICES=$cuda1 predict ${head_tail_dir} ${feature_root_dir}/${feat_type}/weight/feat1_log.pt 1) &
    (CUDA_VISIBLE_DEVICES=$cuda1 optimize ${head_tail_dir} ${feature_root_dir}/${feat_type} ${feature_root_dir}/${feat_type}/weight/feat1_prob.pt 1 prob ; CUDA_VISIBLE_DEVICES=$cuda1 predict data/TREx ${feature_root_dir}/${feat_type}/weight/feat1_prob.pt 1 prob) &
    #(CUDA_VISIBLE_DEVICES=$cuda2 optimize ${head_tail_dir} ${feature_root_dir}/${feat_type} ${feature_root_dir}/${feat_type}/weight/feat2_log.pt 2 ; CUDA_VISIBLE_DEVICES=$cuda2 predict ${head_tail_dir} ${feature_root_dir}/${feat_type}/weight/feat2_log.pt 2) &
    (CUDA_VISIBLE_DEVICES=$cuda2 optimize ${head_tail_dir} ${feature_root_dir}/${feat_type} ${feature_root_dir}/${feat_type}/weight/feat2_prob.pt 2 prob ; CUDA_VISIBLE_DEVICES=$cuda2 predict data/TREx ${feature_root_dir}/${feat_type}/weight/feat2_prob.pt 2 prob) &
    wait
done
