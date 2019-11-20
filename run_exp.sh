#!/usr/bin/env bash

lm=bert_large
raw_temp_dir=$1
raw_temp_socre_dir=$2
raw_temp_socre_dir_train=$3
sort_temp_dir=$4
sort_temp_dir_nom=${sort_temp_dir}_nomanual
sort_temp_score_dir=$5
sort_temp_score_dir_nom=${sort_temp_score_dir}_nomanual
top_rel=30

#set -e

refine_temp() {
    mkdir -p ${3}
    mkdir -p ${4}
    for file in ${1}/*; do
        bfile=$(basename "${file}")
        echo ${bfile}
        python scripts/run_experiments.py \
            --lm_model ${lm} \
            --rel_file ${1}/${bfile} \
            --prefix ${2} \
            --refine_template ${3}/${bfile} \
            --suffix .jsonl > ${4}/${bfile}.out 2>&1
    done
}

get_temp_score() {
    mkdir -p ${3}
    for file in ${1}/*; do
        bfile=$(basename "${file}")
        echo ${bfile}
        python scripts/run_experiments.py \
            --lm_model ${lm} \
            --rel_file ${1}/${bfile} \
            --prefix ${2} \
            --suffix .jsonl \
            --top ${4} \
            --batch_size 32 > ${3}/${bfile}.out 2>&1
   done
}

get_temp_ensemble_score() {
    mkdir -p ${3}${4}
    for file in ${1}/*; do
        bfile=$(basename "${file}")
        echo ${bfile}
        python scripts/run_experiments.py \
            --lm_model ${lm} \
            --rel_file ${1}/${bfile} \
            --prefix ${2} \
            --suffix .jsonl \
            --top ${4} \
            --ensemble \
            --batch_size 32 > ${3}${4}/${bfile}.out 2>&1
    done
}

get_temp_ensemble_dynamic_score() {
    outdir=${3}${4}_$(echo "${5}" | sed -r 's/_//g')
    mkdir -p ${outdir}
    echo ${outdir}
    for file in ${1}/*; do
        bfile=$(basename "${file}")
        echo ${bfile}
        python scripts/run_experiments.py \
            --lm_model ${lm} \
            --rel_file ${1}/${bfile} \
            --prefix ${2} \
            --suffix .jsonl \
            --top ${4} \
            --ensemble \
            --dynamic ${5} \
            --batch_size 32 > ${outdir}/${bfile}.out 2>&1
    done
}

# refine templates
if [ $# -gt 5 ]; then
    refine_temp_dir=$6
    refine_temp_score_dir=$7
    refine_temp ${raw_temp_dir} data/TREx_train/ ${refine_temp_dir} ${refine_temp_score_dir}
    raw_temp_dir=${refine_temp_dir}
fi

# get template score on test set
get_temp_score ${raw_temp_dir} data/TREx/ ${raw_temp_socre_dir} ${top_rel} &
# get template score on training set
get_temp_score ${raw_temp_dir} data/TREx_train/ ${raw_temp_socre_dir_train} ${top_rel} &
wait

# sort all the templates
mkdir -p ${sort_temp_dir}
mkdir -p ${sort_temp_dir_nom}
for file in data/TREx/*; do
    bfile=$(basename "${file}")
    infile=${raw_temp_socre_dir_train}/${bfile}.out
    outfile=${sort_temp_dir}/${bfile}
    outfile_nom=${sort_temp_dir_nom}/${bfile}
    echo ${bfile}
    if [ -f "${infile}" ]; then
        python scripts/ana.py --task sort --inp ${infile} --out ${outfile}
        python scripts/ana.py --task sort --inp ${infile} --out ${outfile_nom} --exclude_first
    fi
done

# evaluate using the top k templates
for top in 1 2 3 4 5 6 7 8 9 10 10000
do
    get_temp_ensemble_score ${sort_temp_dir} data/TREx/ ${sort_temp_score_dir} ${top} &
    get_temp_ensemble_score ${sort_temp_dir_nom} data/TREx/ ${sort_temp_score_dir_nom} ${top} &
    wait
done

: '
wait

for top in 4 5 6
do
    get_temp_ensemble_score ${sort_temp_dir} data/TREx/ ${sort_temp_score_dir} ${top} &
done
wait

for top in 7 8 9
do
    get_temp_ensemble_score ${sort_temp_dir} data/TREx/ ${sort_temp_score_dir} ${top} &
done
wait

for top in 10 10000
do
    get_temp_ensemble_score ${sort_temp_dir} data/TREx/ ${sort_temp_score_dir} ${top} &
done
wait
'

: '
# evaluate using dynamic top k templates
#set -e
sort_top=10
dyn_algo=bt_topk
beam=3
for dyn_top in 1 2 3
do
    CUDA_VISIBLE_DEVICES=5 get_temp_ensemble_dynamic_score ${sort_temp_dir} data/TREx/ ${sort_temp_score_dir} ${sort_top} ${dyn_algo}${dyn_top}-${beam} &
done

for dyn_top in 4 5 6
do
    CUDA_VISIBLE_DEVICES=4 get_temp_ensemble_dynamic_score ${sort_temp_dir} data/TREx/ ${sort_temp_score_dir} ${sort_top} ${dyn_algo}${dyn_top}-${beam} &
done

for dyn_top in 7 8 9
do
    CUDA_VISIBLE_DEVICES=3 get_temp_ensemble_dynamic_score ${sort_temp_dir} data/TREx/ ${sort_temp_score_dir} ${sort_top} ${dyn_algo}${dyn_top}-${beam} &
done

for dyn_top in 10
do
    CUDA_VISIBLE_DEVICES=2 get_temp_ensemble_dynamic_score ${sort_temp_dir} data/TREx/ ${sort_temp_score_dir} ${sort_top} ${dyn_algo}${dyn_top}-${beam} &
done
wait
'
echo done
