#!/usr/bin/env bash

raw_temp_dir=$1
raw_temp_socre_dir=$2
raw_temp_socre_dir_train=$3
sort_temp_dir=$4
sort_temp_score_dir=$5
top_rel=10

refine_temp() {
    mkdir -p ${3}
    mkdir -p ${4}
    for file in ${1}/*; do
        bfile=$(basename "${file}")
        echo ${bfile}
        python scripts/run_experiments.py \
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
            --rel_file ${1}/${bfile} \
            --prefix ${2} \
            --suffix .jsonl \
            --top ${4} \
            --ensemble \
            --batch_size 32 > ${3}${4}/${bfile}.out 2>&1
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
for file in data/TREx/*; do
    bfile=$(basename "${file}")
    infile=${raw_temp_socre_dir_train}/${bfile}.out
    outfile=${sort_temp_dir}/${bfile}
    echo ${bfile}
    if [ -f "${infile}" ]; then
        python scripts/ana.py --task sort --inp ${infile} --out ${outfile}
    fi
done

# evaluate using the top k templates
for top in 1 2 3 4 5 10000
do
    get_temp_ensemble_score ${sort_temp_dir} data/TREx/ ${sort_temp_score_dir} ${top} &
done
wait

echo done
