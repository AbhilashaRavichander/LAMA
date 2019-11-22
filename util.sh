#!/usr/bin/env bash

eval() {
    # | grep -P '^first [0-9].+' | awk -F'[ ,]' '{print $2}'
    # | awk '{sum += $1} END {print "mean = " sum/NR}'
    set -e
    for file in data/TREx/*; do
        bfile=$(basename "${file}")
        outfile=${1}/${bfile}.out
        objfile=output/exp/trex_subobjs/${bfile}.out
        echo ${bfile}
        if [ -f "${outfile}" ]; then
            if [ $# -gt 1 ]; then
                python scripts/ana.py --task out --inp ${outfile} --obj_file ${objfile}
            else
                python scripts/ana.py --task out --inp ${outfile}
            fi
        fi
    done
}

eval_opti() {
    if [ $# -gt 1 ]; then
        python scripts/ana.py --task out_ana_opti --inp $1 --obj_file output/exp/trex_subobjs
    else
        python scripts/ana.py --task out_ana_opti --inp $1
    fi
}

eval_all() {
    set -e
    for file in data/TREx/*; do
        bfile=$(basename "${file}")
        outfile=${1}/${bfile}.out
        objfile=output/exp/trex_subobjs/${bfile}.out
        echo ${bfile}
        if [ -f "${outfile}" ]; then
            if [ $# -gt 1 ]; then
                python scripts/ana.py --task out_all_ana --inp ${outfile} --obj_file ${objfile}
            else
                python scripts/ana.py --task out_all_ana --inp ${outfile}
            fi
        fi
    done
}

ana() {
    set -e
    for file in data/TREx/*; do
        bfile=$(basename "${file}")
        outfile=${1}/${bfile}.out
        echo ${bfile}
        if [ -f "${outfile}" ]; then
            grep "correct-incorrect" ${outfile}
        fi
    done
}

get_ht() {
    set -e
    mkdir -p ${1}
    for file in data/TREx/*; do
        bfile=$(basename "${file}")
        echo ${bfile}
        python scripts/run_experiments.py \
            --rel_file data/TREx_mine/temp/${bfile} \
            --prefix data/TREx/ \
            --suffix .jsonl \
            --top 1 \
            --get_objs \
            --batch_size 32 > ${1}/${bfile}.out 2>&1
    done
}

if [[ $1 == 'eval' ]]; then
    eval $2 $3
elif [[ $1 == 'eval_all' ]]; then
    eval_all $2 $3
elif [[ $1 == 'eval_opti' ]]; then
    eval_opti $2 $3
elif [[ $1 == 'ana' ]]; then
    ana $2
elif [[ $1 == 'get_ht' ]]; then
    get_ht $2
elif [[ $1 == 'other' ]]; then
    echo other
fi
