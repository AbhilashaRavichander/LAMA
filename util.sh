#!/usr/bin/env bash

eval() {
    # | grep -P '^first [0-9].+' | awk -F'[ ,]' '{print $2}'
    # | awk '{sum += $1} END {print "mean = " sum/NR}'
    set -e
    for file in data/TREx/*; do
        bfile=$(basename "${file}")
        outfile=${1}/${bfile}.out
        objfile=output/exp/trex_objs/${bfile}.out
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

if [[ $1 == 'eval' ]]; then
    eval $2 $3
elif [[ $1 == 'other' ]]; then
    echo other
fi
