#!/usr/bin/env bash

raw_temp_file=$1
model_dir=$2
beam=$3
final_temp_dir=$4

forward_model_dir=${model_dir}/wmt19.en-de.joined-dict.ensemble
backward_model_dir=${model_dir}/wmt19.de-en.joined-dict.ensemble

forward_temp_file=${raw_temp_file}.de_${beam}
backward_temp_file=${raw_temp_file}.de_${beam}.en_${beam}

# forward translation
fairseq-interactive \
    --path ${forward_model_dir}/model4.pt ${forward_model_dir} \
    --bpe-codes ${forward_model_dir}/bpecodes \
    --source-lang en --target-lang de --tokenizer moses --bpe fastbpe \
    --beam ${beam} --nbest ${beam} < ${raw_temp_file} &> ${forward_temp_file}

# backward translation
grep -P '^H' ${forward_temp_file} | awk -F'[\t]' '{print $3}' | fairseq-interactive \
    --path ${backward_model_dir}/model4.pt ${backward_model_dir} \
    --bpe-codes ${backward_model_dir}/bpecodes \
    --source-lang de --target-lang en --tokenizer moses --bpe fastbpe \
    --beam ${beam} --nbest ${beam} &> ${backward_temp_file}

# collect templates
python scripts/ana.py \
    --task bt_filter \
    --temp_file data/TREx_mt/rel.txt:${raw_temp_file} \
    --inp ${forward_temp_file}:${backward_temp_file} \
    --out ${final_temp_dir} \
    --beam ${beam}
