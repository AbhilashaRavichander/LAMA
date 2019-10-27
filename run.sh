: '
CUDA_VISIBLE_DEVICES=0 python scripts/rc.py --lm gpt --beam_size=32 > output/squad_gpt_beam32.out 2>&1
CUDA_VISIBLE_DEVICES=0 python scripts/rc.py --lm gpt --beam_size=64 > output/squad_gpt_beam64.out 2>&1
CUDA_VISIBLE_DEVICES=0 python scripts/rc.py --lm gpt --beam_size=128 > output/squad_gpt_beam128.out 2>&1
'
: '
#set -e
for filename in data/TREx_wikipedia_template_refine_nosubtoken_sort/*; do
    bfilename=$(basename "$filename")
    echo ${bfilename}
    python scripts/run_experiments.py \
        --rel_file data/TREx_wikipedia_template_refine_nosubtoken_sort/${bfilename} \
        --prefix data/TREx/ \
        --suffix .jsonl > output/relational_phrase_exp/trex_refine_nosubtoken_run_avg/${bfilename}.out 2>&1
done
'

set -e
for filename in data/TREx/*; do
    bfilename=$(basename "$filename")
    outfilename="output/relational_phrase_exp/trex_refine_nosubtoken_run_avg/$bfilename.out"
    objfilename="output/relational_phrase_exp/trex_objs/$bfilename.out"
    echo $bfilename
    if [ -f "$outfilename" ]; then
        python scripts/ana.py --task out --inp $outfilename --obj_file $objfilename
    fi
done

: '
for filename in data/TREx_wikipedia_template/*; do
    bfilename=$(basename "$filename")
    echo ${bfilename}
    CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py \
        --rel_file data/TREx_wikipedia_template/${bfilename} \
        --prefix data/TREx/ \
        --refine_template data/TREx_wikipedia_template_refine_nosubtoken/${bfilename} \
        --suffix .jsonl > output/relational_phrase_exp/trex_refine_nosubtoken/${bfilename}.out 2>&1
done
'
: '
for bfilename in P47.jsonl P495.jsonl P527.jsonl P530.jsonl P740.jsonl
do
    echo ${bfilename}
    CUDA_VISIBLE_DEVICES=0 python scripts/run_experiments.py \
        --rel_file data/TREx_wikipedia_template_refine_nosubtoken/${bfilename} \
        --prefix data/TREx/ \
        --suffix .jsonl > output/relational_phrase_exp/trex_refine_nosubtoken_run/${bfilename}.out 2>&1
done
'
: '
set -e
for filename in data/TREx/*; do
    bfilename=$(basename "$filename")
    infilename="output/relational_phrase_exp/trex/$bfilename.out"
    outfilename="data/TREx_wikipedia_template_sort/$bfilename"
    echo $bfilename
    if [ -f "$infilename" ]; then
        python scripts/ana.py --task sort --inp $infilename --out $outfilename
    fi
done
'
: '
set -e
for filename in data/TREx/*; do
    bfilename=$(basename "$filename")
    echo ${bfilename}
    python scripts/run_experiments.py \
        --rel_file data/TREx_wikipedia_template/${bfilename} \
        --prefix data/TREx/ \
        --get_objs \
        --suffix .jsonl > output/relational_phrase_exp/trex_objs/${bfilename}.out 2>&1
done
'
