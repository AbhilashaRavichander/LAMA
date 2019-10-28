: '
CUDA_VISIBLE_DEVICES=0 python scripts/rc.py --lm gpt --beam_size=32 > output/squad_gpt_beam32.out 2>&1
CUDA_VISIBLE_DEVICES=0 python scripts/rc.py --lm gpt --beam_size=64 > output/squad_gpt_beam64.out 2>&1
CUDA_VISIBLE_DEVICES=0 python scripts/rc.py --lm gpt --beam_size=128 > output/squad_gpt_beam128.out 2>&1
'
: '
#set -e
for filename in data/TREx_wikipedia_template_refine_nosubtoken_train_sort/*; do
    bfilename=$(basename "$filename")
    echo ${bfilename}
    python scripts/run_experiments.py \
        --rel_file data/TREx_wikipedia_template_refine_nosubtoken_train_sort/${bfilename} \
        --prefix data/TREx/ \
        --ensemble \
        --top 1 \
        --batch_size 32 \
        --suffix .jsonl > output/relational_phrase_exp/trex_refine_nosubtoken_train_run_sort1/${bfilename}.out 2>&1
done
'

# | grep -P '^first [0-9].+' | awk -F'[ ,]' '{print $2}'
set -e
for filename in data/TREx/*; do
    bfilename=$(basename "$filename")
    outfilename="output/relational_phrase_exp/trex_refine_nosubtoken_train_run_sort1/$bfilename.out"
    objfilename="output/relational_phrase_exp/trex_objs/$bfilename.out"
    echo $bfilename
    if [ -f "$outfilename" ]; then
        python scripts/ana.py --task out --inp $outfilename --obj_file $objfilename
    fi
done

: '
#set -e
for filename in data/TREx_wikipedia_template/*; do
    bfilename=$(basename "$filename")
    echo ${bfilename}
    python scripts/run_experiments.py \
        --rel_file data/TREx_wikipedia_template/${bfilename} \
        --prefix data/TREx_train/ \
        --refine_template data/TREx_wikipedia_template_refine_nosubtoken_train/${bfilename} \
        --suffix .jsonl > output/relational_phrase_exp/trex_refine_nosubtoken_train/${bfilename}.out 2>&1
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
    infilename="output/relational_phrase_exp/trex_train/$bfilename.out"
    outfilename="data/TREx_wikipedia_template_train_sort/$bfilename"
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



: '
#set -e
for filename in data/TREx_wikipedia_template_refine_nosubtoken_train/*; do
    bfilename=$(basename "$filename")
    echo ${bfilename}
    python scripts/run_experiments.py \
        --rel_file data/TREx_wikipedia_template_refine_nosubtoken_train/${bfilename} \
        --prefix data/TREx/ \
        --batch_size 32 \
        --suffix .jsonl > output/relational_phrase_exp/trex_refine_nosubtoken_train_run/${bfilename}.out 2>&1
done


for filename in data/TREx/*; do
    bfilename=$(basename "$filename")
    infilename="output/relational_phrase_exp/trex_refine_nosubtoken_train_run/$bfilename.out"
    outfilename="data/TREx_wikipedia_template_refine_nosubtoken_train_sort/$bfilename"
    echo $bfilename
    if [ -f "$infilename" ]; then
        python scripts/ana.py --task sort --inp $infilename --out $outfilename
    fi
done

for top in 2 3 4 5 100
do
    mkdir -p output/relational_phrase_exp/trex_refine_nosubtoken_train_run_sort${top}
    for filename in data/TREx_wikipedia_template_refine_nosubtoken_train_sort/*; do
        bfilename=$(basename "$filename")
        echo ${bfilename}
        python scripts/run_experiments.py \
            --rel_file data/TREx_wikipedia_template_refine_nosubtoken_train_sort/${bfilename} \
            --prefix data/TREx/ \
            --top ${top} \
            --ensemble \
            --batch_size 32 \
            --suffix .jsonl > output/relational_phrase_exp/trex_refine_nosubtoken_train_run_sort${top}/${bfilename}.out 2>&1
    done
done
'
: '
set -e
for filename in data/TREx/*; do
    echo $filename
    python scripts/ana.py --task major_class --inp $filename
done
'
