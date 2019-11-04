from typing import List, Dict, Union, Any
import argparse
import numpy as np
import json
import os
from collections import defaultdict
import scipy.stats
from random import shuffle
from tqdm import tqdm
import urllib.request
import urllib.parse
import time


def avg_by_label(scores: List, labels: Union[List, None]):
    if labels is None:
        return np.mean(scores)
    label2score: Dict[Any, float] = defaultdict(lambda: 0)
    label2count: Dict[Any, int] = defaultdict(lambda: 0)
    assert len(scores) == len(labels), 'scores length not equal to labels length'
    for s, l in zip(scores, labels):
        label2score[l] += s
        label2count[l] += 1
    return np.mean([label2score[k] / label2count[k] for k in label2score])


def out_ana(args):
    stat = []
    templates = []
    objs = None
    obj_entropy = None
    if args.obj_file:
        with open(args.obj_file, 'r') as fin:
            for l in fin:
                if l.startswith('obj_labels'):
                    objs = l.strip().split(' ', 1)[1].split('\t')
        uni, counts = np.unique(objs, return_counts=True)
        counts = counts / np.sum(counts)
        obj_entropy = scipy.stats.entropy(counts)
    with open(args.inp, 'r') as fin:
        for l in fin:
            if l.startswith('P1all '):
                stat.append(list(map(float, l.strip().split(' ')[1].split('\t'))))
            elif l.startswith("{'dataset_filename':"):
                templates.append(eval(l.strip())['template'])
    stat = np.array(stat)
    first = avg_by_label(stat[0], objs)  # the first template is manually designed
    ensemble_score = avg_by_label(np.max(stat, 0), objs)  # ensemble all the templates
    if len(stat) > 1:
        temp_scores = np.array([avg_by_label(s, objs) for s in stat[1:]])
        best = np.argmax(temp_scores)  # the best template (except for the manually designed one)
        best_temp = templates[best + 1]
        best_score = temp_scores[best]
    else:
        best_temp = None
        best_score = 0
    print('first template: {}'.format(templates[0]))
    print('best template: {}'.format(best_temp))
    print('first {:.3f}, best {:.3f}, allbest {:.3f}, ensemble {:.3f}, numtemp {}'.format(
        first, best_score, max(first, best_score), ensemble_score, len(templates)))
    print('obj entropy: {}'.format(obj_entropy))


def wikidata_to_trex(args):
    pattern_file, trex_file = args.inp.split(':')
    with open(pattern_file, 'r') as fin:
        pattern = json.load(fin)
    relations = []
    with open(trex_file, 'r') as fin:
        for l in fin:
            relations.append(json.loads(l))

    for rel in relations:
        root_pid = rel['relation']
        templates = []
        seen_sni = set()
        for pid in pattern:
            if pid == root_pid or pid.startswith(root_pid + '_'):
                for sni in pattern[pid]['snippet']:
                    if tuple(sni[0]) in seen_sni:
                        print('dup snippeit')
                        continue
                    seen_sni.add(tuple(sni[0]))
                    sni_text, sni_count = sni[0][0], sni[1]
                    temp = '[X] {} [Y] .'.format(sni_text) if sni[0][1] == 1 else '[Y] {} [X] .'.format(sni_text)
                    templates.append((temp, sni_count))
        new_relations = []
        new_relations.append(rel)
        new_relations.extend([{
            'relation': root_pid,
            'template': temp[0],
            'label': None,
            'description': None,
            'type': rel['type'],
            'wikipedia_count': temp[1]
        } for temp in sorted(templates, key=lambda x: -x[1])])
        with open(os.path.join(args.out, root_pid + '.jsonl'), 'w') as fout:
            for rel in new_relations:
                fout.write(json.dumps(rel) + '\n')


def rank_templates(args):
    relation_name = os.path.basename(args.inp).split('.', 1)[0]
    templates, scores = [], []
    with open(args.inp, 'r') as fin:
        for l in fin:
            if l.startswith("{'dataset_filename':"):
                temp = eval(l.strip())['template']
                if type(temp) is list:
                    if len(temp) != 1:
                        raise Exception('more than one temp')
                    temp = temp[0]
                templates.append(temp)
            elif l.startswith('P1all '):
                scores.append(np.mean(list(map(float, l.strip().split(' ')[1].split('\t')))))
    temp_set = set()
    templates_new, scores_new = [], []
    for temp, score in zip(templates, scores):
        if temp in temp_set:
            continue
        temp_set.add(temp)
        templates_new.append(temp)
        scores_new.append(score)
    sorted_temps = sorted(zip(templates_new, scores_new), key=lambda x: -x[1])
    with open(args.out, 'w') as fout:
        for temp, score in sorted_temps:
            rel = {'relation': relation_name, 'template': temp, 'score': score}
            fout.write(json.dumps(rel) + '\n')


def major_class(args):
    file2classes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    with open(args.inp, 'r') as fin:
        for l in fin:
            obj = json.loads(l.strip())['obj_label']
            file2classes[args.inp][obj] += 1
    objs = sorted(file2classes[args.inp].items(), key=lambda x: -x[1])
    total = np.sum([obj[1] for obj in objs])
    #print(args.inp, objs, total)
    print('micro {:.3f}, macro {:.3f}'.format(objs[0][1] / total, 1 / len(objs)))


def get_train_data(args, top=1000):
    # wiki_domain/data_new/property_occurrence_prop435k/
    # wiki_domain/data/hiro_wikidata/eid2name.tsv
    occ_dir, name_file = args.inp.split(':')

    eid2name = {}
    with open(name_file, 'r') as fin:
        for l in tqdm(fin):
            l = l.strip().split('\t')
            eid2name[l[0]] = l[1]

    pids = []
    pid2hts: Dict[str, set] = defaultdict(set)
    for root, dirs, files in os.walk('data/TREx/'):
        for file in files:
            pid = file.split('.', 1)[0]
            pids.append(pid)
            with open(os.path.join(root, file), 'r') as fin:
                for l in fin:
                    l = json.loads(l)
                    pid2hts[pid].add((l['sub_uri'], l['obj_uri']))
    print(len(pids), pids)

    for pid in tqdm(pids):
        occ_file = os.path.join(occ_dir, pid + '.txt')
        if not os.path.exists(occ_file):
            raise Exception('{} not exist'.format(occ_file))
        hts = set()
        with open(occ_file, 'r') as fin:
            for l in fin:
                h, t = l.strip().split()
                if (h, t) in pid2hts[pid]:
                    continue
                if h not in eid2name or t not in eid2name:
                    continue
                if len(eid2name[h].split()) > 1 or len(eid2name[t].split()) > 1:
                    continue
                hts.add((h, t))
                if len(hts) >= 10 * top:
                    break
        if len(hts) <= top:
            print('{} less than {}'.format(pid, top))
        hts = list(hts)
        shuffle(hts)
        hts = hts[:top]

        with open(os.path.join(args.out, pid + '.jsonl'), 'w') as fout:
            for h, t in hts:
                rel = {'sub_uri': h, 'obj_uri': t, 'sub_label': eid2name[h], 'obj_label': eid2name[t]}
                fout.write(json.dumps(rel) + '\n')


def parse_ppdb_result(response: str):
    response = json.loads(response)['hits']
    num = response['found']
    hits = [{
        'target': h['target'],
        'score': h['default_formula_expr'],
        'id': h['id']
    } for h in response['hit']]
    return num, hits


def query_ppdb(source: str):
    base_url = 'http://paraphrase.org/api/en/search/?'
    param = {
        'batchNumber': 0,
        'needsPOSList': False,
        'q': source
    }
    count = 0
    results = []
    found_parahrase = set()
    while True:
        param['batchNumber'] = count
        url = base_url + '&' + urllib.parse.urlencode(param)
        #print(url)
        _, hits = parse_ppdb_result(urllib.request.urlopen(url).read().decode('utf-8'))
        if len(hits) == 0:
            break
        count += 1
        for hit in hits:
            if hit['target'] in found_parahrase:
                continue
            found_parahrase.add(hit['target'])
            results.append(hit)
        time.sleep(1)
    print('found {} pharaphrase for "{}"'.format(len(results), source))
    return results


def get_ppdb(args):
    relations = []
    with open(args.inp, 'r') as fin:
        for l in fin:
            relations.append(json.loads(l))
    for rel in relations:
        extend_rels = [rel]
        pid = rel['relation']
        temp = rel['template']
        if temp.startswith('[X]') or temp.startswith('[Y]'):
            results = query_ppdb(temp[3:-5].strip())
            for r in results:
                extend_rels.append({
                    'relation': pid,
                    'template': temp[:4] + r['target'] + temp[-6:],
                    'type': rel['type'],
                    'ppdb_id': r['id'],
                    'ppdb_score': r['score']
                })
        with open(os.path.join(args.out, pid + '.jsonl'), 'w') as fout:
            for r in extend_rels:
                fout.write(json.dumps(r) + '\n')
        time.sleep(5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze output log')
    parser.add_argument('--task', type=str, help='task', required=True, 
        choices=['out', 'wikidata', 'sort', 'major_class', 'get_train_data', 'get_ppdb'])
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--obj_file', type=str, help='obj file', default=None)
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()

    if args.task == 'out':
        out_ana(args)
    elif args.task == 'wikidata':
        wikidata_to_trex(args)
    elif args.task == 'sort':
        rank_templates(args)
    elif args.task == 'major_class':
        major_class(args)
    elif args.task == 'get_train_data':
        get_train_data(args)
    elif args.task == 'get_ppdb':
        get_ppdb(args)

