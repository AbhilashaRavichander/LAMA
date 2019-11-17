from typing import List, Dict, Union, Any, Tuple
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
import torch
from batch_eval_KB_completion import load_file
from subprocess import Popen, PIPE


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
                if l.startswith('sub_obj_label'):
                    objs = l.strip().split(' ', 1)[1].split('\t')
                    objs = objs[1:len(objs):2]
        uni, counts = np.unique(objs, return_counts=True)
        counts = counts / np.sum(counts)
        obj_entropy = scipy.stats.entropy(counts)
    with open(args.inp, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l.startswith('P1all '):
                stat.append(list(map(float, l.strip().split(' ')[1].split('\t'))))
            elif l.startswith('{') and l.endswith('}'):
                templates.append(eval(l)['template'])
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


def out_ana_optimize(args):
    relations = []
    scores = []
    relation = None
    with open(args.inp, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l.startswith('{') and l.endswith('}'):
                relation = eval(l)['relation']
            #if l.strip().startswith("'relation':"):
            #    relation = l.split(':', 1)[1].strip().rstrip(',').strip("'")
                if args.obj_file and os.path.exists(os.path.join(args.obj_file, relation + '.jsonl.out')):
                    with open(os.path.join(args.obj_file, relation + '.jsonl.out'), 'r') as fin:
                        for l in fin:
                            if l.startswith('sub_obj_label'):
                                objs = l.strip().split(' ', 1)[1].split('\t')
                                objs = objs[1:len(objs):2]
                                break
            #elif l.startswith('global Precision at 1:'):
            #    score = float(l.split(':', 1)[1].strip())
            #    relations.append(relation)
            #    scores.append(score)
            elif l.startswith('P1all '):
                stat = list(map(float, l.strip().split(' ')[1].split('\t')))
                if args.obj_file:
                    score = avg_by_label(stat, objs)
                else:
                    score = np.mean(stat)
                relations.append(relation)
                scores.append(score)
    rel_scores = dict(zip(relations, scores))
    rel_scores = [(rel.split('.', 1)[0], rel_scores[rel.split('.', 1)[0]])
                  for rel in listdir_shell('data/TREx') if rel.split('.', 1)[0] in rel_scores]
    print('\n'.join(map(lambda x: '{}\t{}'.format(*x), rel_scores)))
    print('mean: {}'.format(np.mean(scores)))


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
            l = l.strip()
            if l.startswith('{') and l.endswith('}'):
                temp = eval(l)['template']
                if type(temp) is list:
                    if len(temp) != 1:
                        raise Exception('more than one temp')
                    temp = temp[0]
                templates.append(temp)
            elif l.startswith('P1all '):
                scores.append(np.mean(list(map(float, l.strip().split(' ')[1].split('\t')))))
    temp_set = set()
    templates_new, scores_new = [], []
    if args.exclude_first:
        templates = templates[1:]
        scores = scores[1:]
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


def case_ana(args):
    topk = 2
    if args.obj_file:
        with open(args.obj_file, 'r') as fin:
            for l in fin:
                if l.startswith('sub_obj_label'):
                    objs = l.strip().split(' ', 1)[1].split('\t')
                    sub_obj = np.array([(objs[i], objs[i + 1]) for i in range(0, len(objs), 2)])
                    break
    stat = []
    templates = []
    with open(args.inp, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l.startswith('P1all '):
                stat.append(list(map(float, l.strip().split(' ')[1].split('\t'))))
            elif l.startswith('{') and l.endswith('}'):
                templates.append(eval(l)['template'])
    stat = np.array(stat)
    templates = np.array(templates)
    
    temp_rank = np.argsort(-stat.mean(-1))
    solved_by_topk = stat[temp_rank[:topk]].max(0)
    filtered_stat = stat * (1 - solved_by_topk).reshape(1, -1)
    filtered_temp_rank = np.argsort(-filtered_stat.mean(-1))

    print('top temp: {}'.format(templates[temp_rank[:topk]]))
    for i, rank in enumerate(filtered_temp_rank):
        print('#{} {}'.format(i + 1, templates[rank]))
        print('head-tails {}'.format(sub_obj[filtered_stat[rank] == 1]))
        input()


def merge_all_rel(args, top=None):
    with open(args.out, 'w') as fout:
        for root, dirs, files in os.walk(args.inp):
            for file in files:
                rel_name = file.rsplit('.', 1)[0]
                rel_li = []
                with open(os.path.join(root, file), 'r') as fin:
                    for l in fin:
                        rel_li.append(json.loads(l))
                    temps = [rel['template'] for rel in rel_li]
                    if top:
                        temps = temps[:top]
                    rel_li[0]['template'] = temps
                fout.write(json.dumps(rel_li[0]) + '\n')


def split_dev(args):
    for root, dirs, files in os.walk(args.inp):
        for file in files:
            ls = []
            with open(os.path.join(root, file), 'r') as fin:
                for l in fin:
                    ls.append(l.strip())
            shuffle(ls)
            split = int(len(ls) * 0.8)
            if split == 0 or split >= len(ls):
                raise Exception('empty split for {}'.format(file))
            os.makedirs(root + '_train', exist_ok=True)
            os.makedirs(root + '_train_dev', exist_ok=True)
            with open(os.path.join(root + '_train', file), 'w') as fout:
                for l in ls[:split]:
                    fout.write(l + '\n')
            with open(os.path.join(root + '_train_dev', file), 'w') as fout:
                for l in ls[split:]:
                    fout.write(l + '\n')


def listdir_shell(path, *lsargs):
    p = Popen(('ls', path) + lsargs, shell=False, stdout=PIPE, close_fds=True)
    return [path.decode('utf-8').rstrip('\n') for path in p.stdout.readlines()]


def weight_ana(args, top=1):
    weight_file, relation_file = args.inp.split(':')
    relations = load_file(relation_file)
    rel2temp = dict((rel['relation'], rel['template']) for rel in relations)
    weights = torch.load(weight_file)
    peak_count, all_count = 0, 0
    for rel in listdir_shell('data/TREx'):
        rel = rel.split('.', 1)[0]
        weight_dim = weights[rel].size(0)
        weight = weights[rel].exp() / weights[rel].exp().sum()
        max_ind = weight.argmax().item()
        max_weight = weight[max_ind].item()
        more_prob = 0
        if weight_dim > len(rel2temp[rel]):
            more_prob = weight[len(rel2temp[rel]):].sum()
        all_count += 1
        if max_weight >= 0.9:
            peak_count += 1
        print('relation {} max ind {} max prob {} more prob {} template {}'.format(
            rel, max_ind, max_weight, more_prob, rel2temp[rel][max_ind % len(rel2temp[rel])]))
        if top > 1:
            rank = torch.argsort(-weight)[:top]
            print([(rel2temp[rel][i % len(rel2temp[rel])], '{:.3f}'.format(weight[i].item())) for i in rank])
    print('peak {} out of {}, weight dim {}'.format(peak_count, all_count, weight_dim))


def parse_fairseq_file(filename):
    results: List[Tuple[str, float]] = []
    with open(filename, 'r') as fin:
        for l in fin:
            if l.startswith('H-'):
                _, score, sent = l.strip().split('\t')
                results.append((sent, float(score)))
    return results


def bt_filter(args):
    pid_file, raw_tempfile = args.temp_file.split(':')
    relations = list(map(lambda x: x.strip(), open(pid_file, 'r').read().strip().split('\n')))
    raw_temps = list(map(lambda x: x.strip(), open(raw_tempfile, 'r').read().strip().split('\n')))
    num_rel = len(relations)
    rel2temps = defaultdict(lambda: [])
    forward_file, backward_file = args.inp.split(':')
    forward_result = parse_fairseq_file(forward_file)
    backward_result = parse_fairseq_file(backward_file)
    assert len(forward_result) == args.beam * num_rel
    assert len(backward_result) == args.beam * args.beam * num_rel
    fs = np.array([r[1] for r in forward_result]).reshape((num_rel, args.beam, 1))
    bs = np.array([r[1] for r in backward_result]).reshape((num_rel, args.beam, args.beam))
    final_scores = (bs + fs).reshape((num_rel, -1))  # add two avg log prob
    new_temps = np.array([r[0] for r in backward_result]).reshape((num_rel, args.beam * args.beam))
    for i in range(num_rel):
        rel = relations[i]
        rank = np.argsort(-final_scores[i])
        seen = set()
        for r in rank:
            temp = new_temps[i][r]
            score = final_scores[i][r]
            if temp.find('[X]') == -1 or temp.find('[Y]') == -1:
                continue
            if temp not in seen:
                rel2temps[rel].append((temp, score))
                seen.add(temp)
    for i in range(num_rel):
        rel = relations[i]
        raw_temp = raw_temps[i]
        temps = rel2temps[rel]
        with open(os.path.join(args.out, rel + '.jsonl'), 'w') as fout:
            fout.write(json.dumps({'relation': rel, 'template': raw_temp}) + '\n')
            for temp in temps:
                fout.write(json.dumps({'relation': rel, 'template': temp[0], 'bt_log_prob': temp[1]}) + '\n')
            print('{} has {} temps'.format(rel, len(temps)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze output log')
    parser.add_argument('--task', type=str, help='task', required=True, 
        choices=['out', 'wikidata', 'sort', 'major_class', 'get_train_data',
                 'get_ppdb', 'case', 'merge_all_rel', 'split_dev', 'weight_ana',
                 'out_ana_opti', 'bt_filter'])
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--obj_file', type=str, help='obj file', default=None)
    parser.add_argument('--out', type=str, help='output file')
    parser.add_argument('--temp_file', type=str, help='pid and raw temp file', default=None)
    parser.add_argument('--beam', type=int, help='beam size', default=None)
    parser.add_argument('--exclude_first', help='whether to exclude the first template (manual)', action='store_true')
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
    elif args.task == 'case':
        case_ana(args)
    elif args.task == 'merge_all_rel':
        merge_all_rel(args, top=30)
    elif args.task == 'split_dev':
        split_dev(args)
    elif args.task == 'weight_ana':
        weight_ana(args, top=3)
    elif args.task == 'out_ana_opti':
        out_ana_optimize(args)
    elif args.task == 'bt_filter':
        bt_filter(args)
