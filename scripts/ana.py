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
import itertools
from subprocess import Popen, PIPE
from fuzzywuzzy import fuzz
from scipy.stats import pearsonr
import nltk
import matplotlib.pyplot as plt
import pandas as pd


def load_file(filename):
    data = []
    with open(filename, "r") as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data


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


def load_out_file(args):
    stat = []
    templates = []
    subjs = None
    objs = None
    if args.obj_file:
        with open(args.obj_file, 'r') as fin:
            for l in fin:
                if l.startswith('sub_obj_label'):
                    objs = l.strip().split(' ', 1)[1].split('\t')
                    subjs = objs[0:len(objs):2]
                    objs = objs[1:len(objs):2]
        subjs = np.array(subjs)
        objs = np.array(objs)
    first_p, first_temp = [args.exclude_first] * 2
    with open(args.inp, 'r') as fin:
        for l in fin:
            l = l.strip()
            if l.startswith('P1all '):
                if first_p:
                    first_p = False
                    continue
                stat.append(list(map(float, l.split(' ')[1].split('\t'))))
            elif l.startswith('{') and l.endswith('}'):
                if first_temp:
                    first_temp = False
                    continue
                templates.append(eval(l)['template'])
    return np.array(templates), np.array(stat), subjs, objs


def case_study(args):
    templates, stat, subjs, objs = load_out_file(args)
    best_ind = np.argmax(stat.mean(-1))
    print('manual {}'.format(templates[0]))
    print('best {} {}'.format(best_ind, templates[best_ind]))
    for ind in range(len(templates)):
        cases_ind = (stat[ind] - stat[0]) > 0
        cases = np.array(list(zip(subjs[cases_ind], objs[cases_ind])))
        if len(cases) > 0:
            print('{} {}'.format(ind, templates[ind]))
            print('#cases {}'.format(len(cases)))
            #show = np.random.permutation(len(cases))[:10]
            show = list(range(10))[:len(cases)]
            print(cases[show])
            input()
        else:
            print('{} {}'.format(ind, templates[ind]))
            print('#cases {}'.format(len(cases)))


def out_ana(args):
    templates, stat, subjs, objs = load_out_file(args)
    obj_entropy = None
    if objs is not None:
        uni, counts = np.unique(objs, return_counts=True)
        counts = counts / np.sum(counts)
        obj_entropy = scipy.stats.entropy(counts)
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


def out_all_ana(args):
    templates, stat, subjs, objs = load_out_file(args)
    temp_scores = np.array([avg_by_label(s, objs) for s in stat])
    print('\t'.join(map(str, temp_scores)))


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
                    if args.exclude_first:  # remove the manual template
                        temps = temps[1:]
                    if len(temps) == 0:
                        print('{} is empty'.format(rel_name))
                        continue
                    else:
                        print('{} {}'.format(rel_name, len(temps)))
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
    import torch
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


def sub_obj(args):
    subs_li, objs_li, obj_ent_li = [], [], []
    for root, dirs, files in os.walk(args.inp):
        for file in files:
            subs, objs = defaultdict(lambda: 0), defaultdict(lambda: 0)
            with open(os.path.join(root, file), 'r') as fin:
                for l in fin:
                    sub = json.loads(l)['sub_label']
                    obj = json.loads(l)['obj_label']
                    subs[sub] += 1
                    objs[obj] += 1
            counts = np.array(list(objs.values()))
            counts = counts / np.sum(counts)
            obj_entropy = scipy.stats.entropy(counts)
            subs_li.append(len(subs))
            objs_li.append(len(objs))
            obj_ent_li.append(obj_entropy)
            print(file, len(subs), len(objs), obj_entropy, sorted(objs.items(), key=lambda x: -x[1])[0])
    print(np.mean(subs_li), np.mean(objs_li), np.mean(obj_ent_li))

def calc_l1_distance(list1, list2):
    dist = 0
    for x, y in zip(list1, list2):
        dist += abs(x - y)
    return dist

def pairwise_distance(output_dir, args):
    all_rels_temp_scores = []
    all_rels_pred_scores = []
    avg_edit = 0.0
    count = 0
    out_file = open(output_dir.split('/')[1] + '.tsv', 'w', encoding='utf-8')
    for filename in os.listdir(output_dir):
        if filename.endswith('.out'):
            args.inp = os.path.join(output_dir, filename)
            templates, stat, subjs, objs = load_out_file(args)
            all_preds = {}
            if len(templates) != len(stat):
                continue
            relation_type = filename.split('.')[0]
            for template, pred in zip(templates, stat):
                preds = [int(x) for x in pred]
                tokens = nltk.word_tokenize(template[0])
                # [X] -> [ X ]
                tokens = [t for t in tokens if t not in ('[', ']')]
                normalized_template = ' '.join(tokens)
                all_preds[normalized_template] = preds
            pairwise_templates = itertools.combinations(all_preds.keys(), 2)
            template_scores = []
            pred_scores = []
            for pair in pairwise_templates:
                temp_edit_score = nltk.edit_distance(pair[0].split(), pair[1].split())
                avg_edit += temp_edit_score
                count += 1
                pred_l1_score = calc_l1_distance(all_preds[pair[0]], all_preds[pair[1]]) / len(all_preds[pair[0]])
                template_scores.append(temp_edit_score)
                pred_scores.append(pred_l1_score)

                pred_temp_ratio = pred_l1_score / temp_edit_score
                out_file.write('\t'.join((relation_type, pair[0], pair[1], str(pred_temp_ratio))) + '\n')
            # normalize temp edit distance
            min_ = min(template_scores)
            max_ = max(template_scores)
            template_scores = [(s - min_) / (max_ - min_) for s in template_scores]

            print(relation_type, pearsonr(template_scores, pred_scores)[0])
            all_rels_temp_scores += template_scores
            all_rels_pred_scores += pred_scores
    print('Avg edit: ', avg_edit / count)
    return all_rels_temp_scores, all_rels_pred_scores

def rank_by_pairwsie_distance(output_dir, args):
    all_rels_pairs = []
    all_rels_temp_scores = []
    all_rels_pred_scores = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.out'):
            args.inp = os.path.join(output_dir, filename)
            templates, stat, subjs, objs = load_out_file(args)
            all_preds = {}
            if len(templates) != len(stat):
                continue
            relation_type = filename.split('.')[0]

            new_temps = []
            for template, pred in zip(templates, stat):
                preds = [int(x) for x in pred]
                tokens = nltk.word_tokenize(template[0])
                # [X] -> [ X ]
                tokens = [t for t in tokens if t not in ('[', ']')]
                normalized_template = ' '.join(tokens)
                all_preds[normalized_template] = preds
                new_temps.append(normalized_template)

            pairs = []
            template_scores = []
            pred_scores = []
            for temp in new_temps[1:]:
                pair = (new_temps[0], temp)
                temp_edit_score = nltk.edit_distance(pair[0].split(), pair[1].split())
                if temp_edit_score != 1:
                    continue
                temp_edit_score = temp_edit_score / (len(pair[0].split()) + len(pair[1].split()))
                pred_l1_score = np.mean(all_preds[pair[1]]) - np.mean(all_preds[pair[0]])
                pairs.append((relation_type, pair))
                template_scores.append(temp_edit_score)
                pred_scores.append(pred_l1_score)

            all_rels_pairs += pairs
            all_rels_temp_scores += template_scores
            all_rels_pred_scores += pred_scores

    for temp_dist, pred_dist, pair in sorted(zip(all_rels_temp_scores, all_rels_pred_scores, all_rels_pairs), key=lambda x: (-x[1]/x[0])):
        print(pair, temp_dist, pred_dist)
        input()

    return all_rels_temp_scores, all_rels_pred_scores

def rank_edit(args):
    mined_dir = 'output/exp_mt_7/trex/'
    rank_by_pairwsie_distance(mined_dir, args)


def template_div(args):
    print('processing mined')
    mined_dir = 'output/exp_allpids_top30/trex/'
    mined_all_rels_temp_scores, mined_all_rels_pred_scores = pairwise_distance(mined_dir, args)
    print('Overall', pearsonr(mined_all_rels_temp_scores, mined_all_rels_pred_scores)[0])

    print('processing paraphrase')
    para_dir = 'output/exp_mt_7/trex/'
    para_all_rels_temp_scores, para_all_rels_pred_scores = pairwise_distance(para_dir, args)
    print('Overall', pearsonr(para_all_rels_temp_scores, para_all_rels_pred_scores)[0])

    print('plot mined')
    mined_df = pd.DataFrame({'temp': mined_all_rels_temp_scores,
                             'pred': mined_all_rels_pred_scores})
    mined_df['temp_bins'] = pd.cut(x=mined_df['temp'], bins=[-1, 0.2, 0.4, 0.6, 0.8, 1.0],
                                   labels=['[0.0, 0.2]', '(0.2, 0.4]', '(0.4, 0.6]', '(0.6, 0.8]', '(0.8, 1.0]'])
    plt.rcParams.update({'font.size': 14})

    boxplot = mined_df.boxplot(column='pred', by='temp_bins', showfliers=False, vert=True, showmeans=True,
                               boxprops=dict(linewidth=2),
                               flierprops=dict(linewidth=2),
                               medianprops=dict(linewidth=2),
                               whiskerprops=dict(linewidth=2),
                               capprops=dict(linewidth=2),
                               )
    boxplot.set_xlabel('bucketed normalized edit distance between mined prompts', labelpad=10)
    boxplot.set_ylabel('prediction divergence', labelpad=10)
    plt.suptitle("")
    plt.title("")
    plt.tight_layout()

    plt.savefig('correlation_template_div.png')
    plt.savefig('correlation_template_div.eps', format='eps')


def subj_obj_distance_calc(output_dir, args):
    all_rels_length_scores = []
    all_rels_accuracy_scores = []
    for filename in os.listdir(output_dir):
        if filename.endswith('.out'):
            args.inp = os.path.join(output_dir, filename)
            templates, stat, subjs, objs = load_out_file(args)
            if len(templates) != len(stat):
                continue
            relation_type = filename.split('.')[0]
            if relation_type == 'P190':
                # constant
                continue
            length_scores = []
            accuracy_scores = []
            min_length = 0
            max_length = 1000
            for template, pred in zip(templates, stat):
                pred = [int(x) for x in pred]
                acc = sum(pred) / len(pred)
                tokens = nltk.word_tokenize(template[0])
                tokens = [t for t in tokens if t not in ('[', ']')]
                subj_idx = tokens.index('X')
                obj_idx = tokens.index('Y')
                distance = abs(subj_idx - obj_idx)
                if distance > min_length:
                    min_length = distance
                if distance < max_length:
                    max_length = distance
                length_scores.append(distance)
                accuracy_scores.append(acc)
            # normalize lengths
            for idx in range(len(length_scores)):
                length_scores[idx] = (length_scores[idx] - min_length) / (max_length - min_length)
            print(relation_type, pearsonr(length_scores, accuracy_scores)[0])
            all_rels_length_scores += length_scores
            all_rels_accuracy_scores += accuracy_scores
    return all_rels_length_scores, all_rels_accuracy_scores


def subj_obj_distance_analysis(args):
    print('processing mined')
    mined_dir = 'output/exp_allpids_top30/trex/'
    mined_all_rels_length_scores, mined_all_rels_accuracy_scores = subj_obj_distance_calc(mined_dir, args)
    print('Overall', pearsonr(mined_all_rels_length_scores, mined_all_rels_accuracy_scores)[0])

    print('processing paraphrase')
    para_dir = 'output/exp_mt_7/trex/'
    para_all_rels_length_scores, para_all_rels_accuracy_scores = subj_obj_distance_calc(para_dir, args)
    print('Overall', pearsonr(para_all_rels_length_scores, para_all_rels_accuracy_scores)[0])

    fig, ax = plt.subplots()
    ax.set_xlabel('subj-obj distance')
    ax.set_ylabel('accuracy')

    ax.scatter(mined_all_rels_length_scores, mined_all_rels_accuracy_scores, c='blue', label='mined',
               alpha=0.3, edgecolors='none')
    ax.scatter(para_all_rels_length_scores, para_all_rels_accuracy_scores, c='red', label='paraphrase',
               alpha=0.3, edgecolors='none')
    ax.legend()
    ax.grid(True)
    plt.savefig('distance_accuracy.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze output log')
    parser.add_argument('--task', type=str, help='task', required=True, 
        choices=['out', 'wikidata', 'sort', 'major_class', 'get_train_data',
                 'get_ppdb', 'case', 'merge_all_rel', 'split_dev', 'weight_ana',
                 'out_ana_opti', 'bt_filter', 'case_study', 'out_all_ana', 'sub_obj',
                 'template_divergence', 'subj_obj_distance', 'rank_edit'])
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--obj_file', type=str, help='obj file', default=None)
    parser.add_argument('--out', type=str, help='output file')
    parser.add_argument('--temp_file', type=str, help='pid and raw temp file', default=None)
    parser.add_argument('--beam', type=int, help='beam size', default=None)
    parser.add_argument('--exclude_first', help='whether to exclude the first template (manual)', action='store_true')
    args = parser.parse_args()

    if args.task == 'out':
        out_ana(args)
    elif args.task == 'out_all_ana':
        out_all_ana(args)
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
    elif args.task == 'case_study':
        case_study(args)
    elif args.task == 'sub_obj':
        sub_obj(args)
    elif args.task == 'template_divergence':
        template_div(args)
    elif args.task == 'subj_obj_distance':
        subj_obj_distance_analysis(args)
    elif args.task == 'rank_edit':
        rank_edit(args)
