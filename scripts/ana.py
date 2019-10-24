import argparse
import numpy as np
import json
import os


def out_ana(args):
    stat = []
    templates = []
    with open(args.inp, 'r') as fin:
        for l in fin:
            if l.startswith('P1all '):
                stat.append(list(map(float, l.strip().split(' ')[1].split('\t'))))
            elif l.startswith("{'dataset_filename':"):
                templates.append(eval(l.strip())['template'])
    stat = np.array(stat)
    first = np.mean(stat[0])  # the first template is manually designed
    ensemble_score = np.mean(np.max(stat, 0))  # ensemble all the templates
    if len(stat) > 1:
        temp_scores = np.mean(stat[1:], -1)
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
        for pid in pattern:
            if pid == root_pid or pid.startswith(root_pid + '_'):
                for sni in pattern[pid]['snippet']:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='analyze output log')
    parser.add_argument('--task', type=str, help='task', required=True, choices=['out', 'wikidata'])
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()

    if args.task == 'out':
        out_ana(args)
    elif args.task == 'wikidata':
        wikidata_to_trex(args)
