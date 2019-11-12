# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import sys
import logging
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(abspath(__file__))))
print(dirname(dirname(abspath(__file__))))

import argparse
from batch_eval_KB_completion import main as run_evaluation
from batch_eval_KB_completion import load_file
from lama.modules import build_model_by_name
from lama.modules import TempModel
import torch.optim as optim
import torch
import pprint
import statistics
from copy import deepcopy
import json
from os import listdir
import os
from os.path import isfile, join
from shutil import copyfile
from collections import defaultdict

LMs = [
    {
        "lm": "transformerxl",
        "label": "transformerxl",
        "models_names": ["transformerxl"],
        "transformerxl_model_name": 'transfo-xl-wt103',
        "transformerxl_model_dir": "pre-trained_language_models/transformerxl/transfo-xl-wt103/"
    },
    {
        "lm": "elmo",
        "label": "elmo",
        "models_names": ["elmo"],
        "elmo_model_name": 'elmo_2x4096_512_2048cnn_2xhighway',
        "elmo_vocab_name": 'vocab-2016-09-10.txt',
        "elmo_model_dir": "pre-trained_language_models/elmo/original",
        "elmo_warm_up_cycles": 10
    },
        {
        "lm": "elmo",
        "label": "elmo5B",
        "models_names": ["elmo"],
        "elmo_model_name": "elmo_2x4096_512_2048cnn_2xhighway_5.5B",
        "elmo_vocab_name": "vocab-enwiki-news-500000.txt",
        "elmo_model_dir": "pre-trained_language_models/elmo/original5.5B/",
        "elmo_warm_up_cycles": 10
    },
    {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12"
    },
    {
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    }
]

LMs = [
    {
        "lm": "bert",
        "label": "bert_base",
        "models_names": ["bert"],
        "bert_model_name": "bert-base-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-12_H-768_A-12"
    }
]
'''
LMs = [
    {
        "lm": "roberta",
        "label": "roberta_base",
        "models_names": ["roberta"],
        "roberta_model_name": "model.pt",
        "roberta_model_dir": "pre-trained_language_models/roberta/roberta.base",
        "roberta_vocab_name": "dict.txt"
    }
]
'''

def run_experiments(
    relations,
    data_path_pre,
    data_path_post,
    refine_template,
    get_objs,
    batch_size,
    dynamic=None,
    use_prob=False,
    bt_obj=None,
    temp_model=None,
    save=None,
    load=None,
    feature_dir=None,
    enforce_prob=True,
    num_feat=1,
    input_param={
        "lm": "bert",
        "label": "bert_large",
        "models_names": ["bert"],
        "bert_model_name": "bert-large-cased",
        "bert_model_dir": "pre-trained_language_models/bert/cased_L-24_H-1024_A-16",
    },
):
    model = None
    pp = pprint.PrettyPrinter(width=41, compact=True)

    all_Precision1 = []
    type_Precision1 = defaultdict(list)
    type_count = defaultdict(list)

    results_file = open("last_results.csv", "w+")

    if refine_template:
        refine_temp_fout = open(refine_template, 'w')
        new_relations = []
        templates_set = set()

    rel2numtemp = {}
    for relation in relations:  # collect templates
        if 'template' in relation:
            if type(relation['template']) is not list:
                relation['template'] = [relation['template']]
        rel2numtemp[relation['relation']] = len(relation['template'])

    if temp_model is not None:
        if temp_model.startswith('mixture'):
            method = temp_model.split('_')[1]
            if method == 'optimize':  # (extract feature) + optimize
                temp_model = TempModel(rel2numtemp, enforce_prob=enforce_prob, num_feat=num_feat)
                temp_model.train()
                optimizer = optim.Adam(temp_model.parameters(), lr=1e-1)
                temp_model = (temp_model, optimizer)
            elif method == 'precompute':  # extract feature
                temp_model = (None, 'precompute')
            elif method == 'predict':  # predict
                temp_model = TempModel(rel2numtemp, enforce_prob=enforce_prob, num_feat=num_feat)  # TODO: number of feature
                if load is not None:
                    temp_model.load_state_dict(torch.load(load))
                temp_model.eval()
                temp_model = (temp_model, None)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    for relation in relations:
        pp.pprint(relation)
        PARAMETERS = {
            "relation": relation["relation"],
            "dataset_filename": "{}/{}{}".format(
                data_path_pre, relation["relation"], data_path_post
            ),
            "common_vocab_filename": "pre-trained_language_models/common_vocab_cased.txt",
            "template": "",
            "bert_vocab_name": "vocab.txt",
            "batch_size": batch_size,
            "logdir": "output",
            "full_logdir": "output/results/{}/{}".format(
                input_param["label"], relation["relation"]
            ),
            "lowercase": False,
            "max_sentence_length": 100,
            "threads": -1,
            "interactive": False,
        }
        dev_param = deepcopy(PARAMETERS)
        dev_param['dataset_filename'] = '{}/{}{}'.format(data_path_pre + '_dev', relation['relation'], data_path_post)

        if 'template' in relation:
            PARAMETERS['template'] = relation['template']
            dev_param['template'] = relation['template']

        PARAMETERS.update(input_param)
        dev_param.update(input_param)
        print(PARAMETERS)

        args = argparse.Namespace(**PARAMETERS)
        dev_args = argparse.Namespace(**dev_param)

        # see if file exists
        try:
            data = load_file(args.dataset_filename)
        except Exception as e:
            print("Relation {} excluded.".format(relation["relation"]))
            print("Exception: {}".format(e))
            continue

        if model is None:
            [model_type_name] = args.models_names
            model = build_model_by_name(model_type_name, args)

        if temp_model is not None:
            if temp_model[1] == 'precompute':
                features = run_evaluation(
                    args, shuffle_data=False, model=model, refine_template=bool(refine_template),
                    get_objs=get_objs, dynamic=dynamic, use_prob=use_prob, bt_obj=bt_obj, temp_model=temp_model)
                print('save features for {}'.format(relation['relation']))
                torch.save(features, os.path.join(save, relation['relation'] + '.pt'))
                continue
            elif temp_model[1] is not None:  # train temp model
                if feature_dir is None:
                    max_dev_acc = -1e10
                    es = 0
                    for e in range(100):
                        loss = run_evaluation(args, shuffle_data=True, model=model,
                                              refine_template=bool(refine_template),
                                              get_objs=get_objs, dynamic=dynamic,
                                              use_prob=use_prob, bt_obj=bt_obj,
                                              temp_model=temp_model)
                        dev_acc = run_evaluation(args, shuffle_data=False, model=model,
                                                  refine_template=bool(refine_template),
                                                  get_objs=get_objs, dynamic=dynamic,
                                                  use_prob=use_prob, bt_obj=bt_obj,
                                                 temp_model=(temp_model[0], None))
                        if dev_acc > max_dev_acc:
                            max_dev_acc = dev_acc
                            es = 0
                        else:
                            es += 1
                            if es >= 3:  # early stop
                                print('early stop')
                                break
                        print('Loss\t{}\t{}\t{}'.format(relation['relation'], loss, dev_acc))
                else:
                    temp_model_, optimizer = temp_model
                    temp_model_.cuda()
                    min_loss = 1e10
                    es = 0
                    for e in range(500):
                        # SHAPE: (num_sample, num_temp)
                        feature = torch.load(os.path.join(feature_dir, args.relation + '.pt')).cuda()
                        #weight = feature.mean(0)
                        #temp_model[0].set_weight(args.relation, weight)
                        optimizer.zero_grad()
                        loss = temp_model_(args.relation, feature)
                        if os.path.exists(feature_dir + '__dev'):  # TODO: debug
                            dev_feature = torch.load(os.path.join(feature_dir + '_dev', args.relation + '.pt')).cuda()
                            dev_loss = temp_model_(args.relation, dev_feature)
                        else:
                            dev_loss = loss
                        loss.backward()
                        optimizer.step()
                        if dev_loss - min_loss < -1e-3:
                            min_loss = dev_loss
                            es = 0
                        else:
                            es += 1
                            if es >= 10:
                                print('early stop')
                                break
                continue

        Precision1 = run_evaluation(args, shuffle_data=False, model=model,
                                    refine_template=bool(refine_template),
                                    get_objs=get_objs, dynamic=dynamic,
                                    use_prob=use_prob, bt_obj=bt_obj,
                                    temp_model=temp_model)

        if get_objs:
            return

        if refine_template and Precision1 is not None:
            if Precision1 in templates_set:
                continue
            templates_set.add(Precision1)
            new_relation = deepcopy(relation)
            new_relation['old_template'] = new_relation['template']
            new_relation['template'] = Precision1
            new_relations.append(new_relation)
            refine_temp_fout.write(json.dumps(new_relation) + '\n')
            refine_temp_fout.flush()
            continue

        print("P@1 : {}".format(Precision1), flush=True)
        all_Precision1.append(Precision1)

        results_file.write(
            "{},{}\n".format(relation["relation"], round(Precision1 * 100, 2))
        )
        results_file.flush()

        if "type" in relation:
            type_Precision1[relation["type"]].append(Precision1)
            data = load_file(PARAMETERS["dataset_filename"])
            type_count[relation["type"]].append(len(data))

    if refine_template:
        refine_temp_fout.close()
        return

    if temp_model is not None:
        if save is not None and temp_model[0] is not None:
            torch.save(temp_model[0].state_dict(), save)
        return

    mean_p1 = statistics.mean(all_Precision1)
    print("@@@ {} - mean P@1: {}".format(input_param["label"], mean_p1))
    results_file.close()

    for t, l in type_Precision1.items():

        print(
            "@@@ ",
            input_param["label"],
            t,
            statistics.mean(l),
            sum(type_count[t]),
            len(type_count[t]),
            flush=True,
        )

    return mean_p1, all_Precision1


def get_TREx_parameters(data_path_pre="data/"):
    relations = load_file("{}relations.jsonl".format(data_path_pre))
    data_path_pre += "TREx/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_GoogleRE_parameters():
    relations = [
        {"relation": "place_of_birth", "template": "[X] was born in [Y] ."},
        {"relation": "date_of_birth", "template": "[X] (born [Y])."},
        {"relation": "place_of_death", "template": "[X] died in [Y] ."},
    ]
    data_path_pre = "data/Google_RE/"
    data_path_post = "_test.jsonl"
    return relations, data_path_pre, data_path_post


def get_relation_phrase_parameters(args):
    relations = load_file(args.rel_file)
    if args.top:
        print('use top {} relations'.format(args.top))
        relations = relations[:args.top]
    if args.ensemble:
        temps = [rel['template'] for rel in relations]
        relations[0]['template'] = temps
        relations = [relations[0]]
    data_path_pre = args.prefix
    data_path_post = args.suffix
    bt_obj = args.bt_obj
    temp_model = args.temp_model
    save = args.save
    load = args.load
    feature_dir = args.feature_dir
    enforce_prob = args.enforce_prob
    num_feat = args.num_feat
    return relations, data_path_pre, data_path_post, args.refine_template, \
           args.get_objs, args.batch_size, args.dynamic, args.use_prob, \
           bt_obj, temp_model, save, load, feature_dir, enforce_prob, num_feat


def get_test_phrase_parameters(args):
    #relations = [{'relation': 'P27', 'template': '[X] what a terrorist incident in [Y] .'}]
    #relations = [{'relation': 'P27', 'template': '[X] is an [Y] citizen .'}]
    #relations = [{"relation": "P1001", "template": "[X] is the first,,,,andandthe President of [Y] .", "label": None, "description": None, "type": "N-M", "wikipedia_count": 21, "old_template": "[X] in the australian state of [Y] ."}]
    #relations = [{"relation": "P108", "template": ["[X] works for [Y] .", "[Y] commentator [X] ."]}]
    #relations = [{"relation": "P108", "template": ["[Y] commentator [X] ."]}]
    #relations = [{"relation": "P108", "template": ["[X] works for [Y] ."]}]
    #relations = [{"relation": "P19", "template": ["[X] was born in [Y] .", "[X] label [Y] .", "[X] died at [Y] ."]}]
    #relations = [{"relation": "P19", "template": "[X] is born in [Y] ."}]
    relations = [{"relation": "P19", "template": ["[X] is born in [Y] .", "[X] died at [Y] ."]},
                 {"relation": "P108", "template": ["[X] works for [Y] .", "[Y] commentator [X] ."]}]
    data_path_pre = "data/TREx"
    data_path_post = ".jsonl"
    refine_template = None  #'test.out'
    get_objs = False
    batch_size = 32
    dynamic = 'none'
    use_prob = False
    bt_obj = 5
    temp_model = 'mixture_predict'
    save = None
    load = None
    feature_dir = None
    enforce_prob = True
    num_feat = 1
    return relations, data_path_pre, data_path_post, refine_template, get_objs, \
           batch_size, dynamic, use_prob, bt_obj, temp_model, save, load, feature_dir, enforce_prob, num_feat


def get_ConceptNet_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "ConceptNet/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def get_Squad_parameters(data_path_pre="data/"):
    relations = [{"relation": "test"}]
    data_path_pre += "Squad/"
    data_path_post = ".jsonl"
    return relations, data_path_pre, data_path_post


def run_all_LMs(parameters):
    for ip in LMs:
        print(ip["label"])
        run_experiments(*parameters, input_param=ip)


if __name__ == "__main__":
    '''
    print("1. Google-RE")    
    parameters = get_GoogleRE_parameters()
    run_all_LMs(parameters)

    print("2. T-REx")    
    parameters = get_TREx_parameters()
    run_all_LMs(parameters)

    print("3. ConceptNet")
    parameters = get_ConceptNet_parameters()
    run_all_LMs(parameters)

    print("4. SQuAD")
    parameters = get_Squad_parameters()
    run_all_LMs(parameters)
    '''
    logging.disable(logging.WARNING)
    parser = argparse.ArgumentParser(description='run exp for multiple relational phrase')
    parser.add_argument('--rel_file', type=str, default='data/Google_RE_patty_template/place_of_death.jsonl')
    parser.add_argument('--refine_template', type=str, default=None)
    parser.add_argument('--prefix', type=str, default='data/Google_RE/')
    parser.add_argument('--suffix', type=str, default='_test.jsonl')
    parser.add_argument('--top', type=int, default=None)
    parser.add_argument('--ensemble', help='ensemble probs of different templates', action='store_true')
    parser.add_argument('--get_objs', help='print out objects for evaluation', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dynamic', type=str, help='dynamically select template', default='none')
    parser.add_argument('--use_prob', help='use prob instead of log prob', action='store_true')
    parser.add_argument('--temp_model', help='which temp model to use to learn temp weights', default=None)
    parser.add_argument('--save', help='path to save temp model', default=None)
    parser.add_argument('--load', help='path to load temp model', default=None)
    parser.add_argument('--feature_dir', help='dir to features', default=None)
    parser.add_argument('--bt_obj', type=int, help='beam size of bach translation', default=None)
    parser.add_argument('--enforce_prob', help='whether force the feature to be prob', action='store_true')
    parser.add_argument('--num_feat', type=int, help='number of features', default=1)
    args = parser.parse_args()
    parameters = get_relation_phrase_parameters(args)
    #parameters = get_test_phrase_parameters(args)
    run_all_LMs(parameters)
