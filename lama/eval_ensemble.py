import json
import os
import torch.nn.functional as F
from lama.modules import build_model_by_name
from lama.utils import load_vocab
import lama.options as options
import lama.evaluation_metrics as evaluation_metrics


def get_eval_ensemble_parser():
    parser = options.get_eval_generation_parser()
    parser.add_argument('--subject',  help='the subject of a particular fact', required=True)
    parser.add_argument('--relation',  help='the relation of a particular fact', required=True)
    parser.add_argument('--prompts', help='directory to the prompts and weights', default='prompt/mine')
    return parser


def load_prompt_weights(filename: str):
    prompts, weights = [], []
    with open(filename, 'r') as fin:
        for l in fin:
            l = json.loads(l)
            prompt, weight = l['template'], l['weight']
            prompts.append(prompt)
            weights.append(weight)
    return prompts, weights


def parse_prompt(prompt, subject_label, object_label):
    SUBJ_SYMBOL = '[X]'
    OBJ_SYMBOL = '[Y]'
    prompt = prompt.replace(SUBJ_SYMBOL, subject_label)
    prompt = prompt.replace(OBJ_SYMBOL, object_label)
    return [prompt]


def main(args):

    if not args.subject and not args.relation:
        raise ValueError('You need to specify --subject and --relation to query language models.')

    print('Language Models: {}'.format(args.models_names))

    models = {}
    for lm in args.models_names:
        models[lm] = build_model_by_name(lm, args)

    vocab_subset = None
    if args.common_vocab_filename is not None:
        common_vocab = load_vocab(args.common_vocab_filename)
        print('Common vocabulary size: {}'.format(len(common_vocab)))
        vocab_subset = [x for x in common_vocab]

    prompt_file = os.path.join(args.prompts, args.relation + '.jsonl')
    if not os.path.exists(prompt_file):
        raise ValueError('Relation "{}" does not exist.'.format(args.relation))
    prompts, weights = load_prompt_weights(prompt_file)

    for model_name, model in models.items():
        print('\n{}:'.format(model_name))

        index_list = None
        if vocab_subset is not None:
            filter_logprob_indices, index_list = model.init_indices_for_filter_logprobs(vocab_subset)

        ensemble_log_probs = 0
        for prompt, weight in zip(prompts, weights):
            prompt = parse_prompt(prompt, args.subject, model.mask_token)
            log_prob, [token_ids], [masked_indices], _, _ = model.get_batch_generation([prompt], try_cuda=True)

            if vocab_subset is not None:
                filtered_log_probs = model.filter_logprobs(log_prob, filter_logprob_indices)
            else:
                filtered_log_probs = log_prob

            # rank over the subset of the vocab (if defined) for the SINGLE masked tokens
            if masked_indices and len(masked_indices) > 0:
                filtered_log_probs = filtered_log_probs[0][masked_indices[0]]
                ensemble_log_probs += filtered_log_probs * weight

        ensemble_log_probs = F.log_softmax(ensemble_log_probs, dim=0)
        evaluation_metrics.get_ranking(ensemble_log_probs, model.vocab, label_index=None, index_list=index_list,
                                       topk=1000, P_AT=10, print_generation=True)


if __name__ == '__main__':
    parser = get_eval_ensemble_parser()
    args = options.parse_args(parser)
    main(args)
