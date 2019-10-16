import sys
import torch
import numpy as np
sys.path.append('..')
import json
from collections import defaultdict
from tqdm import tqdm

from lama.modules import build_model_by_name
import lama.options as options
from lama.modules import base_connector
from lama import attacks

def main(args):
    try_cuda = torch.cuda.is_available()

    model = build_model_by_name(args.models_names[0], args)
    model.add_hooks()
    embedding_weight = model.get_embedding_weight()

    sentences = ['The theory of relativity was developed by Einstein.',
                 'Windows was developed by Microsoft.']

    sentences = ['(The theory of relativity) was found by (Einstein.)']

    sentences = ['(Barack Obama) was born in (Hawaii.)']

    sentences = ['Him (speaks English.)']

    for _ in range(50):
        for token_to_flip in range(0, 3):  # TODO: for each token in the trigger
            # back propagation
            model.zero_grad()
            loss, tokens, _, unbracket_mask = model.get_rc_loss(sentences, try_cuda=try_cuda)
            # SHAPE: (batch_size, seq_len)
            unbracket_mask = unbracket_mask.bool()
            loss.backward()
            print(loss)

            # SHAPE: (batch_size, seq_len, emb_dim)
            grad = base_connector.extracted_grads[0]
            bs, _, emb_dim = grad.size()
            base_connector.extracted_grads = []

            # TODO
            # SHAPE: (batch_size, unbracket_len, emb_dim)
            #grad = grad.masked_select(F.pad(unbracket_mask, (1, 0), 'constant', False)[:, :-1].unsqueeze(-1)).view(bs, -1, emb_dim)
            grad = grad.masked_select(unbracket_mask.unsqueeze(-1)).view(bs, -1, emb_dim)
            # SHAPE: (1, emb_dim)
            grad = grad.sum(dim=0)[token_to_flip].unsqueeze(0)
            print(grad)

            # SHAPE: (batch_size, unbracket_len)
            tokens = tokens.masked_select(unbracket_mask).view(bs, -1)
            token_tochange = tokens[0][token_to_flip].item()

            # Use hotflip (linear approximation) attack to get the top num_candidates
            candidates = attacks.hotflip_attack(grad, embedding_weight, [token_tochange],
                                                increase_loss=False, num_candidates=10)[0]
            print(model.tokenizer.convert_ids_to_tokens([token_tochange]), model.tokenizer.convert_ids_to_tokens(candidates))
            input()


def pattern_score(args, pattern_json, output_file):
    try_cuda = torch.cuda.is_available()
    model = build_model_by_name(args.models_names[0], args)
    with open(pattern_json, 'r') as fin:
        pattern_json = json.load(fin)

    batch_size = 32
    pid2pattern = defaultdict(lambda: {})
    for pid in tqdm(sorted(pattern_json)):
        #if not pid.startswith('P69_'):
        #    continue
        snippets = pattern_json[pid]['snippet']
        occs = pattern_json[pid]['occs']
        for (snippet, direction), count in snippets:
            if len(snippet) <= 5 or len(snippet) >= 100:  # longer than 5 chars
                continue
            loss = 0
            num_batch = np.ceil(len(occs) / batch_size)
            for b in range(0, len(occs), batch_size):
                occs_batch = occs[b:b + batch_size]
                sentences = ['{} {} ({})'.format(h, snippet, t) i
                             f direction == 1 else '{} {} ({})'.format(t, snippet, h) for h, t in occs_batch]
                #print((snippet, direction), count)
                #print(sentences)
                #input()
                loss += model.get_rc_loss(sentences, try_cuda=try_cuda)[0].item()
            pid2pattern[pid][snippet] = loss / num_batch
        #print(pid)
        #print(sorted(pid2pattern[pid].items(), key=lambda x: x[1]))
        #input()

    with open(output_file, 'w') as fout:
        for pid, pats in pid2pattern.items():
            pats = sorted(pats.items(), key=lambda x: x[1])
            fout.write('{}\t{}\n'.format(pid, json.dumps(pats)))


if __name__ == '__main__':
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    parser = options.get_eval_generation_parser()
    args = options.parse_args(parser)
    #main(args)
    pattern_score(args, 'patterns.json', 'output/patterns.txt')
