from copy import deepcopy
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
sys.path.append('..')
import argparse

from lama.modules import build_model_by_name
import lama.options as options
from lama.modules import base_connector
from lama import attacks

def main(args):
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
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


if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models_names",
        "--m",
        help="comma separated list of language models",
        required=True,
    )
    args = parser.parse_args()
    '''
    parser = options.get_eval_generation_parser()
    args = options.parse_args(parser)
    main(args)
