# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from typing import List
import re
import torch

MASK = "[MASK]"
BERT_UNK = "[UNK]"
BERT_CLS = "[CLS]"
BERT_SEP = "[SEP]"
BERT_PAD = "[PAD]"
ELMO_UNK = "<UNK>"
ELMO_START_SENTENCE = "<S>"
ELMO_END_SENTENCE = "</S>"
OPENAI_UNK = "<unk>"
OPENAI_EOS = "<eos>"
ROBERTA_MASK = "<mask>"
ROBERTA_START_SENTENCE = "<s>"
ROBERTA_END_SENTENCE = "</s>"
ROBERTA_VOCAB_SIZE = 50266

SPECIAL_SYMBOLS = [
    MASK,
    BERT_UNK,
    BERT_CLS,
    BERT_SEP,
    BERT_PAD,
    ELMO_UNK,
    ELMO_START_SENTENCE,
    ELMO_END_SENTENCE,
    OPENAI_UNK,
    OPENAI_EOS
    ]

SPACE_NORMALIZER = re.compile(r"\s+")


def default_tokenizer(line):
    """Default tokenizer for models that don't have one

    Args:
        line: a string representing a sentence

    Returns:
        A list of tokens
    """

    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    line = line.replace(MASK, " "+str(MASK)+" ") #make sure MASK is correctly splitted

    # fix tokenization for parentheses
    line = line.replace('(', " ( ")
    line = line.replace(')', " ) ")

    # fix tokenization for comma
    line = line.replace(',', " , ")

    # fix tokenization for -- (e.g., 1954--1988)
    line = line.replace('--', " -- ")

    result = line.split()
    return result


extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])


class Base_Connector():

    EMB_DIM = 0

    def __init__(self):

        # these variables should be initialized
        self.vocab = None

        # This defines where the device where the model is. Changed by try_cuda.
        self._model_device = 'cpu'

    @property
    def model(self):
        raise NotImplementedError

    @property
    def mask_token(self):
        return MASK

    def tokenize(self, text: str):
        raise NotImplementedError

    def optimize_top_layer(self, vocab_subset):
        """
        optimization for some LM
        """
        pass

    def _init_inverse_vocab(self):
        self.inverse_vocab = {w: i for i, w in enumerate(self.vocab)}

    def try_cuda(self):
        """Move model to GPU if one is available."""
        if torch.cuda.is_available():
            if self._model_device != 'cuda':
                print('Moving model to CUDA')
                self._cuda()
                self._model_device = 'cuda'
        else:
            print('No CUDA found')

    def _cuda(self):
        """Move model to GPU."""
        raise NotImplementedError

    def init_indices_for_filter_logprobs(self, vocab_subset, logger=None):
        index_list = []
        new_vocab_subset = []
        for word in vocab_subset:
            if word in self.inverse_vocab:
                inverse_id = self.inverse_vocab[word]
                index_list.append(inverse_id)
                new_vocab_subset.append(word)
            else:
                msg = "word {} from vocab_subset not in model vocabulary!".format(word)
                if logger is not None:
                    logger.warning(msg)
                else:
                    print("WARNING: {}".format(msg))

        # 1. gather correct indices
        indices = torch.as_tensor(index_list)
        return indices, index_list

    def filter_logprobs(self, log_probs, indices):
        new_log_probs = log_probs.index_select(dim=2 , index=indices)
        return new_log_probs

    def get_id(self, string):
        raise NotImplementedError()

    def get_generation(self, sentences, logger=None):
        [log_probs], [token_ids], [masked_indices] = self.get_batch_generation(
            [sentences], logger=logger, try_cuda=False)
        return log_probs, token_ids, masked_indices

    def get_batch_generation(self, sentences_list, logger= None, try_cuda=True):
        raise NotImplementedError()

    def get_contextual_embeddings(self, sentences):
        """Compute the contextual embeddings of a list of sentences

        Parameters:
        sentences (list[list[string]]): list of elements. Each element is a list
                                        that contains either a single sentence
                                        or two sentences

        Returns:
        encoder_layers (list(Tensor)): a list of the full sequences of encoded-hidden-states
                            at the end of each attention block (e.g., 12 full
                            sequences for BERT-base,), each encoded-hidden-state
                            is a torch.FloatTensor of size [batch_size,
                            sequence_length, hidden_size]
        sentence_lengths (list[int]): list of lenghts for the sentences in the
                                      batch
        tokenized_text_list: (list[list[string]]): tokenized text for the sentences
                                                   in the batch
        """
        raise NotImplementedError()

    def add_hooks(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == self.EMB_DIM:
                    module.weight.requires_grad = True
                    module.register_backward_hook(extract_grad_hook)

    def get_embedding_weight(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.Embedding):
                if module.weight.shape[0] == self.EMB_DIM:
                    return module.weight.detach()

    def get_rc_loss(self, sentences_list: List[str], try_cuda: bool = False):
        raise NotImplementedError()

    def zero_grad(self):
        self.model.zero_grad()

    def tokenizer_with_mask(self, sentence: str, split_pattern='\[|\]'):
        tokenized_text = []
        bracket_indices = []
        unbracket_indices = []
        for chunk_idx, chunk in enumerate(re.split(split_pattern, sentence)):
            chunk = chunk.strip()
            if chunk:
                chunk_tokens = self.tokenizer.tokenize(chunk)
                tokenized_text.extend(chunk_tokens)
                mask = 0 if chunk_idx % 2 == 0 else 1  # "(" and ")" always occur in pairs
                bracket_indices.extend([mask] * len(chunk_tokens))
                unbracket_indices.extend([1 - mask] * len(chunk_tokens))
        return tokenized_text, bracket_indices, unbracket_indices
