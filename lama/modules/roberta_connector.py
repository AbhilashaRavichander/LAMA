# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from fairseq.models.roberta import RobertaModel
from fairseq import utils
from fairseq.tokenizer import tokenize_line
import torch
from lama.modules.base_connector import *


class RobertaVocab(object):
    def __init__(self, roberta):
        self.roberta = roberta

    def __getitem__(self, arg):
        value = ""
        try:
            predicted_token_bpe = self.roberta.task.source_dictionary.string([arg])
            if (
                predicted_token_bpe.strip() == ROBERTA_MASK
                or predicted_token_bpe.strip() == ROBERTA_START_SENTENCE
            ):
                value = predicted_token_bpe.strip()
            else:
                value = self.roberta.bpe.decode(str(predicted_token_bpe)).strip()
        except Exception as e:
            print(arg)
            print(predicted_token_bpe)
            print(value)
            print("Exception {} for input {}".format(e, arg))
        return value


class Roberta(Base_Connector):
    def __init__(self, args):
        super().__init__()
        roberta_model_dir = args.roberta_model_dir
        roberta_model_name = args.roberta_model_name
        roberta_vocab_name = args.roberta_vocab_name
        self.dict_file = "{}/{}".format(roberta_model_dir, roberta_vocab_name)
        self.roberta_model = RobertaModel.from_pretrained(
            roberta_model_dir, checkpoint_file=roberta_model_name
        )
        self.bpe = self.roberta_model.bpe
        self.task = self.roberta_model.task
        self._build_vocab()
        self._init_inverse_vocab()
        self.max_sentence_length = args.max_sentence_length

    @property
    def model(self):
        return self.roberta_model

    @property
    def mask_token(self):
        return ROBERTA_MASK

    def tokenize(self, text: str):
        masked_text = text.replace(MASK, ROBERTA_MASK)
        text_spans = masked_text.split(ROBERTA_MASK)
        text_spans_bpe = ' {0} '.format(ROBERTA_MASK).join(
            [self.bpe.encode(text_span.rstrip())for text_span in text_spans]).strip()
        text_spans_bpe = ROBERTA_START_SENTENCE + ' ' + text_spans_bpe
        return tokenize_line(text_spans_bpe)

    def _cuda(self):
        self.model.cuda()

    def _build_vocab(self):
        self.vocab = []
        for key in range(ROBERTA_VOCAB_SIZE):
            predicted_token_bpe = self.task.source_dictionary.string([key])
            try:
                value = self.bpe.decode(predicted_token_bpe)

                if value[0] == " ":  # if the token starts with a whitespace
                    value = value.strip()
                else:
                    # this is subword information
                    value = "_{}_".format(value)

                if value in self.vocab:
                    # print("WARNING: token '{}' is already in the vocab".format(value))
                    value = "{}_{}".format(value, key)

                self.vocab.append(value)

            except Exception as e:
                self.vocab.append(predicted_token_bpe.strip())

    def get_id(self, input_string):
        # Roberta predicts ' London' and not 'London'
        string = " " + str(input_string).strip()
        text_spans_bpe = self.bpe.encode(string.rstrip())
        tokens = self.task.source_dictionary.encode_line(text_spans_bpe, append_eos=False, add_if_not_exist=False)
        return tokens.long().cpu().numpy().tolist()

    def get_batch_generation(self, sentences_list, logger=None, try_cuda=True, relation_mask=None):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tensor_list = []
        relation_mask_tensor_list = []
        masked_indices_list = []
        max_len = 0
        output_tokens_list = []
        for sid, masked_inputs_list in enumerate(sentences_list):

            tokens_list = []
            rel_mask_list = []

            for idx, masked_input in enumerate(masked_inputs_list):
                if relation_mask is None:
                    # 2. sobstitute [MASK] with <mask>
                    masked_input = masked_input.replace(MASK, ROBERTA_MASK)
                    text_spans = masked_input.split(ROBERTA_MASK)
                    text_spans_bpe = ' {0} '.format(ROBERTA_MASK).join([self.bpe.encode(text_span.rstrip()) for text_span in text_spans]).strip()
                    prefix = ''
                    if idx == 0:
                        prefix = ROBERTA_START_SENTENCE
                    text_spans_bpe = prefix + ' ' + text_spans_bpe
                    tokens_list.append(self.task.source_dictionary.encode_line(text_spans_bpe, append_eos=True, add_if_not_exist=False))
                else:
                    tokens_list.append(self.task.source_dictionary.encode_line(' '.join(masked_input), append_eos=True, add_if_not_exist=False))
                    rel_mask_list.append(torch.tensor(relation_mask[sid][idx] + [0]).long())  # eos

            tokens = torch.cat(tokens_list)[: self.max_sentence_length]
            output_tokens_list.append(tokens.long().cpu().numpy())
            if len(tokens) > max_len:
                max_len = len(tokens)
            tensor_list.append(tokens)
            masked_index = (tokens == self.task.mask_idx).nonzero().numpy()
            for x in masked_index:
                masked_indices_list.append([x[0]])

            if relation_mask is not None:
                relation_mask_tensor_list.append(torch.cat(rel_mask_list)[: self.max_sentence_length])

        pad_id = self.task.source_dictionary.pad()
        tokens_list = []
        for tokens in tensor_list:
            pad_lenght = max_len - len(tokens)
            if pad_lenght > 0:
                pad_tensor = torch.full([pad_lenght], pad_id, dtype=torch.int)
                tokens = torch.cat((tokens, pad_tensor))
            tokens_list.append(tokens)

        raw_tokens = torch.stack(tokens_list).long()
        if relation_mask is not None:
            rel_mask_tensor = torch.nn.utils.rnn.pad_sequence(relation_mask_tensor_list, batch_first=True, padding_value=0)

        if relation_mask is not None:
            batch_tokens = raw_tokens * rel_mask_tensor
        else:
            batch_tokens = raw_tokens

        with torch.no_grad():
            # with utils.eval(self.model.model):
            self.model.eval()
            self.model.model.eval()
            log_probs, extra = self.model.model(
                batch_tokens.to(device=self._model_device),
                features_only=False,
                return_all_hiddens=False,
            )

        mask_tensor = (batch_tokens == self.task.mask_idx).long() if relation_mask is None else rel_mask_tensor

        return log_probs.cpu(), output_tokens_list, masked_indices_list, raw_tokens, mask_tensor

    def get_contextual_embeddings(self, sentences_list, try_cuda=True):
        # TBA
        return None
