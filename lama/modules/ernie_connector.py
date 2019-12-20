from typing import Dict, List, Tuple
import os
from knowledge_bert import BertTokenizer, BertForMaskedLM, BasicTokenizer
from knowledge_bert.tokenization import whitespace_tokenize_ent
import numpy as np
from lama.modules.base_connector import *
import torch.nn.functional as F
import tagme
tagme.GCUBE_TOKEN = '59ebe41e-05bd-48bd-83db-804a7855f104-843339462'


class CustomBasicTokenizer(BasicTokenizer):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def tokenize(self, text, ents):
        """Tokenizes a piece of text."""
        text, drop_idx = self._clean_text(text)
        # update ents
        if len(drop_idx) > 0:
            for i, ent in enumerate(ents):
                cnt = sum([True if j < ent[1] else False for j in drop_idx])
                ent[1] -= cnt
                cnt = sum([True if j < ent[2] else False for j in drop_idx])
                ent[2] -= cnt
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        orig_tokens = whitespace_tokenize_ent(text, ents)
        split_tokens = []
        split_ents = []
        for token, ent in orig_tokens:
            num_toks = -1
            if MASK in token:  # pass MASK forward
                split_tokens.append(MASK)
                num_toks = 1
                if token != MASK:
                    remaining_chars = token.replace(MASK, '').strip()
                    if remaining_chars:
                        split_tokens.append(remaining_chars)
                        num_toks += 1
            else:
                if self.do_lower_case:
                    token = token.lower()
                    token = self._run_strip_accents(token)
                cur = self._run_split_on_punc(token)
                num_toks = len(cur)
                split_tokens.extend(cur)
            split_ents.extend([ent] + ['UNK'] * (num_toks - 1))
        #output_tokens = whitespace_tokenize(" ".join(split_tokens))
        #assert len(output_tokens) == len(split_ents)
        output_tokens = split_tokens
        return zip(output_tokens, split_ents)


class Ernie(Base_Connector):
    def __init__(self, args, vocab_subset=None):
        super().__init__()

        if args.bert_model_dir is None:
            raise ValueError('require bert_model_dir')

        self.dict_file = os.path.join(args.bert_model_dir, args.bert_vocab_name)
        print('loading ERNIE model from {}'.format(args.bert_model_dir))

        # ERNIE is uncased
        do_lower_case = True

        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        cbt = CustomBasicTokenizer(do_lower_case=do_lower_case)
        self.tokenizer.basic_tokenizer = cbt

        # original vocab
        self.map_indices = None
        self.vocab = list(self.tokenizer.ids_to_tokens.values())
        self._init_inverse_vocab()

        # Load pre-trained model (weights)
        self.ernie_model, _ = BertForMaskedLM.from_pretrained(args.bert_model_dir)
        self.ernie_model.eval()

        # Load entity embeddings
        print('loading entity embeddings')
        self.ent_map = {}
        with open(os.path.join(args.kg_path, 'entity_map.txt')) as fin:
            for line in fin:
                name, qid = line.strip().split('\t')
                self.ent_map[name] = qid

        self.entity2id = {}
        with open(os.path.join(args.kg_path, 'entity2id.txt'), 'r') as fin:
            fin.readline()
            for line in fin:
                qid, eid = line.strip().split('\t')
                self.entity2id[qid] = int(eid)
        vecs = np.load(os.path.join(args.kg_path, 'entity2vec.npy'))  # the first element is pad with all zeros
        self.kg_emb = torch.nn.Embedding.from_pretrained(torch.FloatTensor(vecs))

        self.bert_model = self.ernie_model.bert
        self.pad_id = self.inverse_vocab[BERT_PAD]
        self.unk_index = self.inverse_vocab[BERT_UNK]

    @property
    def model(self):
        return self.ernie_model

    def tokenize(self, text: str):
        return self.tokenizer.tokenize(text, [])[0]

    def get_id(self, string):
        toks, _ = self.tokenizer.tokenize(string, [])
        inds = self.tokenizer.convert_tokens_to_ids(toks)
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            inds = self.convert_ids(inds)
        return inds

    def get_ents(self, ann):
        ents = []
        if ann is None:
            return ents
        # Keep annotations with a score higher than 0.3
        for a in ann.get_annotations(0.3):
            if a.entity_title not in self.ent_map:
                continue
            ents.append([self.ent_map[a.entity_title], a.begin, a.end, a.score])
        return ents

    def ent_to_qid(self, ann: List[Tuple[str, int, int, float]]):
        ents = []
        for a in ann:
            if a[0] not in self.ent_map:
                continue
            ents.append([self.ent_map[a[0]], a[1], a[2], a[3]])
        return ents

    def __get_token_ids_from_tensor(self, indexed_string):
        token_ids = []
        if self.map_indices is not None:
            # map indices to subset of the vocabulary
            indexed_string = self.convert_ids(indexed_string)
            token_ids = np.asarray(indexed_string)
        else:
            token_ids = indexed_string
        return token_ids

    def _cuda(self):
        self.ernie_model.cuda()

    def get_batch_generation(self, sentences_list, entity_list, logger=None, try_cuda=True, relation_mask=None):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        tokens_tensor_li, ents_tensor_li, segments_tensor_li, \
        ent_mask_li, masked_indices_li, attn_mask_li = [], [], [], [], [], []
        for sentences, ents_a in zip(sentences_list, entity_list):
            text_a = sentences[0]  # TODO: handle the second sentence
            #text_a_ann = tagme.annotate(text_a)
            #ents_a = self.get_ents(text_a_ann)
            #ents_a = self.ent_to_qid(ents_a)
            tokens_a, entities_a = self.tokenizer.tokenize(text_a, ents_a)

            tokens = ['[CLS]'] + tokens_a + ['[SEP]']
            ents = ['UNK'] + entities_a + ['UNK']

            '''
            for e in ents:
                if e != 'UNK':
                    print('entity detected ' + e)
            '''

            segments_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)
            masked_indices = []
            masked_list = []
            for i in range(len(tokens)):
                token = tokens[i]
                if token == MASK:
                    masked_indices.append(i)
                    masked_list.append(0)
                else:
                    masked_list.append(1)
            masked_indices_li.append(masked_indices)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
            indexed_ents = []
            ent_mask = []
            for ent in ents:
                if ent != 'UNK' and ent in self.entity2id:
                    indexed_ents.append(self.entity2id[ent])
                    ent_mask.append(1)
                else:
                    indexed_ents.append(-1)
                    ent_mask.append(0)
            ent_mask[0] = 1

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor(indexed_tokens)
            ents_tensor = torch.tensor(indexed_ents)
            segments_tensor = torch.tensor(segments_ids)
            ent_mask = torch.tensor(ent_mask)
            attn_mask = torch.tensor(input_mask)

            tokens_tensor_li.append(tokens_tensor)
            ents_tensor_li.append(ents_tensor)
            segments_tensor_li.append(segments_tensor)
            ent_mask_li.append(ent_mask)
            attn_mask_li.append(attn_mask)

        # Padding
        tokens_tensor = torch.nn.utils.rnn.pad_sequence(tokens_tensor_li, batch_first=True, padding_value=0)
        ents_tensor = torch.nn.utils.rnn.pad_sequence(ents_tensor_li, batch_first=True, padding_value=-1)
        segments_tensor = torch.nn.utils.rnn.pad_sequence(segments_tensor_li, batch_first=True, padding_value=0)
        ent_mask = torch.nn.utils.rnn.pad_sequence(ent_mask_li, batch_first=True, padding_value=0)
        attn_mask = torch.nn.utils.rnn.pad_sequence(attn_mask_li, batch_first=True, padding_value=0)

        # Entity embedding
        ents_tensor = self.kg_emb(ents_tensor + 1)

        with torch.no_grad():
            logits = self.ernie_model(tokens_tensor.to(self._model_device),
                                      ents_tensor.to(self._model_device),
                                      ent_mask=ent_mask.to(self._model_device),
                                      token_type_ids=segments_tensor.to(self._model_device),
                                      attention_mask=attn_mask.to(self._model_device))
            log_probs = F.log_softmax(logits, dim=-1).cpu()
        token_ids_list = []
        for indexed_string in tokens_tensor.numpy():
            token_ids_list.append(self.__get_token_ids_from_tensor(indexed_string))
        return log_probs, token_ids_list, masked_indices_li, tokens_tensor, None
