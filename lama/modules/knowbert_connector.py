from typing import Dict, List, Tuple
from kb.include_all import ModelArchiveFromParams
from kb.knowbert_utils import KnowBertBatchifier
from allennlp.common import Params
from allennlp.data import Instance, DataIterator, Vocabulary
import numpy as np
import torch
from lama.modules.base_connector import Base_Connector
import torch.nn.functional as F


class KnowBert(Base_Connector):
    def __init__(self, args, vocab_subset=None):
        super().__init__()

        if args.bert_model_dir is None:
            raise ValueError('require bert_model_dir')

        # load batcher
        self.batcher = KnowBertBatchifier(args.bert_model_dir,
                                          batch_size=args.batch_size,
                                          masking_strategy='full_mask')
        self.tokenizer = self.batcher.tokenizer_and_candidate_generator

        # init vocab
        self.vocab = list(self.tokenizer.bert_tokenizer.ids_to_tokens.values())
        self._init_inverse_vocab()

        # Load pre-trained model (weights)
        params = Params({'archive_file': args.bert_model_dir})
        self.knowbert = ModelArchiveFromParams.from_params(params=params)
        self.knowbert.eval()

        self.mask_ind = self.tokenizer.bert_tokenizer.vocab['[MASK]']

    @property
    def model(self):
        return self.knowbert

    def tokenize(self, text: str):
        word_pieces = self.tokenizer._tokenize_text(text)[1]
        return [p for w in word_pieces for p in w]

    def get_id(self, text: str):
        toks = self.tokenize(text)
        inds = [self.tokenizer.bert_tokenizer.vocab[t] for t in toks]
        return inds

    def _cuda(self):
        self.knowbert.cuda()

    def to_device_recursive(self, data: Dict):
        if type(data) is not dict:
            return data.to(self._model_device)
        return dict((k, self.to_device_recursive(v)) for k, v in data.items())

    def get_batch_generation(self, sentences_list, entity_list, logger=None, try_cuda=True, relation_mask=None):
        if not sentences_list:
            return None
        if try_cuda:
            self.try_cuda()

        with torch.no_grad():
            for batch in self.batcher.iter_batches([s[0] for s in sentences_list], verbose=False):
                # SHAPE: (batch_size, timesteps)
                tokens_tensor = batch['tokens']['tokens']
                model_output = self.knowbert(**self.to_device_recursive(batch))
                # SHAPE: (batch_size, timesteps, vocab_size)
                logits, _ = self.knowbert.pretraining_heads(
                    model_output['contextual_embeddings'], model_output['pooled_output']
                )
                log_probs = F.log_softmax(logits, dim=-1).cpu()

        assert log_probs.size(0) == len(sentences_list), 'multiple batch'
        ind = np.arange(log_probs.size(1))
        masked_indices_li = [ind[mask == 1].tolist() for mask in tokens_tensor.eq(self.mask_ind).numpy()]
        return log_probs, tokens_tensor.numpy(), masked_indices_li, tokens_tensor, None


def knowbert_fill(sentences, model, batcher, vocab, mask_ind=0):
    for batch in batcher.iter_batches(sentences, verbose=False):
        print(batch['tokens']['tokens'])
        model_output = model(**batch)
        print([vocab[w] for w in batch['tokens']['tokens'][0].numpy()])
        logits, _ = model.pretraining_heads(model_output['contextual_embeddings'], model_output['pooled_output'])
        log_probs = F.log_softmax(logits, dim=-1).cpu()
        topk = torch.topk(log_probs[0, mask_ind], 5, 0)[1]
        print([vocab[t.item()] for t in topk])


def knowbert_fill2(sentences, model, batcher, vocab, mask_start=0, mask_end=0, config_file=None, top=10):
    iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 32}))
    config = Params.from_file(config_file)
    vocab_params = config['vocabulary']
    iterator.index_with(Vocabulary.from_params(vocab_params))
    instances = []
    for sent in sentences:
        token_candidates = batcher.tokenizer_and_candidate_generator.tokenize_and_generate_candidates(sent.replace('[MASK]', ' [MASK] '))
        masked_tokens = token_candidates['tokens'].copy()
        for i in range(mask_start, mask_end):
            masked_tokens[i] = '[MASK]'
        token_candidates['tokens'] = masked_tokens

        # mask out the entity candidates
        candidates = token_candidates['candidates']
        for candidate_key in candidates.keys():
            indices_to_mask = []
            for k, candidate_span in enumerate(candidates[candidate_key]['candidate_spans']):
                if (candidate_span[0] >= mask_start and candidate_span[0] <= mask_end-1) or (candidate_span[1] >= mask_start and candidate_span[1] <= mask_end-1):
                    indices_to_mask.append(k)
            for ind in indices_to_mask:
                candidates[candidate_key]['candidate_entities'][ind] = ['@@MASK@@']
                candidates[candidate_key]['candidate_entity_priors'][ind] = [1.0]
            if len(indices_to_mask) == 0:
                candidates[candidate_key]['candidate_spans'].append([mask_start, mask_end-1])
                candidates[candidate_key]['candidate_entities'].append(['@@MASK@@'])
                candidates[candidate_key]['candidate_entity_priors'].append([1.0])
                candidates[candidate_key]['candidate_segment_ids'].append(0)
        fields = batcher.tokenizer_and_candidate_generator.convert_tokens_candidates_to_fields(token_candidates)
        instances.append(Instance(fields))
    for batch in iterator(instances, num_epochs=1, shuffle=False):
        print(batch['tokens']['tokens'])
        model_output = model(**batch)
        print([vocab[w] for w in batch['tokens']['tokens'][0].numpy()])
        logits, _ = model.pretraining_heads(model_output['contextual_embeddings'], model_output['pooled_output'])
        log_probs = F.log_softmax(logits, dim=-1).cpu()
        for mask_ind in range(mask_start, mask_end):
            topk = torch.topk(log_probs[0, mask_ind], top, -1)[1]
            print([vocab[t.item()] for t in topk])
