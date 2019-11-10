from typing import Dict
import torch
import torch.nn as nn


class TempModel(nn.Module):
    def __init__(self, rel2numtemp: Dict[str, int], enforce_prob: bool=True):
        super(TempModel, self).__init__()
        self.rel2numtemp = rel2numtemp
        self.enforce_prob = enforce_prob
        for rel, numtemp in rel2numtemp.items():
            setattr(self, rel, nn.Parameter(torch.zeros(numtemp)))

    def set_weight(self, relation: str, new_weight: torch.Tensor):
        weight = getattr(self, relation)
        weight[:] = new_weight

    def forward(self,
                relation: str,
                features: torch.FloatTensor,  # SHAPE: (batch_size, num_temp, vocab_size)
                target: torch.LongTensor=None  # SHAPE: (batch_size,)
                ):
        # SHAPE: (num_temp)
        weight = getattr(self, relation)
        if self.enforce_prob:
            weight = weight.exp()
            weight = weight / weight.sum()
        if len(features.size()) == 3:
            # SHAPE: (batch_size, vocab_size)
            features = (features * weight.view(1, -1, 1)).sum(1).exp()
            if target is not None:
                #loss = nn.CrossEntropyLoss(reduction='mean')(features, target)
                # SHAPE: (batch_size,)
                loss = torch.gather(features, dim=1, index=target.view(-1, 1))
                loss = -loss.mean()
                return loss
        elif len(features.size()) == 2:
            # SHAPE: (batch_size,)
            features = (features * weight.view(1, -1)).sum(1)
            loss = -features.log().mean()
            return loss
        else:
            raise NotImplementedError
        return features



