import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .utils import (
    get_centroids,
    get_cossim,
    calc_loss,
)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, intent_logits, targets):
        intent_loss_fct = nn.CrossEntropyLoss(self.alpha)
        ce_loss = intent_loss_fct(
            intent_logits.view(-1, len(self.alpha)), targets.view(-1)
        )

        pt = torch.exp(-ce_loss)

        loss = ((1 - pt) ** self.gamma * ce_loss).mean()

        return loss



class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        0.5 * (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()



class GE2ELoss(nn.Module):
    
    def __init__(self):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.b = nn.Parameter(torch.tensor(-5.0), requires_grad=True)

    def forward(self, embeddings):
        torch.clamp(self.w, 1e-6)
        centroids = get_centroids(embeddings)
        cossim = get_cossim(embeddings, centroids)
        sim_matrix = self.w*cossim + self.b
        loss, _ = calc_loss(sim_matrix)
        loss = loss / (embeddings.shape[0] * embeddings.shape[1])
        return loss