import torch
import torch.nn as nn
import torch.nn.functional as F


def getmask(k):
    ones = torch.ones((k, k))
    mask = ones.fill_diagonal_(0)
    mask = mask.bool()
    return mask


def cont_loss_(anchor, e_pos, h_pos, h_neg):
    cos = nn.CosineSimilarity(dim=-1)
    k, _ = anchor.size()
    e_pos_logits = cos(anchor, e_pos).reshape(k, 1)

    mask = getmask(k)  # mask掉对角线上的元素
    neg_logits = cos(anchor.unsqueeze(1),
                     anchor.unsqueeze(0))[mask].reshape(k, -1)
    hard_neg_sim = cos(anchor, h_neg).unsqueeze(1)
    neg_logits = torch.cat((neg_logits, hard_neg_sim), dim=1)
    h_pos_logits = cos(anchor, h_pos).reshape(k, 1)
    pos_logits = torch.cat((e_pos_logits, h_pos_logits), dim=1)

    logits2 = torch.cat((pos_logits, neg_logits), dim=1)

    pos_label = torch.Tensor([1 for _ in range(pos_logits.size(1))]).to(pos_logits.device).long()
    neg_label = torch.Tensor([0 for _ in range(neg_logits.size(1))]).to(neg_logits.device).long()
    labels = torch.cat([pos_label, neg_label])

    loss2 = -(F.log_softmax(logits2, dim=0) * labels).sum() / labels.sum()

    return loss2/k