import torch
import torch.nn as nn
import torch.nn.functional as F


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
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, targets, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean() if size_average else losses.sum()


class HardTripletLoss(nn.Module):
    """
    From "In Defense of the Triplet Loss"
    For each anchor takes the hardest positive and negative pair
    """

    def __init__(self, margin):
        super(HardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, targets, size_average=True):
        ALMOST_INF = 9999.9

        anchor_pos_dists = torch.cdist(anchor, positive)
        anchor_neg_dists = torch.cdist(anchor, negative)
        
        pos_mask, neg_mask = self.create_masks(targets)

        max_pos,_ = torch.max(anchor_pos_dists*pos_mask, dim=1)
        min_neg,_ = torch.min(anchor_neg_dists + ALMOST_INF*pos_mask, dim=1) 

        losses = F.relu(max_pos - min_neg + self.margin)                
        return losses.mean() if size_average else losses.sum()

    def create_masks(self,targets):
        mask = targets.repeat(targets.size()[0])
        mask = mask.view((targets.size()[0], targets.size()[0]))
        pos_mask = (targets == mask.t())
        neg_mask = (targets != mask.t())

        return pos_mask, neg_mask


if __name__ == "__main__":
    criterion = HardTripletLoss(margin=1.0)

    anchor = torch.rand((6,2))
    pos = torch.rand((6,2))
    neg = torch.rand((6,2))
    target = torch.tensor([1,1,2,2,3,3])

    loss = criterion(anchor, pos, neg, target)
    print(loss)