import torch
import torch.nn as nn
import torch.nn.functional as F

def _apply_margin(x, m):
    if isinstance(m, float):
        return (x + m).clamp(min=0)

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean(), (losses > 0).sum().item()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, targets):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        
        return losses.mean(), (losses > 0).sum().item()


class HardTripletLoss(nn.Module):
    """
    From "In Defense of the Triplet Loss"
    For each anchor takes the hardest positive and negative pair
    """

    def __init__(self, margin):
        super(HardTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, targets):
        ALMOST_INF = 9999.9
        anchor_dists = torch.cdist(anchor, anchor)
        mask_pos = (targets[None, :] == targets[:, None]).float()

        furthest_positive = torch.max(anchor_dists * mask_pos, dim=0)[0]
        furthest_negative = torch.min(anchor_dists + ALMOST_INF*mask_pos, dim=0)[0]

        losses = F.relu(furthest_positive - furthest_negative + self.margin)
        
        return losses.mean(), (losses > 0).sum().item()


if __name__ == "__main__":
    criterion = HardTripletLoss(margin=1.0)

    anchor = torch.rand((6,2))
    target = torch.tensor([1,1,2,2,3,3])

    loss, active_triplets = criterion(anchor, target)
    print(pos)