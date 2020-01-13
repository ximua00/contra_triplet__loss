import torch
import torch.nn as nn
import torch.nn.functional as F

def _apply_margin(x, m):
    if isinstance(m, float):
        return (x + m).clamp(min=0)
    elif m.lower() == "soft":
        return F.softplus(x)
    elif m.lower() == "none":
        return x
    else:
        raise NotImplementedError("The margin %s is not implemented in BatchHard!" % m)

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


class BatchHard(nn.Module):
    """
    From "In Defense of the Triplet Loss"
    For each anchor takes the hardest positive and negative pair
    """

    def __init__(self, margin):
        super(BatchHard, self).__init__()
        self.margin = margin

    def forward(self, anchor, targets):
        ALMOST_INF = 9999.9

        anchor_dists = torch.cdist(anchor, anchor)
        mask_pos = (targets[None, :] == targets[:, None]).float()

        furthest_positive = torch.max(anchor_dists * mask_pos, dim=1)[0]
        furthest_negative = torch.min(anchor_dists + ALMOST_INF*mask_pos, dim=1)[0]

        losses = F.relu(furthest_positive - furthest_negative + self.margin)
        return losses.mean(), (losses > 0).sum().item()
    

def batch_soft(anchor, pids, margin, T=1.0):
    """Calculates the batch soft.
    Instead of picking the hardest example through argmax or argmin,
    a softmax (softmin) is used to sample and use less difficult examples as well.
    Args:
        cdist (2D Tensor): All-to-all distance matrix, sized (B,B).
        pids (1D tensor): PIDs (classes) of the identities, sized (B,).
        margin: The margin to use, can be 'soft', 'none', or a number.
        T (float): The temperature of the softmax operation.
    """
    # mask where all positives are set to true
    mask_pos = pids[None, :] == pids[:, None]
    mask_neg = ~mask_pos.data

    anchor_dists = torch.cdist(anchor, anchor)

    # only one copy
    cdist_max = anchor_dists.clone()
    cdist_max[mask_neg] = -float('inf')
    cdist_min = anchor_dists.clone()
    cdist_min[mask_pos] = float('inf')

    print(cdist_min)
    
    # NOTE: We could even take multiple ones by increasing num_samples,
    #       the following `gather` call does the right thing!
    idx_pos = torch.multinomial(F.softmax(cdist_max/T, dim=1), num_samples=1)
    idx_neg = torch.multinomial(F.softmin(cdist_min/T, dim=1), num_samples=2)
    print(idx_neg)

    positive = anchor_dists.gather(dim=1, index=idx_pos)[:,0]  # Drop the extra (samples) dim
    negative = anchor_dists.gather(dim=1, index=idx_neg) [:,1]

    print(negative)
    
    losses = _apply_margin(positive - negative, margin)
    return losses.mean(), (losses > 0).sum().item()

class BatchSoft(nn.Module):
    """BatchSoft implementation using softmax.
    
    Also by Tristani as Adaptivei Weighted Triplet Loss.
    If T is close to 0, them this is similar to Batch Hard
    """

    def __init__(self, margin, T=1.0, **kwargs):
        """
        Args:
            m: margin
            T: Softmax temperature
        """
        super(BatchSoft, self).__init__()
        self.name = "BatchSoft(m={}, T={})".format(margin, T)
        self.margin = margin
        self.T = T

    def forward(self, anchor, pids):
        return batch_soft(anchor, pids, self.margin, self.T)


if __name__ == "__main__":
    criterion = BatchSoft(margin=1.0, T=1.0)

    anchor = torch.rand((6,2))
    target = torch.tensor([1,1,2,2,3,3])

    loss, active_samples = criterion(anchor, target)