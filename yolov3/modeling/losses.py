
from torch import  nn
import torch

__all__ = ["CELossNoSoftmax"]

class CELossNoSoftmax(nn.Module):
    """
    loss = - log(x[class]), class in target(class range), [0, 1, ...]
    """
    def __init__(self, reduction="mean"):
        super(CELossNoSoftmax, self).__init__()
        self.cross_entropy_loss = nn.NLLLoss(reduction="none")
        self.reduction = reduction
    def forward(self, x, target):
        loss = self.cross_entropy_loss(-x, target)
        loss= -torch.log(loss)
        if self.reduction == "mean" or self.reduction is "none":
            return loss.mean()
        elif self.reduction == "sum":
            return  loss.sum()
        else:
            print("not support reduction format:{}".format(self.reduction))
            return

class YoloV3RegLoss(nn.Module):
    def __init__(self, redution="mean"):
        super(YoloV3RegLoss, self).__init__()
        self.bce_loss = nn.BCELoss(reduction=redution)
    def forward(self, x, target):
        assert x.shape == target.shape, "x, target shape must equal: {}!={} !!".format(x.shape, target.shape)
        return self.bce_loss(x, target)

if __name__ == "__main__":
    criterion = CELossNoSoftmax(reduction="mean")

    x = torch.Tensor([[0.9, 0.1, 0.0], [0.6, 0.3, 0.1]])

    target = torch.Tensor([0, 1]).to(torch.long)

    loss = criterion(x, target)

    print(loss.item())