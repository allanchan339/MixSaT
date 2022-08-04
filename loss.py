import torch
import torch.nn.functional as F


class NLLLoss(torch.nn.Module):
    def __init__(self):
        super(NLLLoss, self).__init__()

    def forward(self, output, labels):
        # return torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output))
        return torch.mean(torch.mean(labels * -torch.log(output) + (1 - labels) * -torch.log(1 - output)))


class SigmoidFocalLoss(torch.nn.Module):
    """
        Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range(0, 1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor(1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2, reduction: str = "mean",) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # p = torch.sigmoid(inputs)
        p = inputs  # as inputs passed sigmoid
        
        # ce_loss = F.binary_cross_entropy_with_logits(
        #     inputs, targets, reduction="none")
        
        ce_loss = F.binary_cross_entropy( # this program use 3 sigmoid???  
            inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

class BinaryFocalLoss(torch.nn.Module):
    def __init__(self, gamma = 2, *, pos_weight = None, from_logits=False, label_smoothing=None, reduction="mean", **kwargs) -> None:
        super().__init__()
        self.gamma = gamma 
        self.pos_weight = pos_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        p = inputs
        q = 1 - p
        eps=torch.tensor(1e-10)

        # avoid take log from 0
        p = torch.maximum(p, eps)
        q = torch.maximum(q, eps)

        pos_loss = -(q ** self.gamma) * torch.log(p)
        if self.pos_weight is not None:
            pos_loss *= self.pos_weight
        
        neg_loss = -(p ** self.gamma) * torch.log(q)

        labels = targets.bool()
        loss = torch.where(labels, pos_loss, neg_loss)

        if self.reduction == "mean":
            return torch.mean(loss)

        elif self.reduction == "sum":
            return torch.sum(loss)
        
        else:
            return loss
