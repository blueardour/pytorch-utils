import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# label smooth
class CrossEntropyLabelSmooth(nn.Module):
  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


# KD
# https://github.com/peterliht/knowledge-distillation-pytorch/blob/ef06124d67a98abcb3a5bc9c81f7d0f1f016a7ef/model/net.py#L100
def loss_fn_kd(outputs, labels, teacher_outputs, params):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = params.distill_loss_alpha
    if alpha < 1.0:
        loss = F.cross_entropy(outputs, labels) * (1. - alpha)
    else:
        loss = 0

    T = getattr(params, 'distill_loss_temperature', 1.)
    if hasattr(params, 'distill_loss_type'):
        if params.distill_loss_type == 'hard':
            distillation_loss = F.cross_entropy(outputs, teacher_outputs.argmax(dim=1))

        elif params.distill_loss_type == 'soft-b':
            predict = F.log_softmax(outputs/T, dim=1)
            targets = F.softmax(teacher_outputs/T, dim=1)
            predict = predict.unsqueeze(2)
            targets = targets.unsqueeze(1)
            distillation_loss = -torch.bmm(targets, predict)
            distillation_loss = distillation_loss.mean()

        elif params.distill_loss_type == 'soft':
            predict = F.log_softmax(outputs/T, dim=1)
            targets = F.softmax(teacher_outputs/T, dim=1)
            distillation_loss = nn.KLDivLoss()(predict, targets) * (T * T)

        else:
            raise RuntimeError("non-known distill_loss_type: {}".format(params.distill_loss_type))
    else:
        raise RuntimeError("non-known distill_loss_type")

    #if hasattr(params, 'keyword') and 'debug' in params.keyword:
    #    logging.info('loss: cross entropy ({}), KD loss ({})'.format(loss.item(), distillation_loss.item()))
    loss = loss + distillation_loss * alpha
    return loss


class FocalLoss(nn.Module):
    def __init__(self, use_sigmoid=True, alpha=-1., gamma=2., reduction='none'):
        super(FocalLoss, self).__init__()
        self.use_sigmoid = use_sigmoid
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        num_classes = inputs.size(1)
        alpha = self.alpha
        gamma = self.gamma
        reduction = self.reduction
        targets = F.one_hot(targets, num_classes=num_classes)
        #targets = F.one_hot(targets, num_classes=num_classes + 1)
        #targets = targets[:, :num_classes]

        inputs = inputs.float()
        targets = targets.float()
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

