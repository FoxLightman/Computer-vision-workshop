import torch
import torch.nn.functional as F



### Define tversky loss function
def tversky_loss_from_logits(logits, targets, alpha=0.7, beta=0.3, eps=1e-7):
    """
    tversky_loss_from_logits(logits, targets, alpha=0.7, beta=0.3, eps=1e-7)

    Binary Tversky loss from logits (overlap loss generalizing Dice/IoU).

    Tversky index:
    TI = TP / (TP + alpha*FN + beta*FP)

    Loss:
    1 - TI

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits, shape [B, 1, H, W].
    targets : torch.Tensor
        Binary mask, shape [B, 1, H, W] or [B, H, W].
    alpha : float
        Weight for false negatives (FN). Higher alpha increases recall.
    beta : float
        Weight for false positives (FP). Higher beta increases precision.
    eps : float
        Numerical stability constant.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss.

    Notes
    -----
    - Common convention is alpha + beta = 1 (not required, but convenient).
    - In sparse-object segmentation, overly recall-biased settings (alpha>>beta)
    can lead to all-positive collapse, especially if combined with large pos_weight.
    """
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    probs = torch.sigmoid(logits)

    tp = (probs * targets).sum(dim=(1,2,3))
    fp = (probs * (1 - targets)).sum(dim=(1,2,3))
    fn = ((1 - probs) * targets).sum(dim=(1,2,3))

    t = (tp + eps) / (tp + alpha * fn + beta * fp + eps)
    return 1 - t.mean()


def bce_tversky_loss(logits, targets, pos_weight=None, bce_weight=0.5, alpha=0.7, beta=0.3):
    """
    bce_tversky_loss(logits, targets, pos_weight=None, bce_weight=0.5,
                    alpha=0.7, beta=0.3)

    Combined binary loss: BCEWithLogits (optionally weighted) + Tversky.

    Computes:
    loss = bce_weight * BCEWithLogits(logits, targets; pos_weight)
        + (1 - bce_weight) * TverskyLoss(logits, targets; alpha, beta)

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits, shape [B, 1, H, W].
    targets : torch.Tensor
        Binary mask, shape [B, 1, H, W] or [B, H, W].
    pos_weight : torch.Tensor or None
        Positive class weight for BCE. If None, BCE is unweighted.
    bce_weight : float
        Mixing weight. Higher bce_weight makes training more stable and conservative.
    alpha, beta : float
        Tversky parameters controlling FN/FP tradeoff.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss.

    Notes
    -----
    - If the model collapses to all-positive, increase beta or bce_weight and/or reduce pos_weight.
    - If the model collapses to all-negative, increase pos_weight or alpha and/or reduce bce_weight.
    """
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
    tv  = tversky_loss_from_logits(logits, targets, alpha=alpha, beta=beta)
    return bce_weight * bce + (1 - bce_weight) * tv

### focal loss fungtion
def focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction="mean"):
    """
    focal_loss_with_logits(logits, targets, alpha=0.25, gamma=2.0, reduction="mean")

    Binary focal loss computed from logits.

    Focal loss modifies BCE to down-weight easy examples and focus training on hard pixels.
    Typical form:
    FL = alpha_t * (1 - p_t)^gamma * BCE(logits, targets)

    where p_t is the probability of the true class:
    p_t = p if y==1 else (1-p)

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits, shape [B, 1, H, W].
    targets : torch.Tensor
        Binary mask, shape [B, 1, H, W] or [B, H, W], values in {0,1}.
    alpha : float
        Class balancing factor for positives (alpha for y=1, 1-alpha for y=0).
        Increase alpha to emphasize positives (higher recall).
    gamma : float
        Focusing parameter. gamma=0 reduces to (alpha-weighted) BCE.
        Larger gamma down-weights easy pixels more strongly.
    reduction : str
        "mean", "sum", or "none" aggregation over pixels/batch.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss if reduction != "none"; else per-pixel loss map.

    Notes
    -----
    - Focal loss can reduce background dominance without requiring large pos_weight.
    - alpha and gamma interact: higher gamma often requires tuning alpha for sparse positives.
    """
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    pt = p * targets + (1 - p) * (1 - targets)  # prob of the true class

    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - pt).pow(gamma) * bce

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss

### binary cross entropy + dice
def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, dice_weight: float = 0.5):
    """
    bce_dice_loss(logits, targets, dice_weight=0.5)

    Binary segmentation loss: BCEWithLogits + soft Dice.

    Computes:
    loss = (1 - dice_weight) * BCEWithLogits(logits, targets)
        + dice_weight * (1 - soft_dice(sigmoid(logits), targets))

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits, shape [B, 1, H, W].
    targets : torch.Tensor
        Binary mask, shape [B, 1, H, W] or [B, H, W], values in {0,1}.
    dice_weight : float
        Weight of the Dice term. 0 => pure BCE. 1 => pure Dice loss.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss suitable for backprop.

    Notes
    -----
    - Dice term helps with class imbalance but can cause instability (all-positive/all-negative collapse)
    if overweighted. Start with dice_weight ~ 0.1–0.3 for sparse objects.
    """
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets)

    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1,2,3))
    denom = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2 * inter + 1e-7) / (denom + 1e-7)
    dice_loss = 1 - dice.mean()

    return (1 - dice_weight) * bce + dice_weight * dice_loss

### binary cross entropy + dice, bce with positive weight
def bce_dice_loss_weighted(logits, targets, pos_weight: torch.Tensor, dice_weight: float = 0.5):
    """
    bce_dice_loss_weighted(logits, targets, pos_weight, dice_weight=0.5)

    Binary segmentation loss: weighted BCEWithLogits + soft Dice.

    This is the same as bce_dice_loss, but the BCE term uses pos_weight to
    increase the penalty for false negatives (positive class errors):

    BCE = binary_cross_entropy_with_logits(..., pos_weight=pos_weight)

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits, shape [B, 1, H, W].
    targets : torch.Tensor
        Binary mask, shape [B, 1, H, W] or [B, H, W], values in {0,1}.
    pos_weight : torch.Tensor
        Positive class weight, usually shape [1] for single-channel output.
        Should be on the same device as logits.
    dice_weight : float
        Weight of the Dice term.

    Returns
    -------
    loss : torch.Tensor
        Scalar loss.

    Notes
    -----
    - pos_weight is typically set to neg/pos pixel ratio (or a clipped/swept version).
    - If pos_weight is too large, the model can collapse to all-positive predictions.
    """
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1,2,3))
    denom = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    dice = (2 * inter + 1e-7) / (denom + 1e-7)
    dice_loss = 1 - dice.mean()

    return (1 - dice_weight) * bce + dice_weight * dice_loss



