import time
import torch
from dataclasses import dataclass

### define dice from logits for binary classification and for multiclasses

def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-7):
    """
    dice_iou_from_logits(logits, targets, thr=0.5, eps=1e-7)

    Compute Dice and IoU metrics for BINARY segmentation from raw logits.

    This function is intended for logging/monitoring, not for backpropagation.
    It converts logits -> probabilities with sigmoid, thresholds to a binary mask,
    and then computes overlap metrics.

    Parameters
    ----------
    logits : torch.Tensor
        Model outputs (raw logits), shape [B, 1, H, W] (or compatible).
    targets : torch.Tensor
        Ground truth mask, shape [B, 1, H, W] or [B, H, W], values in {0,1}.
    thr : float
        Probability threshold applied to sigmoid(logits) to obtain binary predictions.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    dice : float
        Mean Dice coefficient over the batch.
    iou : float
        Mean Intersection-over-Union over the batch.

    Notes
    -----
    - For heavily imbalanced data, thr=0.5 is often not optimal; tune threshold on validation.
    - For multiclass exclusive segmentation, use dice_iou_from_logits_mc instead.
    """
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()

    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()

    # per-batch aggregates
    inter = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds * targets).sum(dim=(1,2,3))
    dice = (2 * inter + eps) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps)
    iou  = (inter + eps) / (union + eps)

    return dice.mean().item(), iou.mean().item()

@torch.no_grad()
def dice_iou_from_logits_mc(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int | None = None,
    exclude_background: bool = True,
    eps: float = 1e-7
):
    """
    dice_iou_from_logits_mc(logits, targets, num_classes, ignore_index=None,
                            exclude_background=True, eps=1e-7)

    Compute Dice and IoU for EXCLUSIVE MULTICLASS segmentation from logits.

    This version uses argmax over class logits to obtain predicted class IDs.
    Dice/IoU are computed per class by comparing (pred==c) vs (target==c),
    then averaged (macro average) across classes.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits, shape [B, C, H, W].
    targets : torch.Tensor
        Class index map, shape [B, H, W], dtype torch.long, values 0..C-1.
    num_classes : int
        Number of classes C.
    ignore_index : int or None
        If set, pixels with target==ignore_index are excluded from metric computation.
    exclude_background : bool
        If True, metrics are averaged over classes 1..C-1 (skips class 0).
    eps : float
        Numerical stability constant.

    Returns
    -------
    dice_mean : float
        Macro-averaged Dice (across included classes).
    iou_mean : float
        Macro-averaged IoU (across included classes).

    Notes
    -----
    - For deployment reporting, you may also want per-class metrics rather than only macro average.
    - If your dataset contains many empty-GT patches, consider reporting metrics separately on non-empty GT.
    """
    preds = logits.argmax(dim=1)  # [B,H,W]

    if ignore_index is not None:
        valid = (targets != ignore_index)
    else:
        valid = torch.ones_like(targets, dtype=torch.bool)

    class_ids = range(1, num_classes) if exclude_background else range(0, num_classes)

    dice_list = []
    iou_list = []

    for c in class_ids:
        pred_c = (preds == c) & valid
        targ_c = (targets == c) & valid

        inter = (pred_c & targ_c).sum(dim=(1, 2)).float()
        pred_sum = pred_c.sum(dim=(1, 2)).float()
        targ_sum = targ_c.sum(dim=(1, 2)).float()
        union = pred_sum + targ_sum - inter

        dice = (2 * inter + eps) / (pred_sum + targ_sum + eps)
        iou  = (inter + eps) / (union + eps)

        dice_list.append(dice.mean())
        iou_list.append(iou.mean())

    if len(dice_list) == 0:  # num_classes==1 edge case
        return 0.0, 0.0

    dice_mean = torch.stack(dice_list).mean().item()
    iou_mean  = torch.stack(iou_list).mean().item()
    return dice_mean, iou_mean



@dataclass
class EpochStats:
    loss: float
    dice: float
    iou: float

def _batch_to_tensor(images, targets, device):
    """
    images, targets come from your collate_fn => tuples of length B.
    Returns:
      x: [B, C, H, W] float32
      y: [B, 1, H, W] float32 (binary)
    """
    # If your dataset already returns torch tensors, stack directly.
    # If it returns numpy arrays, convert to torch first.
    if not torch.is_tensor(images[0]):
        images = [torch.as_tensor(im) for im in images]
    if not torch.is_tensor(targets[0]):
        targets = [torch.as_tensor(tg) for tg in targets]

    x = torch.stack(images, dim=0).to(device)  # [B, C, H, W] or [B, H, W]
    y = torch.stack(targets, dim=0).to(device)

    # Ensure channel dimension for image
    if x.ndim == 3:          # [B, H, W]
        x = x.unsqueeze(1)   # [B, 1, H, W]

    # Ensure channel dimension for target
    if y.ndim == 3:          # [B, H, W]
        y = y.unsqueeze(1)   # [B, 1, H, W]

    x = x.float()
    y = y.float()

    return x, y


def _batch_to_tensor_mc(images, targets, device):
    """
    images: tuple of tensors, each [C,H,W] or [H,W]
    targets: tuple of tensors, each [H,W] (class indices) or [1,H,W]
    Returns:
      x: [B, Cin, H, W] float32
      y: [B, H, W] long
    """
    # If your dataset already returns torch tensors, stack directly.
    # If it returns numpy arrays, convert to torch first.
    if not torch.is_tensor(images[0]):
        images = [torch.as_tensor(im) for im in images]
    if not torch.is_tensor(targets[0]):
        targets = [torch.as_tensor(tg) for tg in targets]
    
    x = torch.stack(images, dim=0).to(device)
    y = torch.stack(targets, dim=0).to(device)

    # image: enforce channel dim
    if x.ndim == 3:          # [B,H,W]
        x = x.unsqueeze(1)   # [B,1,H,W]
    x = x.float()

    # target: enforce [B,H,W] long with class indices
    if y.ndim == 4 and y.shape[1] == 1:  # [B,1,H,W] -> [B,H,W]
        y = y[:, 0]
    y = y.long()

    return x, y

### Make evaluate system

@torch.no_grad()
def evaluate(
    model, data_loader, device,
    loss_fn=None,          # <- pass your loss callable(logits, y) -> scalar tensor
    log_every: int = 50,
    thr: float = 0.5       # metric threshold (binary case)
):
    model.eval()

    if loss_fn is None:
        criterion = torch.nn.BCEWithLogitsLoss()
        loss_fn = lambda logits, y: criterion(logits, y)

    loss_sum, dice_sum, iou_sum, n_batches = 0.0, 0.0, 0.0, 0
    t0 = time.time()

    for step, (images, targets) in enumerate(data_loader, start=1):
        x, y = _batch_to_tensor(images, targets, device)

        logits = model(x)
        loss = loss_fn(logits, y)

        dice, iou = dice_iou_from_logits(logits, y, thr=thr)

        loss_sum += loss.item()
        dice_sum += dice
        iou_sum += iou
        n_batches += 1

        if (step % log_every) == 0:
            dt = time.time() - t0
            it_s = step / dt if dt > 0 else float("inf")
            print(f"[eval ] step {step:5d}/{len(data_loader)} | "
                  f"loss {loss_sum/n_batches:.4f} | dice {dice_sum/n_batches:.4f} | "
                  f"iou {iou_sum/n_batches:.4f} | {it_s:.2f} it/s")

    return EpochStats(loss_sum / n_batches, dice_sum / n_batches, iou_sum / n_batches)


### Make train one epoch code
def train_one_epoch(
    model, data_loader, optimizer, device,
    amp: bool = True,
    loss_fn=None,                 # <- pass your loss here: callable(logits, y) -> scalar tensor
    log_every: int = 50,
    thr: float = 0.5              # metric threshold (binary case)
):
    model.train()

    use_amp = amp and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if loss_fn is None:
        # default: plain BCEWithLogitsLoss
        criterion = torch.nn.BCEWithLogitsLoss()
        loss_fn = lambda logits, y: criterion(logits, y)

    loss_sum, dice_sum, iou_sum, n_batches = 0.0, 0.0, 0.0, 0
    win_loss, win_dice, win_iou, win_n = 0.0, 0.0, 0.0, 0

    log_hist = {"step": [], "loss": [], "dice": [], "iou": []}
    t0 = time.time()

    for step, (images, targets) in enumerate(data_loader, start=1):
        x, y = _batch_to_tensor(images, targets, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda", enabled=True):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        dice, iou = dice_iou_from_logits(logits.detach(), y.detach(), thr=thr)

        loss_sum += loss.item()
        dice_sum += dice
        iou_sum += iou
        n_batches += 1

        win_loss += loss.item()
        win_dice += dice
        win_iou  += iou
        win_n += 1

        if (step % log_every) == 0:
            dt = time.time() - t0
            it_s = step / dt if dt > 0 else float("inf")

            avg_loss = win_loss / win_n
            avg_dice = win_dice / win_n
            avg_iou  = win_iou  / win_n

            log_hist["step"].append(step)
            log_hist["loss"].append(avg_loss)
            log_hist["dice"].append(avg_dice)
            log_hist["iou"].append(avg_iou)

            print(f"[train] step {step:5d}/{len(data_loader)} | "
                  f"loss {avg_loss:.4f} | dice {avg_dice:.4f} | iou {avg_iou:.4f} | "
                  f"{it_s:.2f} it/s")

            win_loss, win_dice, win_iou, win_n = 0.0, 0.0, 0.0, 0

    if win_n > 0:
        log_hist["step"].append(step)
        log_hist["loss"].append(win_loss / win_n)
        log_hist["dice"].append(win_dice / win_n)
        log_hist["iou"].append(win_iou / win_n)

    epoch_stats = EpochStats(loss_sum / n_batches, dice_sum / n_batches, iou_sum / n_batches)
    return epoch_stats, log_hist