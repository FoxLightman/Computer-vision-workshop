import torch
import numpy as np
from torchvision.transforms import functional as TF
from itertools import product
from scipy import ndimage as ndi

def infer_patch(model, patch_np: np.ndarray, device, thr: float = 0.5):
    """
    patch_np: HxW or CxHxW (grayscale expected).
    returns: prob (HxW float), pred (HxW uint8)
    """
    model.eval()

    x = TF.to_tensor(patch_np)

    # shape -> [1, C, H, W]
    if x.ndim == 2:
        x = x.unsqueeze(0)          # [1, H, W]
    x = x.unsqueeze(0)          # [1, C, H, W]

    x = x.to(device).float()

    with torch.no_grad():
        logits = model(x)           # [1, 1, H, W]
        prob = torch.sigmoid(logits)[0, 0]    # [H, W]
        pred = (prob > thr).to(torch.uint8)   # [H, W]

    return prob.cpu().numpy(), pred.cpu().numpy()

def compute_starts(length: int, patch: int, step: int):
    """
    Starts at 0, advances by `step`, and forces the last start to be (length - patch)
    so both borders are covered and every patch has constant size.
    """
    if patch > length:
        return np.array([0], dtype=int)  # caller must handle padding/cropping if needed

    starts = np.arange(0, length - patch + 1, step, dtype=int)
    last = length - patch
    if starts.size == 0:
        starts = np.array([0], dtype=int)
    if starts[-1] != last:
        starts = np.append(starts, last)
    return np.unique(starts)


@torch.no_grad()
def infer_full_image_tiled_silent(
    model,
    img_np: np.ndarray,
    patch_size: int,
    step: int,
    device=None,
    thr: float = 0.5,
):
    """
    Binary segmentation tiling inference.
    - img_np: HxW (grayscale) or CxHxW (C=1/3)
    - patch_size: e.g., 512
    - step: e.g., 256 for 50% overlap
    - normalize_fn: optional callable patch_np -> patch_np (float32), must match training
    Returns:
      prob_map: HxW float32 (0..1)
      pred_map: HxW uint8 (0/1)
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # Ensure channel-first if present
    if img_np.ndim == 3 and img_np.shape[0] not in (1, 3) and img_np.shape[-1] in (1, 3):
        img_np = np.moveaxis(img_np, -1, 0)  # HWC -> CHW

    if img_np.ndim == 2:
        H, W = img_np.shape
        C = 1
    elif img_np.ndim == 3:
        C, H, W = img_np.shape
        if C not in (1, 3):
            raise ValueError(f"Unexpected channel count: {C}")
    else:
        raise ValueError(f"Unexpected img_np shape: {img_np.shape}")

    # Accumulators in full-resolution coordinates
    prob_acc = np.zeros((H, W), dtype=np.float32)
    w_acc = np.zeros((H, W), dtype=np.float32)

    ys = compute_starts(H, patch_size, step)
    xs = compute_starts(W, patch_size, step)

    for y0 in ys:
        for x0 in xs:
            if img_np.ndim == 2:
                patch = img_np[y0:y0 + patch_size, x0:x0 + patch_size]
            else:
                patch = img_np[:, y0:y0 + patch_size, x0:x0 + patch_size]
            
            # To torch: [1, C, H, W]
            x = TF.to_tensor(patch)
            if x.ndim == 2:
                x = x.unsqueeze(0)          # [1, H, W]
            x = x.unsqueeze(0)            # [1, C, H, W]
            x = x.to(device).float()
            
            with torch.no_grad():
                logits = model(x)           # [1, 1, H, W]
                prob = torch.sigmoid(logits)[0, 0].cpu().numpy()    # [H, W]

            prob_acc[y0:y0 + patch_size, x0:x0 + patch_size] += prob
            w_acc[y0:y0 + patch_size, x0:x0 + patch_size] += 1.0
            
    prob_map = prob_acc / np.maximum(w_acc, 1e-8)
    pred_map = (prob_map > thr).astype(np.uint8)
    return prob_map, pred_map

@torch.no_grad()
def infer_full_image_tiled(
    model,
    img_np: np.ndarray,
    patch_size: int,
    step: int,
    device=None,
    thr: float = 0.5,
    progress=None,
):
    """
    Run tiled patch-based inference on a full image for binary segmentation.

    This function splits an input image into overlapping patches, runs the model
    on each patch independently, and reconstructs the full-resolution probability
    map by averaging predictions in overlapping regions. It is useful for inference
    on images that are too large to process in a single forward pass.

    Parameters
    ----------
    model : torch.nn.Module
        Trained segmentation model. It is used in evaluation mode.

    img_np : np.ndarray
        Input image as either:
        - H x W for grayscale input, or
        - C x H x W for channel-first input with C in {1, 3}, or
        - H x W x C for channel-last input with C in {1, 3}; it will be converted
        internally to channel-first format.

    patch_size : int
        Size of square patches extracted from the image.

    step : int
        Sliding-window stride between neighboring patches. Smaller values increase
        overlap and usually make stitching smoother, but also increase inference time.

    device : torch.device or str, optional
        Device used for inference. If None, the device is inferred from the model
        parameters.

    thr : float, default=0.5
        Threshold applied to the reconstructed probability map to obtain the final
        binary prediction map.

    progress : callable, optional
        Optional progress-bar wrapper such as tqdm. It must accept an iterable and
        optional keyword arguments like total=... and desc=.... If None, inference
        runs silently without a progress bar.

    Returns
    -------
    prob_map : np.ndarray
        Full-resolution float32 probability map of shape H x W with values in [0, 1].

    pred_map : np.ndarray
        Full-resolution uint8 binary prediction map of shape H x W with values 0 or 1.

    Notes
    -----
    - The function assumes binary segmentation and applies torch.sigmoid to model
    outputs.
    - Overlapping patch predictions are merged by simple averaging.
    - The input preprocessing should match the one used during training.
    - The model output is expected to have shape [B, 1, H, W].

    Example
    -------
    >>> from tqdm.auto import tqdm
    >>> prob_map, pred_map = infer_full_image_tiled(
    ...     model,
    ...     img_np,
    ...     patch_size=512,
    ...     step=256,
    ...     progress=tqdm,
    ... )
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    if progress is None:
        progress = lambda iterable, **kwargs: iterable

    # Ensure channel-first if present
    if img_np.ndim == 3 and img_np.shape[0] not in (1, 3) and img_np.shape[-1] in (1, 3):
        img_np = np.moveaxis(img_np, -1, 0)  # HWC -> CHW

    if img_np.ndim == 2:
        H, W = img_np.shape
        C = 1
    elif img_np.ndim == 3:
        C, H, W = img_np.shape
        if C not in (1, 3):
            raise ValueError(f"Unexpected channel count: {C}")
    else:
        raise ValueError(f"Unexpected img_np shape: {img_np.shape}")

    # Accumulators in full-resolution coordinates
    prob_acc = np.zeros((H, W), dtype=np.float32)
    w_acc = np.zeros((H, W), dtype=np.float32)

    ys = compute_starts(H, patch_size, step)
    xs = compute_starts(W, patch_size, step)

    coords = list(product(ys, xs))
    total = len(ys) * len(xs)

    for y0, x0 in progress(coords, total=total):
        if img_np.ndim == 2:
            patch = img_np[y0:y0 + patch_size, x0:x0 + patch_size]
        else:
            patch = img_np[:, y0:y0 + patch_size, x0:x0 + patch_size]
            
        # To torch: [1, C, H, W]
        x = TF.to_tensor(patch)
        if x.ndim == 2:
            x = x.unsqueeze(0)          # [1, H, W]
        x = x.unsqueeze(0)            # [1, C, H, W]
        x = x.to(device).float()
            
        with torch.no_grad():
            logits = model(x)           # [1, 1, H, W]
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()    # [H, W]

        prob_acc[y0:y0 + patch_size, x0:x0 + patch_size] += prob
        w_acc[y0:y0 + patch_size, x0:x0 + patch_size] += 1.0
            
    prob_map = prob_acc / np.maximum(w_acc, 1e-8)
    pred_map = (prob_map > thr).astype(np.uint8)
    return prob_map, pred_map


### Filter low confidence hallucinations on the masks

def filter_mask_by_confident_overlap(
    prediction_map: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    connectivity: int = 2,
    return_intermediate: bool = False,
):
    """
    Filter low-confidence hallucinations in a prediction/confidence map by
    keeping only those low-threshold connected components that overlap at least
    one pixel of the high-threshold mask.

    This function is designed for segmentation post-processing when the model
    outputs a 2D confidence map (for example probabilities in the range [0, 1])
    and you want to suppress weak isolated false positives while preserving
    uncertain regions that are attached to a very confident core.

    The logic is the following:

    1. Build a "candidate" mask using a lower threshold.
       These are all pixels that are considered potentially real.

    2. Build a "confident" mask using a higher threshold.
       These are pixels that are considered highly reliable.

    3. Split the low-threshold mask into connected components.

    4. For each connected component of the low-threshold mask, test whether it
       has at least one overlapping pixel with the high-threshold mask.

    5. Keep the whole component if the overlap is nonzero. Remove it otherwise.

    This method is often more appropriate than directly thresholding the map
    with a single value because it allows uncertain borders of real objects to
    survive as long as they are connected to a very confident core.

    Parameters
    ----------
    prediction_map : numpy.ndarray
        A 2D array containing the confidence, probability, or score for each
        pixel. In the most common use case, values are floats in the interval
        [0, 1], but the function also works for any numeric map as long as the
        thresholds are chosen consistently.

    low_threshold : float
        Threshold used to construct the low-confidence candidate mask:
            low_mask = prediction_map >= low_threshold

        Pixels above this threshold are considered possible object pixels.
        This threshold should usually be permissive enough to include the full
        object shape, including uncertain boundaries.

    high_threshold : float
        Threshold used to construct the high-confidence anchor mask:
            high_mask = prediction_map >= high_threshold

        Pixels above this threshold are considered strongly reliable object
        pixels. A low-threshold component is kept only if it overlaps this
        high-threshold mask.

        In most practical uses, high_threshold should be greater than or equal
        to low_threshold.

    connectivity : int, optional
        Connectivity rule used to define which neighboring pixels belong to the
        same connected component. This parameter is passed to
        scipy.ndimage.generate_binary_structure(rank=2, connectivity=...).

        For 2D images, the most common choices are:

        - connectivity=1 :
            4-connectivity. Pixels are connected only through direct horizontal
            and vertical neighbors (up, down, left, right). Diagonal contact
            does not count as connected.

        - connectivity=2 :
            8-connectivity. Pixels are connected through horizontal, vertical,
            and diagonal neighbors. This is often more natural for segmentation
            masks containing slanted, curved, or thin structures.

        The default is 2.

    return_intermediate : bool, optional
        If False (default), return only the final filtered mask.

        If True, also return a dictionary of intermediate arrays that can be
        useful for debugging, visualization, or understanding the pipeline:
            - "low_mask"
            - "high_mask"
            - "labels"
            - "num_components"

    Returns
    -------
    filtered_mask : numpy.ndarray of bool
        Boolean mask of the same shape as prediction_map. It contains only the
        low-threshold connected components that overlap the high-threshold
        mask.

    info : dict, optional
        Returned only if return_intermediate=True. Contains:
            low_mask : bool array
                Mask obtained by thresholding with low_threshold.
            high_mask : bool array
                Mask obtained by thresholding with high_threshold.
            labels : int array
                Connected-component label image for low_mask. Background is 0.
                Components are numbered from 1 to num_components.
            num_components : int
                Total number of connected components found in low_mask.

    Raises
    ------
    ValueError
        If prediction_map is not a 2D array.
        If high_threshold is smaller than low_threshold.
        If connectivity is not valid for a 2D structure.
        If thresholds are not numeric.

    Notes
    -----
    Why this method is useful
    -------------------------
    A direct threshold at a medium value often leaves many small hallucinated
    blobs. A direct threshold at a high value often removes real object edges.
    This method combines both ideas:

    - the low threshold preserves the full candidate shapes
    - the high threshold provides reliable anchors
    - the overlap test removes isolated weak components

    Why overlap instead of IoU
    --------------------------
    In many cases, simple nonzero overlap is enough and is easier to interpret.
    If a low-threshold component touches a confident core, it is kept. If it
    does not touch any confident core, it is removed.

    IoU can also be used, but it is often unnecessarily strict here because the
    high-threshold region is usually much smaller than the low-threshold
    component.

    Important limitation
    --------------------
    This method assumes that real objects usually contain at least a small
    high-confidence region. If a true object never reaches high_threshold
    anywhere, that object will be removed entirely.

    Examples
    --------
    Basic usage:

    >>> filtered = filter_mask_by_confident_overlap(
    ...     prediction_map=pred,
    ...     low_threshold=0.35,
    ...     high_threshold=0.75,
    ...     connectivity=2,
    ... )

    Get intermediate results for visualization:

    >>> filtered, info = filter_mask_by_confident_overlap(
    ...     prediction_map=pred,
    ...     low_threshold=0.35,
    ...     high_threshold=0.75,
    ...     connectivity=2,
    ...     return_intermediate=True,
    ... )
    >>> low_mask = info["low_mask"]
    >>> high_mask = info["high_mask"]
    >>> labels = info["labels"]

    Visual interpretation
    ---------------------
    If a weak blob exists alone in the image and does not contain any highly
    confident pixels, it will be removed.

    If a real object has a strong center and weaker edges, the whole object
    will be preserved, because its low-threshold component overlaps the
    high-threshold core.

    See Also
    --------
    scipy.ndimage.label
        Labels connected components in a binary image.
    scipy.ndimage.generate_binary_structure
        Generates the neighborhood structure that defines connectivity.
    """
    if prediction_map.ndim != 2:
        raise ValueError("prediction_map must be a 2D array.")

    if not np.issubdtype(type(low_threshold), np.number):
        raise ValueError("low_threshold must be numeric.")

    if not np.issubdtype(type(high_threshold), np.number):
        raise ValueError("high_threshold must be numeric.")

    if high_threshold < low_threshold:
        raise ValueError("high_threshold must be greater than or equal to low_threshold.")

    if connectivity not in (1, 2):
        raise ValueError("For a 2D image, connectivity should usually be 1 or 2.")

    low_mask = prediction_map >= low_threshold
    high_mask = prediction_map >= high_threshold

    structure = ndi.generate_binary_structure(rank=2, connectivity=connectivity)
    labels, num_components = ndi.label(low_mask, structure=structure)

    filtered_mask = np.zeros_like(low_mask, dtype=bool)

    for component_id in range(1, num_components + 1):
        component_mask = labels == component_id

        # Keep this component only if it intersects the high-confidence mask.
        if np.any(component_mask & high_mask):
            filtered_mask |= component_mask

    if return_intermediate:
        info = {
            "low_mask": low_mask,
            "high_mask": high_mask,
            "labels": labels,
            "num_components": num_components,
        }
        return filtered_mask, info

    return filtered_mask