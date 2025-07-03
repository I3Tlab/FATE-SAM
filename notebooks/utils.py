import torch
import numpy as np
import pandas as pd
from scipy import stats
np.random.seed(100)

# --------------------------------------------------------------------------------
# SAM2 PREDICTORS
# --------------------------------------------------------------------------------
def sam2_predictor():
    from sam2.build_sam import build_sam2_video_predictor

    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    device = device_setup()

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    return predictor

def sam2_predictor_fate():
    from sam2.build_sam import build_sam2_video_predictor_fate

    sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    device = device_setup()

    predictor = build_sam2_video_predictor_fate(model_cfg, sam2_checkpoint, device=device)
    return predictor

# --------------------------------------------------------------------------------
# MISC.
# --------------------------------------------------------------------------------
def device_setup():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    return device

def manhattan_distance_per_pixel(target_feats, image_feats):
    distances = []
    for t_feat, i_feat in zip(target_feats, image_feats):
        dist = torch.abs(t_feat - i_feat).sum(dim=-1).mean().item()
        distances.append(dist)
    return sum(distances) / len(distances)

def add_support_image(existing_tensor, similarity_results, compute_device=torch.device("cpu")):
    new_images_tensor = torch.stack([data['image'] for data in similarity_results.values()], dim=0)
    new_images_tensor = new_images_tensor.to(compute_device)
    existing_tensor = existing_tensor.to(compute_device)
    updated_tensor = torch.cat((existing_tensor, new_images_tensor), dim=0)
    return updated_tensor

# --------------------------------------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------------------------------------
def dice_coefficient_label(S, R, label):
    S_label = (S == 1).astype(np.int32)
    R_label = (R == label).astype(np.int32)
    intersection = np.logical_and(S_label, R_label).sum()
    return (2 * intersection) / (S_label.sum() + R_label.sum())

def volume_overlap_error_label(S, R, label):
    S_label = (S == 1).astype(np.int32)
    R_label = (R == label).astype(np.int32)
    intersection = np.logical_and(S_label, R_label).sum()
    union = np.logical_or(S_label, R_label).sum()
    return 1 - (intersection / union)

def volume_difference_label(S, R, label):
    S_label = (S == 1).astype(np.int32)
    R_label = (R == label).astype(np.int32)
    return (S_label.sum() - R_label.sum()) / R_label.sum()

def average_hausdorff(S, R, label):
    from scipy.spatial.distance import directed_hausdorff
    pred_points = np.argwhere(S == 1)
    gt_points = np.argwhere(R == label)
    if pred_points.size == 0 or gt_points.size == 0:
        return float('inf')
    hausdorff_pred_to_gt = directed_hausdorff(pred_points, gt_points)[0]
    hausdorff_gt_to_pred = directed_hausdorff(gt_points, pred_points)[0]
    return min(hausdorff_pred_to_gt, hausdorff_gt_to_pred)

def evaluation(pred_mask, query_label, actual_label):
    eval = {}
    pred_mask = pred_mask.squeeze()
    dice_val = dice_coefficient_label(pred_mask, query_label, actual_label)
    eval[f"DICE {actual_label}"] = dice_val

    voe = volume_overlap_error_label(pred_mask, query_label, actual_label)
    eval[f"VOE {actual_label}"] = voe

    vd = volume_difference_label(pred_mask, query_label, actual_label)
    eval[f"VD {actual_label}"] = vd

    ahd = average_hausdorff(pred_mask, query_label, actual_label)
    eval[f"AHD {actual_label}"] = ahd
    return eval

def mean_and_ci(df):
    results = []

    for col in df.columns:
        valid_values = df[col][np.isfinite(df[col]) & (df[col] != 0)]
        valid_values = valid_values[valid_values != 0]
        mean = valid_values.mean()
        sem = stats.sem(valid_values, nan_policy='omit')
        ci = sem * stats.t.ppf((1 + 0.95) / 2, len(valid_values) - 1)
        results.append({'Column': col, 'Mean': mean, '95% CI Lower': mean - ci, '95% CI Upper': mean + ci})
    return pd.DataFrame(results)



