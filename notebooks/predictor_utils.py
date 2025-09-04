import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pickle
import warnings
from tqdm import tqdm

import nibabel as nib
import imageio
import cv2
from collections import OrderedDict
from scipy.ndimage import zoom
warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
np.random.seed(100)
pd.set_option('display.max_rows', None)

import sys
sys.path.append(".")

from sam2.utils.misc import load_video_frames
from notebooks.utils import *

@torch.inference_mode()
def compute_features(folder_path, images, predictor, pickle_path=None, batch_size=1):
    """
    Compute deep features for each frame in a video using a given predictor.
    Optionally saves the features to a pickle file.
    """
    inference_state = predictor.init_state(folder_path)
    inference_state['images'] = images
    inference_state['num_frames'] = inference_state['images'].size(0)

    features = []
    for idx in range(inference_state['num_frames']):
        _, _, feature, _, _ = predictor._get_image_feature(
            inference_state, idx, batch_size=batch_size
        )
        features.append(feature)

    if pickle_path:
        with open(pickle_path, "wb") as f:
            pickle.dump(features, f)
    return features

def find_top_similar_images_embed(support_images, support_features, support_label_array, query_features, top_n=3):
    """
    Identify the top-N most similar support images for each query feature using Manhattan distance.
    """
    all_results = []
    for i, feat_query in tqdm(enumerate(query_features), desc="Finding Similar Images"):
        similarities = []
        for feat_support in support_features:
            if (
                    len(feat_query) == len(feat_support) and
                    all(tq.shape == ts.shape for tq, ts in zip(feat_query, feat_support))
            ):
                dist = manhattan_distance_per_pixel(feat_query, feat_support)
                similarities.append(dist)
            else:
                similarities.append(float('inf'))

        similarities_tensor = torch.tensor(similarities)

        sorted_indices = torch.argsort(similarities_tensor, descending=False)

        result = {}
        count = 0
        for idx in sorted_indices:
            if (support_label_array[idx] > 0).any():
                result[idx.item()] = {
                    'image': support_images[idx],
                    'label': support_label_array[idx],
                    'score': similarities_tensor[idx].item(),
                }
                count += 1
            if count >= top_n:
                break

        all_results.append(result)
    return all_results

def run_inference_single_volume(image_folder, label, similarity_results, predictor, num_classes=0):
    """
    Run segmentation inference on a single 3D image volume using top-N support images.
    Returns segmentation predictions and evaluation metrics (if labels are provided).
    """
    inference_state = predictor.init_state(image_folder, offload_video_to_cpu=True, offload_state_to_cpu=True)
    image_len = inference_state['num_frames']
    start_frame_idx = image_len // 2

    inference_state['images'] = add_support_image(inference_state['images'], similarity_results[start_frame_idx])
    inference_state['num_frames'] += len(similarity_results[start_frame_idx])
    predictor.reset_state(inference_state)
    mask_added_flag = False

    for idx, (k, s) in enumerate(similarity_results[start_frame_idx].items()):
        actual_labels = range(1, num_classes) if num_classes > 0 else sorted(np.unique(s["label"]))[1:]
        if actual_labels:
            mask_added_flag = True
            for actual_label in actual_labels:
                mask = (s["label"] == actual_label).astype(float)
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=idx + image_len,
                    obj_id=actual_label,
                    mask=mask,
                )

    def _propagate_and_predict(reverse=False, offset=0):
        """
        Helper function to propagate masks through the volume and collect predictions.
        """
        result = []
        seg_predictions = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video_fate(
                inference_state, similarity_results, start_frame_idx=start_frame_idx + offset, reverse=reverse):
            seg_predictions[out_frame_idx] = {
                out_obj_id: (out_mask_logits[x] > 0.0).cpu().numpy() for x, out_obj_id in enumerate(out_obj_ids)
            }

            if label is not None:
                row = {"query_img_idx": out_frame_idx}
                slice_label = label[out_frame_idx]
                for obj_id, pred_mask in seg_predictions[out_frame_idx].items():
                    row.update(evaluation(pred_mask, slice_label, obj_id))
                result.append(row)

        return {"result": result, "seg_predictions": seg_predictions}

    if mask_added_flag:
        out_reverse = _propagate_and_predict(reverse=True)
        out_forward = _propagate_and_predict(reverse=False, offset=1)

        dice_scores = out_reverse['result'] + out_forward['result']
        seg_predictions = {**out_reverse['seg_predictions'], **out_forward['seg_predictions']}
        dice_df = pd.DataFrame(dice_scores).sort_values(by="query_img_idx") if label is not None else None
        return dice_df, seg_predictions

    return None, None

def run_single_image_inference(query_image_path, query_label_path, support_images, support_labels, num_classes=0, support_features=None):
    """
    Full pipeline for running inference on a query image using support images and labels.
    Returns segmentation results and metrics.
    """
    query_image = load_image(query_image_path)

    if query_label_path is not None:
        query_label = load_label(query_label_path)
    else:
        query_label = None

    predictor = sam2_predictor()
    fate_predictor = sam2_predictor_fate()

    if support_features is None:
        support_features = compute_features(
            folder_path=query_image_path,
            images=support_images,
            predictor=predictor,
            pickle_path=None,
            batch_size=1
        )

    query_feature = compute_features(
        folder_path=query_image_path,
        images=query_image,
        predictor=predictor
    )

    similarity_results = find_top_similar_images_embed(
        support_images, support_features, support_labels, query_feature, top_n=3
    )

    dice_df, seg_predictions = run_inference_single_volume(
        image_folder=query_image_path,
        label=query_label,
        similarity_results=similarity_results,
        predictor=fate_predictor,
        num_classes=num_classes
    )

    return dice_df, seg_predictions

def load_image(image_folder):
    """
    Load video frames from a folder and return them as a tensor.
    """
    compute_device = device_setup()
    image_tensor, _, _ = load_video_frames(
        video_path=image_folder,
        image_size=1024,
        offload_video_to_cpu=True,
        async_loading_frames=False,
        compute_device=compute_device
    )
    return image_tensor

def load_label(label_path):
    """
    Load label volume from a NIfTI file and return as (num_frames, H, W).
    """
    label_vol = nib.load(label_path).get_fdata()
    label_vol = np.transpose(label_vol, (2, 0, 1))
    return label_vol

def load_support_data_from_loader(loader):
    """
    Load support images and labels from a data loader.
    """
    sup_data = loader.load_data()
    support_images = [d['image'] for d in sup_data]
    support_images = torch.cat(support_images, dim=0)
    support_labels = [label for d in sup_data for label in d['label']]
    return support_images, support_labels

def nii_to_jpg(nii_file, output_folder):
    """
    Convert a NIfTI file to individual JPG slice images.
    """
    img = nib.load(nii_file)
    img_data = img.get_fdata()
    os.makedirs(output_folder, exist_ok=True)
    num_slices = img_data.shape[2]
    for i in range(num_slices):
        slice_data = img_data[:, :, i]
        min_val = np.min(slice_data)
        max_val = np.max(slice_data)
        if max_val > min_val:
            slice_norm = (slice_data - min_val) / (max_val - min_val)
        else:
            slice_norm = np.zeros_like(slice_data)
        slice_norm = (slice_norm * 255).astype(np.uint8)
        output_path = os.path.join(output_folder, f"slice_{i:03d}.jpg")
        plt.imsave(output_path, slice_norm, cmap='gray')

def save_3d(merged_volume, volume_path, reference_img):
    """
    Save a 3D volume as a NIfTI file using a reference image for affine/header.
    """
    reference_affine = reference_img.affine
    reference_header = reference_img.header
    new_nifti_img = nib.Nifti1Image(merged_volume, reference_affine, header=reference_header)
    nib.save(new_nifti_img, volume_path)

def overlay_and_show(
        video_segments,
        output_folder,
        volume_idx,
        reference_img,
        label_alpha=0.9
):
    original_volume = np.transpose(reference_img.get_fdata(), (2, 0, 1))

    PREDEFINED_COLORS = {
        1: [104, 122, 97, int(255 * label_alpha)],
        2: [173, 166, 139, int(255 * label_alpha)],
        3: [173, 116, 83, int(255 * label_alpha)],
        4: [113, 149, 173, int(255 * label_alpha)],
        5: [130, 144, 118, int(255 * label_alpha)],
        6: [156, 141, 111, int(255 * label_alpha)],
        7: [143, 102, 74, int(255 * label_alpha)],
        8: [90, 130, 155, int(255 * label_alpha)],
        9: [76, 109, 130, int(255 * label_alpha)],
        10: [63, 93, 112, int(255 * label_alpha)],
        11: [150, 80, 80, int(255 * label_alpha)],
        12: [200, 120, 130, int(255 * label_alpha)],
        13: [173, 170, 100, int(255 * label_alpha)],
        14: [120, 90, 130, int(255 * label_alpha)],
        15: [140, 100, 150, int(255 * label_alpha)],
    }

    all_masks = []
    all_frames = []

    for out_frame_idx, masks_dict in sorted(video_segments.items(), key=lambda x :x[0]):
        print(out_frame_idx)
        frame_folder = os.path.join(output_folder, f'segmentation_{volume_idx}')
        os.makedirs(frame_folder, exist_ok=True)

        merged_mask = np.zeros_like(original_volume[0], dtype=np.uint8)
        for out_obj_id, mask in masks_dict.items():
            mask = np.squeeze(mask)
            scale_factors = (
                original_volume.shape[1] / mask.shape[0],
                original_volume.shape[2] / mask.shape[1]
            )
            resized = zoom(mask, scale_factors, order=0)
            merged_mask[resized > 0] = out_obj_id

        original_frame = original_volume[out_frame_idx]
        min_val, max_val = np.min(original_frame), np.max(original_frame)
        if max_val - min_val != 0:
            norm_slice = (
                    (original_frame - min_val) / (max_val - min_val) * 255
            ).astype(np.uint8)
        else:
            norm_slice = np.zeros_like(original_frame, dtype=np.uint8)

        h, w = norm_slice.shape
        background_bgr = cv2.cvtColor(norm_slice, cv2.COLOR_GRAY2BGR)
        background_rgba = np.dstack([background_bgr, np.full((h, w), 255, dtype=np.uint8)])
        label_rgba = np.zeros((h, w, 4), dtype=np.uint8)

        for label_val, color in PREDEFINED_COLORS.items():
            label_mask = (merged_mask == label_val)
            label_rgba[label_mask] = color

        label_alpha_float = label_rgba[..., 3] / 255.0
        label_alpha_float_3c = np.stack([label_alpha_float] * 3, axis=-1)
        out_rgb = (
                label_rgba[..., :3].astype(np.float32) * label_alpha_float_3c
                + background_rgba[..., :3].astype(np.float32) * (1 - label_alpha_float_3c)
        ).astype(np.uint8)
        out_alpha = np.where(label_rgba[..., 3] > 0, label_rgba[..., 3], background_rgba[..., 3])
        rgba_composited = np.dstack([out_rgb, out_alpha])

        save_path = os.path.join(frame_folder, f'{int(out_frame_idx):03d}.png')
        imageio.imwrite(save_path, rgba_composited)
        all_masks.append(merged_mask)
        all_frames.append(rgba_composited)

    return np.stack(all_frames, axis=0), np.stack(all_masks, axis=0)

def save_results(video_segments, output_folder, volume_idx, reference_img):
    """
    Save final segmentation results: overlay images and merged 3D volume in .nii.gz format.
    """
    video_segments = OrderedDict(sorted(video_segments.items()))
    _, merged_volume = overlay_and_show(video_segments, output_folder, volume_idx, reference_img)
    volume_path = os.path.join(output_folder, f'merged_volume_{volume_idx}.nii.gz')
    save_3d(merged_volume, volume_path, reference_img)
    print(f"3D volume saved to {volume_path} as a .nii.gz file.")
    return volume_path
