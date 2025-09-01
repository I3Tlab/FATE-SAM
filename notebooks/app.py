import re
import os
import tempfile
import warnings

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import nibabel as nib
import numpy as np
import streamlit as st

warnings.filterwarnings("ignore")

from predictor_utils import *
import fate_sam_predict


# === App Setup ===
st.title("FATE SAM INFERENCE")
st.sidebar.header("Upload Data")

st.session_state.setdefault("prediction_done", False)
st.session_state.setdefault("volume_idx", None)
st.session_state.setdefault("seg_predictions", None)
st.session_state.setdefault("df", None)
st.session_state.setdefault("current_image_path", None)
st.session_state.setdefault("save_toggle_prev", False)
st.session_state.setdefault("num_classes_input", "")

# === Temp Directories ===
tmp_output_folder = "predictions"
temp_dir = tempfile.TemporaryDirectory()
temp_root = temp_dir.name
temp_test_imgs_dir = os.path.join(temp_root, "test_imgs")
temp_test_labels_dir = os.path.join(temp_root, "test_labels")
temp_support_imgs_dir = os.path.join(temp_root, "support_imgs")
temp_support_labels_dir = os.path.join(temp_root, "support_labels")

os.makedirs(temp_test_imgs_dir, exist_ok=True)
os.makedirs(temp_test_labels_dir, exist_ok=True)
os.makedirs(temp_support_imgs_dir, exist_ok=True)
os.makedirs(temp_support_labels_dir, exist_ok=True)


# === Helpers ===
FNAME_PATTERN = re.compile(
    r"^(?P<stem>.+?)_(?P<idx>\d+?)_(?P<kind>img|label)\.nii\.gz$",
    re.IGNORECASE,
)

def parse_pairs(uploaded_files):
    """
    Parse and pair files named as xxxx_####_(img|label).nii.gz.
    Returns dict: {id_key: {'image': UploadedFile|None, 'label': UploadedFile|None}}
    where id_key = f"{stem}_{idx}"
    """
    pairs = {}
    unmatched = []
    for uf in uploaded_files or []:
        name = os.path.basename(uf.name)
        m = FNAME_PATTERN.match(name)
        if not m:
            unmatched.append(name)
            continue
        stem = m.group("stem")
        idx = m.group("idx")
        kind = m.group("kind").lower()
        key = f"{stem}_{idx}"
        if key not in pairs:
            pairs[key] = {"image": None, "label": None}
        kind_key = "image" if kind == "img" else "label"
        pairs[key][kind_key] = uf
    return pairs, unmatched


def normalize_and_save(uf, dst_dir):
    """
    Save uploaded file into dst_dir with a normalized name:
    Returns the full destination path.
    """
    name = os.path.basename(uf.name)
    if name.lower().endswith("_img.nii.gz"):
        base = name[:-len("_img.nii.gz")]
    elif name.lower().endswith("_label.nii.gz"):
        base = name[:-len("_label.nii.gz")]
    else:
        # fallback: strip .nii.gz if present
        base = name[:-len(".nii.gz")] if name.lower().endswith(".nii.gz") else name
    norm_name = f"{base}.nii.gz"
    dst_path = os.path.join(dst_dir, norm_name)
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "wb") as f:
        f.write(uf.read())
    return dst_path


def save_nifti_to_jpg_folder(nifti_path, output_folder):
    """
    Convert a 3D NIfTI volume into per-slice JPGs (z-axis).
    """
    img = nib.load(nifti_path)
    data = img.get_fdata()
    os.makedirs(output_folder, exist_ok=True)
    for i in range(data.shape[2]):
        slice_img = data[:, :, i]
        # robust normalization
        mn, mx = float(np.min(slice_img)), float(np.max(slice_img))
        denom = (mx - mn) if (mx - mn) != 0 else 1.0
        slice_img_norm = (slice_img - mn) / (denom + 1e-5)
        plt.imsave(os.path.join(output_folder, f"{i:03d}.jpg"), slice_img_norm, cmap="gray")


def load_nifti_data(nifti_path):
    img = nib.load(nifti_path)
    return img.get_fdata()


def plot_slices(data1, data2=None, slice_idx=None, title1="Image", title2="Label"):
    if slice_idx is None:
        slice_idx = data1.shape[2] // 2
    img1 = data1[:, :, slice_idx]
    mn1, mx1 = float(np.min(img1)), float(np.max(img1))
    img1 = (img1 - mn1) / ((mx1 - mn1) + 1e-5)

    if data2 is not None:
        img2 = data2[:, :, slice_idx]
        mn2, mx2 = float(np.min(img2)), float(np.max(img2))
        img2 = (img2 - mn2) / ((mx2 - mn2) + 1e-5)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(img1, cmap="gray"); axs[0].set_title(title1); axs[0].axis("off")
        axs[1].imshow(img2, cmap="gray"); axs[1].set_title(title2); axs[1].axis("off")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(img1, cmap="gray"); ax.set_title(title1); ax.axis("off")
    st.pyplot(fig)


# === Upload ===
test_mix = st.sidebar.file_uploader(
    "Upload TEST files",
    type=["nii.gz"],
    accept_multiple_files=True,
    key="test_mix",
)
support_mix = st.sidebar.file_uploader(
    "Upload SUPPORT files",
    type=["nii.gz"],
    accept_multiple_files=True,
    key="support_mix",
)

test_pairs, test_unmatched = parse_pairs(test_mix)
support_pairs, support_unmatched = parse_pairs(support_mix)

if test_unmatched:
    st.sidebar.warning(f"[TEST] Ignored files (name pattern mismatch): {', '.join(test_unmatched)}")
if support_unmatched:
    st.sidebar.warning(f"[SUPPORT] Ignored files (name pattern mismatch): {', '.join(support_unmatched)}")

# === Save TEST ===
saved_test_items = []
for key, pair in test_pairs.items():
    img_path = None
    lbl_path = None
    jpg_folder = None

    if pair["image"] is not None:
        img_path = normalize_and_save(pair["image"], temp_test_imgs_dir)
        jpg_folder = os.path.join(temp_test_imgs_dir, key)
        save_nifti_to_jpg_folder(img_path, jpg_folder)

    if pair["label"] is not None:
        lbl_path = normalize_and_save(pair["label"], temp_test_labels_dir)

    saved_test_items.append(
        {"key": key, "image_path": img_path, "label_path": lbl_path, "jpg_folder": jpg_folder}
    )

# === Save SUPPORT ===
saved_support_items = [] 
for key, pair in support_pairs.items():
    img_path = None
    lbl_path = None
    jpg_folder = None

    if pair["image"] is not None:
        img_path = normalize_and_save(pair["image"], temp_support_imgs_dir)
        jpg_folder = os.path.join(temp_support_imgs_dir, key)
        save_nifti_to_jpg_folder(img_path, jpg_folder)

    if pair["label"] is not None:
        lbl_path = normalize_and_save(pair["label"], temp_support_labels_dir)

    saved_support_items.append(
        {"key": key, "image_path": img_path, "label_path": lbl_path, "jpg_folder": jpg_folder}
    )


# === Number of Classes ===
def _autodetect_num_classes_from_support():
    label_files = [f for f in os.listdir(temp_support_labels_dir) if f.lower().endswith(".nii.gz")]
    if not label_files:
        return None
    sample_path = os.path.join(temp_support_labels_dir, label_files[0])
    try:
        data = nib.load(sample_path).get_fdata()
        unique_vals = np.unique(np.round(data).astype(np.int64))
        return int(len(unique_vals))
    except Exception:
        return None

if st.session_state["num_classes_input"] == "":
    _auto = _autodetect_num_classes_from_support()
    if _auto is not None:
        st.session_state["num_classes_input"] = str(_auto)
st.sidebar.text_input(
    "[OPTIONAL] Number of Classes (include background)",
    key="num_classes_input",
    placeholder="auto-detects from support label",
)
num_classes = None
_val = st.session_state["num_classes_input"].strip()
if _val.isdigit() and int(_val) > 0:
    num_classes = int(_val)

# === Visualization ===
available_tests = [it["key"] for it in saved_test_items if it["image_path"] is not None]
selected_test_key = st.sidebar.selectbox("Select TEST volume", options=available_tests) if available_tests else None
current_test = next((it for it in saved_test_items if it["key"] == selected_test_key), None)

if current_test and current_test["image_path"]:
    st.header("Uploaded Data")
    with st.expander("Show Test Image + Label", expanded=False):
        q_data = load_nifti_data(current_test["image_path"])
        l_data = load_nifti_data(current_test["label_path"]) if current_test["label_path"] else None
        slice_idx = st.slider("Choose a slice:", 0, q_data.shape[2] - 1, q_data.shape[2] // 2, 1, key="test_slice")
        plot_slices(q_data, l_data, slice_idx, "Test Image", "Test Label")

support_img_files = [f for f in os.listdir(temp_support_imgs_dir) if f.endswith(".nii.gz")]
support_keys = [os.path.splitext(os.path.splitext(f)[0])[0] for f in support_img_files] 

if support_img_files:
    with st.expander("Show Support Images + Labels", expanded=False):
        selected_support_key = st.selectbox("Select a support volume to view", support_keys, key="support_select")
        if selected_support_key:
            img_path = os.path.join(temp_support_imgs_dir, f"{selected_support_key}.nii.gz")
            img_data = load_nifti_data(img_path)

            label_path = os.path.join(temp_support_labels_dir, f"{selected_support_key}.nii.gz")
            label_data = load_nifti_data(label_path) if os.path.exists(label_path) else None

            slice_idx = st.slider(
                "Choose a slice:",
                0, img_data.shape[2] - 1, img_data.shape[2] // 2, 1,
                key=f"support_slice_{selected_support_key}"
            )
            plot_slices(img_data, label_data, slice_idx, "Support Image", "Support Label")


# === Prediction ===
if st.sidebar.button("Predict"):
    if (current_test is None) or (current_test["jpg_folder"] is None):
        st.error("Please upload and select a valid TEST pair (img[, label]).")
    elif not support_img_files:
        st.error("Please upload SUPPORT files (must include at least one *_img.nii.gz).")
    else:
        st.info("Running prediction...")
        volume_idx = current_test["key"]
        test_jpg_folder = current_test["jpg_folder"]
        test_label_path = current_test["label_path"]

        df, seg_predictions = fate_sam_predict.run_fate_sam_prediction(
            test_image_path=test_jpg_folder, 
            test_label_path=test_label_path, 
            support_images_path=temp_support_imgs_dir,
            support_labels_path=temp_support_labels_dir,
            num_classes=num_classes
        )

        st.session_state.prediction_done = True
        st.session_state.volume_idx = volume_idx
        st.session_state.seg_predictions = seg_predictions
        st.session_state.df = df
        st.session_state.current_image_path = current_test.get("image_path")
        
# === Prediction Visualization ===
if st.session_state.prediction_done and st.session_state.volume_idx is not None:
    st.header("Slice-by-Slice Overlay Viewer")

    with st.container(border=True):
        save_now = st.toggle(
            "💾 Save Prediction",
        )
        if save_now and not st.session_state.save_toggle_prev:
            seg_predictions = st.session_state.get("seg_predictions")
            volume_idx = st.session_state.volume_idx
            ref_img_path = st.session_state.get("current_image_path")

            if seg_predictions is None:
                st.error("No predictions available. Please run prediction first.")
            elif not ref_img_path:
                st.error("Reference image path missing; cannot save.")
            else:
                try:
                    volume_path = save_results(
                        video_segments=seg_predictions,
                        output_folder=tmp_output_folder,
                        volume_idx=volume_idx,
                        reference_img=nib.load(ref_img_path)
                    )
                    st.success(f"Predicted Volume Saved at {volume_path}")
                except Exception as e:
                    st.error(f"Saving failed: {e}")

            st.session_state.save_toggle_prev = True
        elif not save_now and st.session_state.save_toggle_prev:
            st.session_state.save_toggle_prev = False

    @st.cache_resource
    def load_img_files(frame_folder):
        return sorted([
            os.path.join(frame_folder, f)
            for f in os.listdir(frame_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    frame_folder = os.path.join(temp_test_imgs_dir, f"{st.session_state.volume_idx}")
    image_files = load_img_files(frame_folder)
    num_slices = len(image_files)

    if num_slices == 0:
        st.error("No overlay images found!")
    else:
        slice_idx = st.slider(
            label="",
            min_value=0,
            max_value=num_slices - 1,
            value=max(0, num_slices // 2 - 1),
            step=1
        )

        current_test_meta = next((it for it in saved_test_items if it["key"] == st.session_state.volume_idx), None)
        show_test_label = False
        if current_test_meta and current_test_meta["label_path"] and os.path.exists(current_test_meta["label_path"]):
            show_test_label = st.toggle("Show Test Label", value=False)

        image_path = image_files[slice_idx]
        overlay_img = mpimg.imread(image_path)

        if show_test_label:
            test_label_data = nib.load(current_test_meta["label_path"]).get_fdata()
            if slice_idx < test_label_data.shape[2]:
                label_slice = test_label_data[:, :, slice_idx]
                label_slice_norm = label_slice / (np.max(label_slice) + 1e-5)

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                axs[0].imshow(overlay_img); axs[0].set_title("Overlay Prediction"); axs[0].axis("off")
                axs[1].imshow(label_slice_norm, cmap="gray"); axs[1].set_title("Test Label"); axs[1].axis("off")
                st.pyplot(fig)
            else:
                st.error("Selected slice is out of range for the test label!")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(overlay_img)
            ax.set_title("Overlay Prediction")
            ax.axis("off")
            st.pyplot(fig)
