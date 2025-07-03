import streamlit as st
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tempfile
import warnings
warnings.filterwarnings("ignore")

from predictor_utils import *
import fate_sam_predict

tmp_output_folder = "predictions"
st.title("FATE SAM INFERENCE")

st.sidebar.header("Upload Data")

query_image_file = st.sidebar.file_uploader("Upload Query Image (.nii.gz)", type=["nii.gz"])
query_label_file = st.sidebar.file_uploader("Upload Query Label (.nii.gz, optional)", type=["nii.gz"])
support_images_files = st.sidebar.file_uploader("Upload Support Images (.nii.gz, multiple files)", type=["nii.gz"], accept_multiple_files=True)
support_labels_files = st.sidebar.file_uploader("Upload Support Labels (.nii.gz, multiple files)", type=["nii.gz"], accept_multiple_files=True)
num_classes_input = st.sidebar.text_input("Number of Classes (include background)", value="1")

try:
    num_classes = int(num_classes_input)
    if num_classes <= 0:
        st.sidebar.error("Please enter a positive integer for the number of classes.")
        num_classes = None
except ValueError:
    st.sidebar.error("Please enter a valid integer for the number of classes.")
    num_classes = None

temp_dir = tempfile.TemporaryDirectory()
temp_query_images_dir = os.path.join(temp_dir.name, "query_images")
temp_query_labels_dir = os.path.join(temp_dir.name, "query_labels")
temp_support_images_dir = os.path.join(temp_dir.name, "support_images")
temp_support_labels_dir = os.path.join(temp_dir.name, "support_labels")

os.makedirs(temp_query_images_dir, exist_ok=True)
os.makedirs(temp_query_labels_dir, exist_ok=True)
os.makedirs(temp_support_images_dir, exist_ok=True)
os.makedirs(temp_support_labels_dir, exist_ok=True)

if query_label_file is not None:
    temp_query_label_path = os.path.join(temp_query_labels_dir, query_label_file.name)
    with open(temp_query_label_path, "wb") as f:
        f.write(query_label_file.read())
else:
    temp_query_label_path = None

if support_labels_files:
    for file in support_labels_files:
        file_path = os.path.join(temp_support_labels_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

def save_nifti_to_jpg_folder(nifti_path, output_folder):
    img = nib.load(nifti_path)
    data = img.get_fdata()
    for i in range(data.shape[2]):
        slice_img = data[:, :, i]
        slice_img_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img))
        plt.imsave(os.path.join(output_folder, f"{i:03d}.jpg"), slice_img_norm, cmap="gray")

if query_image_file is not None:
    temp_query_nii_path = os.path.join(temp_query_images_dir, query_image_file.name)
    with open(temp_query_nii_path, "wb") as f:
        f.write(query_image_file.read())

    volume_name = query_image_file.name.replace(".nii.gz", "")
    volume_output_folder = os.path.join(temp_query_images_dir, volume_name)
    os.makedirs(volume_output_folder, exist_ok=True)

    save_nifti_to_jpg_folder(temp_query_nii_path, volume_output_folder)

if support_images_files:
    for file in support_images_files:
        temp_support_nii_path = os.path.join(temp_support_images_dir, file.name)
        with open(temp_support_nii_path, "wb") as f:
            f.write(file.read())

        volume_name = file.name.replace(".nii.gz", "")
        volume_output_folder = os.path.join(temp_support_images_dir, volume_name)
        os.makedirs(volume_output_folder, exist_ok=True)

        save_nifti_to_jpg_folder(temp_support_nii_path, volume_output_folder)

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "volume_idx" not in st.session_state:
    st.session_state.volume_idx = None

if st.sidebar.button("Predict"):
    if (query_image_file is None or not support_images_files or not support_labels_files):
        st.error("Please upload all required files: Query Image, Support Images, Support Labels.")
    elif num_classes is None:
        st.error("Please enter a valid number of classes.")
    else:
        st.info("Running prediction...")

        volume_idx = query_image_file.name.replace(".nii.gz", "")
        volume_folder = os.path.join(temp_query_images_dir, volume_idx)

        df, seg_predictions = fate_sam_predict.run_fate_sam_prediction(
            query_image_path=volume_folder,
            query_label_path=temp_query_label_path,
            support_images_path=temp_support_images_dir,
            support_labels_path=temp_support_labels_dir,
            num_classes=num_classes
        )

        st.success("Prediction completed!")

        overlay_and_save(
            video_segments=seg_predictions,
            output_folder=tmp_output_folder,
            volume_idx=volume_idx,
            reference_img=nib.load(temp_query_nii_path),
        )

        st.session_state.prediction_done = True
        st.session_state.volume_idx = volume_idx

if st.session_state.prediction_done:
    if st.sidebar.button("Reset Viewer"):
        st.session_state.prediction_done = False
        st.session_state.volume_idx = None

if st.session_state.prediction_done and st.session_state.volume_idx is not None:
    st.header("Slice-by-Slice Overlay Viewer")

    @st.cache_resource
    def load_image_files(frame_folder):
        return sorted([
            os.path.join(frame_folder, f)
            for f in os.listdir(frame_folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

    frame_folder = os.path.join(tmp_output_folder, f'segmentation_{st.session_state.volume_idx}')
    image_files = load_image_files(frame_folder)
    num_slices = len(image_files)

    if num_slices == 0:
        st.error("No overlay images found!")
    else:
        slice_idx = st.slider(
            label="",
            min_value=0,
            max_value=num_slices - 1,
            value=num_slices // 2 - 1,
            step=1
        )

        show_query_label = False
        if temp_query_label_path is not None:
            show_query_label = st.toggle("Show Query Label", value=False)

        image_path = image_files[slice_idx]
        overlay_img = mpimg.imread(image_path)

        if show_query_label:
            query_label_nifti = nib.load(temp_query_label_path)
            query_label_data = query_label_nifti.get_fdata()

            if slice_idx < query_label_data.shape[2]:
                label_slice = query_label_data[:, :, slice_idx]
                label_slice_norm = label_slice / (np.max(label_slice) + 1e-5)

                fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                axs[0].imshow(overlay_img)
                axs[0].set_title("Overlay Prediction")
                axs[0].axis("off")

                axs[1].imshow(label_slice_norm, cmap="gray")
                axs[1].set_title("Query Label")
                axs[1].axis("off")

                st.pyplot(fig)
            else:
                st.error("Selected slice is out of range for the query label!")
        else:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(overlay_img)
            ax.set_title("Overlay Prediction")
            ax.axis("off")
            st.pyplot(fig)