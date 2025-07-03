import os
import nibabel as nib
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sam2.utils.misc import load_video_frames
from utils import device_setup

class ImageLabelLoader:
    def __init__(self, images_path, labels_path, image_size=1024):
        self.images_path = images_path
        self.labels_path = labels_path
        self.image_size = image_size

    def load_data(self):
        """
        Loads all images and corresponding labels from the given folder structure.
        """
        data = []
        for folder_name in os.listdir(self.images_path):
            folder_path = os.path.join(self.images_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            compute_device = device_setup()
            image, video_height, video_width = load_video_frames(
                video_path=folder_path,
                image_size=self.image_size,
                offload_video_to_cpu=True,
                async_loading_frames=False,
                compute_device=compute_device,
            )
            label_file = f"{folder_name}.nii.gz"
            label_path = os.path.join(self.labels_path, label_file)
            if os.path.exists(label_path):
                label = nib.load(label_path).get_fdata()
                label = np.transpose(label, (2, 0, 1))
            else:
                raise FileNotFoundError(f"Label file {label_file} not found in {self.labels_path}")

            # Append to data list
            data.append({
                'image': image,
                'image_folder': folder_path,
                'label': label,
                'volume_idx': int(re.search(r"(\d+)\.nii\.gz$", label_file).group(1))
            })

        return data

    def get_train_test_split(self, data):
        query_data, sup_data = train_test_split(data, test_size=0.10, random_state=2)
        return query_data, sup_data