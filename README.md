# Few-Shot Adaptation of Training-Free Foundation Model for 3D Medical Image Segmentation

- [Introduction](#introduction)
- [Getting Started](#getting-started)
<!--  * [Installation](#Installation)
  * [Offline Training](#offline-training)
  * [Online Adaptation](#online-adaptation) -->
- [Publication](#publication)
- [Contacts](#contacts)
<!-- - [Star History](#star-history)-->

## Introduction
Few-shot Adaptation of Training-frEe SAM (FATE-SAM) is a versatile framework for 3D medical image segmentation that adapts the pretrained SAM2 model without fine-tuning. By leveraging a support example-based memory mechanism and volumetric consistency, FATE-SAM enables prompt-free, training-free segmentation across diverse medical datasets. This approach achieves robust performance while eliminating the need for large annotated datasets. For more details and results, please refer to our [[paper]](https://arxiv.org/abs/2501.09138).

![figure1.svg](resources%2Ffigure1.jpg)


## Getting Started
### Installation
0. Download and Install the appropriate version of NVIDIA driver and CUDA for your GPU.
1. Download and install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://docs.anaconda.com/miniconda/).
2. Clone this repo and cd to the project path.
```bash
git clone git@github.com:I3Tlab/FATE-SAM.git
cd FATE-SAM
```
3. Create and activate the Conda environment:
```bash
conda create --name FATE_SAM python=3.10.12
conda activate FATE_SAM
```
4. Install dependencies
```bash
pip install -r requirements.txt
```

### Support Image Generation
1. Image-label examples can but found in `notebooks/data`
2. To generate your own support images: we recommend saving them following the **MSD**  and **SAM2** convention
    <details>
    <summary>Sample dataset directory structure (click to view details)</summary>
    
    ```commandline
    <dataset>/
    ├── imagesQuery/0001.nii.gz
    ├── imagesQuery_jpg/           # 2D slices of query image
    │   └── 0001/
    │       ├── slice_000.jpg
    │       ├── slice_001.jpg
    │       └── ...
    ├── imagesSupport/             # 3D support images (.nii.gz)
    │   ├── 0002.nii.gz
    │   ├── 0003.nii.gz
    │   └── ...
    ├── imagesSupport_jpg/         # 2D slices of support images
    │   ├── 0002/
    │   │   ├── slice_000.jpg
    │   │   └── ...
    │   ├── 0003/
    │   │   ├── slice_000.jpg
    │   │   └── ...
    ├── labelsSupport/             # 3D support labels (.nii.gz)
    │   ├── 0002.nii.gz
    │   └── 0003.nii.gz
    └── 
    ```
    </details>


### Inference with command line
Data format: follow the sample dataset directory structure described above 
```bash
python notebooks/fate_sam_predict.py \
  --query_image_path <path-to-query-image-jpg-folder> \
  --query_label_path <path-to-query-label.nii.gz> \
  --support_images_path <path-to-support-images-jpg-dir> \
  --support_labels_path <path-to-support-labels-dir> \
  --num_classes <number-of-classes>
```

### Inference with Streamlit app
Data format: all .nii.gz (query label is optional)
```batsh
streamlit run notebooks/app.py
```

### Output the segmentation results for further usage
Overlayed segmentation slices are output to the `predictions/` folder

### Publication
```bibtex
@article{he2025few,
  title={Few-Shot Adaptation of Training-Free Foundation Model for 3D Medical Image Segmentation},
  author={He, Xingxin and Hu, Yifan and Zhou, Zhaoye and Jarraya, Mohamed and Liu, Fang},
  journal={arXiv preprint arXiv:2501.09138},
  year={2025}
}
```

### Contacts
[Intelligent Imaging Innovation and Translation Lab](https://liulab.mgh.harvard.edu/) [[github]](https://github.com/I3Tlab) at the Athinoula A. Martinos Center of Massachusetts General Hospital and Harvard Medical School
* Xingxin He (xihe2@mgh.harvard.edu)
* Yifan Hu (yihu2@mgh.harvard.edu)
* Fang Liu (fliu12@mgh.harvard.edu)

149 13th Street, Suite 2301
Charlestown, Massachusetts 02129, USA
