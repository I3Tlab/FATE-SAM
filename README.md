# Few-Shot Adaptation of Training-Free Foundation Model for 3D Medical Image Segmentation

💡 Tip: Use Ctrl+Click (Windows/Linux), Command+Click (macOS) or right-click to open links in a new tab.

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Publication](#publication)
- [Contacts](#contacts)

## Introduction
Few-shot Adaptation of Training-frEe SAM (FATE-SAM) is a versatile framework for 3D medical image segmentation that adapts the pretrained SAM2 model without fine-tuning. By leveraging a support example-based memory mechanism and volumetric consistency, FATE-SAM enables prompt-free, training-free segmentation across diverse medical datasets. This approach achieves robust performance while eliminating the need for large annotated datasets. For more details and results, please refer to our [[paper]](https://arxiv.org/abs/2501.09138).

![figure1.svg](resources%2Ffigure1.jpg)

## Getting Started
The hardware environment we tested.
- OS: Ubuntu 20.04.6 LTS
- CPU: Intel Core i9-10980XE CPU @ 3.00GHz * 36
- GPU: NVIDIA RTX A5000

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

### Inference with command line
To perform inference with command line, support images and labels are needed. We provided a few support examples which can be found in `notebooks/data` covering different anatomies adopted from following publicly available datasets.

| Dataset | Anatomy | Segmentation Objects | Modality |
|-----|-----|-----| ----- |
| [SKI10](https://huggingface.co/datasets/YongchengYAO/SKI10) | Knee | (1) femur (2) femoral cartilage (3) tibia (4) tibial cartilage | MRI |
| [BTCV](https://www.synapse.org/Synapse:syn3193805/wiki/) | Abdomen | (1) spleen (2) right kidney (3) left kidney (4) gallbladder (5) esophagus (6) liver (7) stomach (8) aorta (9) inferior vena cava (10) portal vein and splenic vein (11) pancreas (12) right adrenal gland (13) left adrenal gland | CT |
| [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html)| Heart  | (1) left ventricle (2) right ventricle (3) myocardium | Cine-MRI |
| [MSD](http://medicaldecathlon.com/)-Hippocampus| Brain | (1) anterior (2) posterior | MRI |
| [MSD](http://medicaldecathlon.com/)-Prostate| Prostate | (1) peripheral zone (2) transition zone | MRI |

Run inference with command line.
```bash
python notebooks/fate_sam_predict.py \
  --test_image_path <path-to-test-image-jpg-folder> \
  --support_images_path <path-to-support-images-jpg-dir> \
  --support_labels_path <path-to-support-labels-dir> \
```

### Inference with Streamlit app
Run inference with GUI.
```batsh
streamlit run notebooks/app.py
```
<!--[ ] insert video for GUI inference here-->

### Support Image Generation

To use your own support images and labels, please save the support images as .nii format. We recommoned using [3D slicer](https://www.slicer.org/) to generat the support label and saving them as following formats and folder structures
 
<details>
  <summary>Sample dataset directory structure (click to view details)</summary>
    
```commandline
<dataset>/
├── Testimages/0001.nii.gz
├── Supportimages/             # 3D support images (.nii.gz)
│   ├── 0002.nii.gz
│   ├── 0003.nii.gz
│   └── ...
├── Supportlabels/             # 3D support labels (.nii.gz)
│   ├── 0002.nii.gz
│   └── 0003.nii.gz
└── 
```
</details>
    
<!--[ ] insert video for support generation here-->

   

### Output the segmentation results for further usage
For commond line inference, segmentation results are output to the `predictions/` folder as .nii format which can be used for further usage.
For GUI inference, segmentation results can be output to specified path as .nii format.


<!--### TODO
[ ] GUI video demo refinement
    [ ] normal speed
    [ ] intermediate result showing
    [ ] optimize layout and button
    [ ] consistency check
[ ] Support generation video demo
[ ] Output options of GUI-->

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
