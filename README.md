# Few-Shot Adaptation of Training-Free Foundation Model for 3D Medical Image Segmentation

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Publication](#publication)
- [Contacts](#contacts)

## Introduction
Few-shot Adaptation of Training-frEe SAM (FATE-SAM) is a versatile framework for 3D medical image segmentation that adapts the pretrained SAM2 model without fine-tuning. By leveraging a support example-based memory mechanism and volumetric consistency, FATE-SAM enables prompt-free, training-free segmentation across diverse medical datasets. This approach achieves robust performance while eliminating the need for large annotated datasets. For more details and results, please refer to our <a href="https://arxiv.org/abs/2501.09138" target="_blank">paper</a>.

![figure1.svg](resources%2Ffigure1.jpg)

## Getting Started
The hardware environment we tested.
- OS: Ubuntu 20.04.6 LTS
- CPU: Intel Core i9-10980XE CPU @ 3.00GHz * 36
- GPU: NVIDIA RTX A5000

### Installation
0. Download and install the appropriate version of NVIDIA driver and CUDA for your GPU.
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
5. Download checkpoints

    You can either download the `sam2.1_hiera_large.pt` directly using this <a href="https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt" target="_blank">link</a>  and place it into the `checkpoints/` directory, or follow the official <a href="https://github.com/facebookresearch/sam2/tree/main?tab=readme-ov-file#download-checkpoints" target="_blank">SAM2 repository</a> for more options.

### Quick Inference (Single Volume)

Run inference with GUI
```bash
streamlit run notebooks/app.py
```
<details> 
  <summary>▶️ Watch GUI Demo Video</summary>
  
  <div align="center">
    <a href="https://youtu.be/fN9Zy7Tew0I" target="_blank">
      <img src="./resources/FATE_SAM_DEMO.gif" 
           alt="Demo Video" 
           width="750" 
           style="border:2px solid #000; border-radius:8px;">
    </a>
    <p><em>Click to watch on YouTube</em></p>
  </div>  
</details>


### Batch Inference
Run inference with Command Line
```bash
python notebooks/fate_sam_predict.py \
       --test_image_path <path-to-test-image-nii.gz-folder> \
       --support_images_path <path-to-support-images-jpg-dir> \
       --support_labels_path <path-to-support-labels-nii.gz-dir> \
```

### Support Data 
To perform inference, support images and labels are needed. We provided a few support examples which can be found in `notebooks/data` covering different anatomies adopted from following publicly available datasets.
<table>
  <tr>
    <th style="width:20%">Dataset</th>
    <th style="width:10%">Anatomy</th>
    <th>Segmentation Objects</th>
    <th style="width:10%">Modality</th>
  </tr>
  <tr>
    <td><a href="https://huggingface.co/datasets/YongchengYAO/SKI10">SKI10</a></td>
    <td>Knee</td>
    <td>(1) femur (2) femoral cartilage (3) tibia (4) tibial cartilage</td>
    <td>MRI</td>
  </tr>
  <tr>
    <td><a href="https://www.synapse.org/Synapse:syn3193805/wiki/">BTCV</a></td>
    <td>Abdomen</td>
    <td>(1) spleen (2) right kidney (3) left kidney (4) gallbladder (5) esophagus (6) liver (7) stomach (8) aorta (9) inferior vena cava (10) portal vein and splenic vein (11) pancreas (12) right adrenal gland (13) left adrenal gland</td>
    <td>CT</td>
  </tr>
  <tr>
    <td><a href="https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html">ACDC</a></td>
    <td>Heart</td>
    <td>(1) left ventricle (2) right ventricle (3) myocardium</td>
    <td>Cine-MRI</td>
  </tr>
  <tr>
    <td><a href="http://medicaldecathlon.com/">MSD-Hippocampus</a></td>
    <td>Brain</td>
    <td>(1) anterior (2) posterior</td>
    <td>MRI</td>
  </tr>
  <tr>
    <td><a href="http://medicaldecathlon.com/">MSD-Prostate</a></td>
    <td>Prostate</td>
    <td>(1) peripheral zone (2) transition zone</td>
    <td>MRI</td>
  </tr>
</table>

To use your own support images and labels, please save the support images as .nii format. We recommoned using [3D slicer](https://www.slicer.org/) to generat the support label and saving them as following formats and folder structures.

Please follow [Data Format](./notebooks/data/DATA_FORMAT.md) for more guidance. 
<!--[ ] insert video for support generation here-->
<details>
<summary>Sample dataset directory structure (click to view details)</summary>
  
```commandline
<dataset>/
├── Test/
│   └── 0001_img.nii.gz
│── Support/             
│   ├── 0002_img.nii.gz
│   ├── 0002_label.nii.gz
│   ├── 0003_img.nii.gz
│   │── 0003_label.nii.gz
│   └── ...
└── 
```
Note: Always use the suffix  `_img` for support/test images and `_label` for their corresponding labels. This naming convention ensures the files are correctly matched during processing.
</details>

### Output
For GUI inference, segmentation results can be output to specified path as .nii format.
For commond line inference, segmentation results are output to the `predictions/` folder as .nii format which can be used for further usage.


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
