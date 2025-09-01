# Data Format Guide

> This guide expalins how to prepare your data for FATE-SAM

## Convert Custom Data
The formatter script will:

1. Standardize all 3D inputs to **`.nii.gz`**
2. Generate per-slice jpg folder for support images are produced along z-axis

```bash
python data_formatter.py \
  --test_images      /path/to/raw/test_images \
  --test_labels      /path/to/raw/test_labels \
  --support_images   /path/to/raw/support_images \
  --support_labels   /path/to/raw/support_labels \
  --out_root         /data/dataset_name
```

## Input Data Formats
* **Accepted input formats (images & labels)**: `.nii`, `.nii.gz`, `.mhd`, `.mha`, `.nrrd`
* **3D volumes**: `{name}.nii.gz`
  * `name` may be any ASCII string, e.g., `case_0001`, `liver_001`
  * Zero-padding recommended, e.g., `case_0001`, `case_0002`
  * volumes are treated as (Z, Y, X)

## Output Directory Organization
```
<data_root>/
├── TestImages/               # 3D images (.nii.gz)
├── TestLabels/               # 3D labels (.nii.gz), optional
├── SupportImages/            # 3D images (.nii.gz)
├── SupportLabels/            # 3D labels (.nii.gz)
└── SupportImage_slices/      # JPG slices for SupportImages only (per case folder)
```