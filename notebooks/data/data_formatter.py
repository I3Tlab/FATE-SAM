#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import SimpleITK as sitk
from PIL import Image

# === Utils ===
def is_med3d(path: Path) -> bool:
    sfx = "".join(path.suffixes).lower()
    return (
        sfx.endswith(".nii")
        or sfx.endswith(".nii.gz")
        or sfx.endswith(".mhd")
        or sfx.endswith(".mha")
        or sfx.endswith(".nrrd")
    )

def list_volumes(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.is_file() and is_med3d(p)], key=lambda p: p.as_posix())

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_volume_sitk(src: Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    img = sitk.ReadImage(str(src))
    arr_zyx = sitk.GetArrayFromImage(img)  # (z,y,x)
    sp_xyz = img.GetSpacing()              # (sx,sy,sz)
    spacing_zyx = (sp_xyz[2], sp_xyz[1], sp_xyz[0])
    if arr_zyx.ndim == 4:
        arr_zyx = arr_zyx[0]
    if arr_zyx.ndim != 3:
        raise ValueError(f"Expected 3D volume at {src}, got {arr_zyx.shape}")
    return arr_zyx, spacing_zyx

def write_nifti_gz(arr_zyx: np.ndarray, spacing_zyx: Tuple[float,float,float], out_path: Path):
    ensure_dir(out_path.parent)
    img = sitk.GetImageFromArray(arr_zyx)
    img.SetSpacing((spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]))
    sitk.WriteImage(img, str(out_path), useCompression=True)

def robust_u8(img2d: np.ndarray, p_lo: float = 0.5, p_hi: float = 99.5) -> np.ndarray:
    a = img2d.astype(np.float32)
    lo, hi = np.percentile(a, [p_lo, p_hi])
    if hi <= lo:
        lo, hi = float(a.min()), float(a.max()) if a.size else (0.0, 1.0)
    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo + 1e-8)
    a = (a * 255.0).round().astype(np.uint8)
    return a

def save_volume_as_jpg_slices(arr_zyx: np.ndarray, out_dir: Path):
    ensure_dir(out_dir)
    Z = arr_zyx.shape[0]
    for i in range(Z):
        sl = arr_zyx[i]
        u8 = robust_u8(sl)
        Image.fromarray(u8, mode="L").save(out_dir / f"{i+1:03d}.jpg", quality=95)

def out_name_for(src: Path) -> str:
    for suf in (".nii.gz", ".nii", ".mhd", ".mha", ".nrrd"):
        if src.name.lower().endswith(suf):
            return src.name[: -len(suf)]
    return src.stem

# === Process ===
def process_bucket(in_root: Path, out_root: Path, treat_as_label: bool = False) -> List[Path]:
    vols = list_volumes(in_root)
    ensure_dir(out_root)
    written = []
    for src in vols:
        base = out_name_for(src)
        out_nii = out_root / f"{base}.nii.gz"
        arr, sp = read_volume_sitk(src)
        if treat_as_label and not np.issubdtype(arr.dtype, np.integer):
            arr = np.rint(arr).astype(np.int16)
        write_nifti_gz(arr, sp, out_nii)
        written.append((base, arr))
        print(f"→ {out_nii}")
    return written

def main():
    ap = argparse.ArgumentParser(description="Ingest medical 3D volumes → .nii.gz + JPG slices for support images.")
    ap.add_argument("--test_images", required=True, type=Path)
    ap.add_argument("--test_labels", required=True, type=Path)
    ap.add_argument("--support_images", required=True, type=Path)
    ap.add_argument("--support_labels", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    args = ap.parse_args()

    out_root = args.out_root
    out_test_images  = out_root / "TestImages"
    out_test_labels  = out_root / "TestLabels"
    out_support_imgs = out_root / "SupportImages"
    out_support_labs = out_root / "SupportLabels"
    out_support_slices = out_root / "SupportImage_slices"

    # Standardize to .nii.gz
    process_bucket(args.test_images, out_test_images, treat_as_label=False)
    process_bucket(args.test_labels, out_test_labels, treat_as_label=True)
    support_written = process_bucket(args.support_images, out_support_imgs, treat_as_label=False)
    process_bucket(args.support_labels, out_support_labs, treat_as_label=True)

    # Also make JPG slices for support images 
    for base, arr in support_written:
        save_volume_as_jpg_slices(arr, out_support_slices / base)

    print("✅ Ingest complete.")

if __name__ == "__main__":
    main()
