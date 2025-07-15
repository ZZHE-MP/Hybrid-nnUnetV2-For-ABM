## Hybrid-nnU-Net

Hybrid-nnU-Net is an enhanced medical image segmentation pipeline that builds upon the standard nnU-Net framework by incorporating a Boolean integration strategy. This hybrid approach aims to refine the prediction of active bone marrow (ABM) regions in pelvic radiotherapy by combining the nnU-Net prediction with manually contoured whole bone marrow (BM) masks, followed by evaluation against PET-based ABM ground truth (ABM-GT).

<img width="4000" height="2250" alt="ABM-2" src="https://github.com/user-attachments/assets/d663004f-2935-4a11-bfbd-d74e6ce291c4" />

## Overview

The nnU-Net is a powerful, self-configuring deep learning framework for biomedical image segmentation. While nnU-Net provides strong out-of-the-box performance, Hybrid-nnU-Net introduces a domain-informed post-processing step by enforcing anatomical consistency between the predicted ABM region and the manually segmented bone marrow.

The refinement is achieved via a Boolean AND operation between:

The nnU-Net-predicted ABM mask (ABM-nnUnet.nii.gz), and

The manually contoured whole BM (BM.nii.gz).

The intersection result is then compared against PET-derived ABM ground truth (ABM-GT.nii.gz) using:

Dice Similarity Coefficient (DSC)

Hausdorff Distance (95th percentile, HD95)

## Directory Structure

Hybrid-nnUNet/
├── nnUNet-v2.zip               # Pretrained nnU-Net v2 model and prediction outputs
├── configuration.py            # Configuration settings for dataset paths and processing
├── paths.py                    # File path definitions for BM, ABM, and GT masks
├── boolean_integration.py      # Script for Boolean integration and evaluation (DSC, HD95)
├── pyproject.toml              # Project metadata and dependency management
├── README.md                   # This documentation file

## Boolean Integration and Evaluation

The following Python script demonstrates the core logic:

python

import SimpleITK as sitk
import numpy as np

## Load input images

abm_nnUnet = sitk.ReadImage("ABM-nnUnet.nii.gz")
bm_whole = sitk.ReadImage("BM.nii.gz")
abm_gt = sitk.ReadImage("ABM-GT.nii.gz")

## Sanity check: ensure spatial alignment

assert abm_nnUnet.GetSize() == bm_whole.GetSize() == abm_gt.GetSize(), "Size mismatch"
assert abm_nnUnet.GetSpacing() == bm_whole.GetSpacing() == abm_gt.GetSpacing(), "Spacing mismatch"
assert abm_nnUnet.GetOrigin() == bm_whole.GetOrigin() == abm_gt.GetOrigin(), "Origin mismatch"

## Boolean intersection: predicted ABM ∩ whole BM

abm_nnUnet_bin = abm_nnUnet > 0
bm_bin = bm_whole > 0
intersection_mask = sitk.And(abm_nnUnet_bin, bm_bin)
sitk.WriteImage(intersection_mask, "ABM-nnUnet_AND_BM.nii.gz")

## Dice computation

intersection_array = sitk.GetArrayFromImage(intersection_mask) > 0
abm_gt_array = sitk.GetArrayFromImage(abm_gt) > 0
intersection_volume = np.logical_and(intersection_array, abm_gt_array).sum()
volume_sum = intersection_array.sum() + abm_gt_array.sum()
dice = (2.0 * intersection_volume) / (volume_sum + 1e-8)
print(f"Dice Similarity Coefficient: {dice:.4f}")

## Hausdorff Distance (HD95)

intersection_bin = sitk.GetImageFromArray(intersection_array.astype(np.uint8))
intersection_bin.CopyInformation(abm_gt)
abm_gt_bin = sitk.GetImageFromArray(abm_gt_array.astype(np.uint8))
abm_gt_bin.CopyInformation(abm_gt)
hausdorff = sitk.HausdorffDistanceImageFilter()
hausdorff.Execute(intersection_bin, abm_gt_bin)
hd95 = hausdorff.GetHausdorffDistance()
print(f"Hausdorff Distance (95th percentile): {hd95:.2f} mm")
Evaluation Metrics
Dice Similarity Coefficient (DSC): Measures volumetric overlap between the hybrid segmentation and ground truth.

HD95: Robust version of the Hausdorff Distance that reflects surface agreement and spatial outliers.

<img width="4000" height="2250" alt="ABM-2" src="https://github.com/user-attachments/assets/b61c5a90-4380-4e95-ac54-b6c537e5449e" />
<img width="4000" height="2250" alt="ABM-1" src="https://github.com/user-attachments/assets/2714f64b-5ec1-4160-a6f7-00bdca9bc82c" />


## Use Cases

This hybrid pipeline is ideal for:

Improving clinical reliability of deep learning-based ABM segmentation.

Post-processing refinement using anatomical constraints.

PET-guided evaluation of segmentation performance.

## Citation

If you use this pipeline, please cite the original nnU-Net paper and your extended methodology:

@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
  publisher={Nature Publishing Group}
}
If Hybrid-nnU-Net is part of your publication, consider citing your own work describing the Boolean integration method.

## Acknowledgements

This pipeline was jointly developed by Peking University Shenzhen Hospital and Xi’an Jiaotong-Liverpool University in the context of radiation oncology research, with a focus on active bone marrow (ABM) delineation using PET imaging and deep learning methods.
