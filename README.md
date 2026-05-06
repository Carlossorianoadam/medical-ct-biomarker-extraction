# Medical CT Biomarker Extraction

Deep learning pipeline for automatic anatomical segmentation and biomarker extraction from abdominal CT scans using nnU-Net v2.

The project focuses on vertebral and psoas muscle analysis for body composition assessment through medical imaging.

---

## Main Features

- Multi-class CT segmentation
- Vertebral detection from T12 to L4
- Bilateral psoas segmentation
- Automatic Hounsfield Unit (HU) analysis
- Cross-sectional and volumetric measurements
- Anatomical fallback logic
- Structured CSV / JSON outputs
- Visualization utilities for quality control

---

## Pipeline

```text
CT Scan
   ↓
Preprocessing
   ↓
nnU-Net Segmentation
   ↓
3D Anatomical Masks
   ↓
Biomarker Extraction
   ↓
CSV / JSON Results
```

---

## Segmented Structures

The model predicts:

- T12 vertebra
- L1 vertebra
- L2 vertebra
- L3 vertebra
- L4 vertebra
- Left psoas muscle
- Right psoas muscle

---

## Extracted Biomarkers

### Vertebral analysis
- Trabecular attenuation (HU)
- Vertebral volumetric estimation

### Muscle analysis
- Psoas cross-sectional area
- Mean muscle attenuation (HU)
- Bilateral psoas volume

---

## Technologies

- Python
- PyTorch
- nnU-Net v2
- NumPy
- NiBabel
- Matplotlib
- MONAI

---

## Training

### Dataset preprocessing

```bash
python prepare_dataset.py
python generate_dataset_json.py
python prepare_train_test_split.py
```

### nnU-Net preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity
```

### Training

#### 2D model

```bash
nnUNetv2_train 001 2d 0
```

#### 3D full-resolution model

```bash
nnUNetv2_train 001 3d_fullres 0
```

---

## Inference

### Run segmentation

```bash
python run_nnunet_inference.py
```

### Extract biomarkers

```bash
python extract_biomarkers.py
```

---

## Performance

| Structure | Dice Score |
|---|---|
| Vertebrae | 0.92 - 0.94 |
| Psoas Muscles | ~0.85 |
| Global Mean | ~0.92 |

---

## Repository Structure

```text
medical-ct-biomarker-extraction/
│
├── src/
│   ├── data_preparation/
│   ├── inference/
│   └── clinical_analysis/
│
├── examples/
├── docs/
├── configs/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The project uses CT volumes derived from the public TotalSegmentator dataset.

Only vertebral and psoas-related anatomical labels were selected for the segmentation task.

---

## Notes

This repository does not contain:
- hospital CT scans,
- patient data,
- clinical outputs,
- or trained weights.

The repository is intended for educational and research purposes only.

---

## Author

Carlos Soriano Adam  
Data Science Student — Universitat Politècnica de València
