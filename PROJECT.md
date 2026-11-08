# Project: M2FTrans MyoPS380 Adaptation (`myops380_train304_test76_last_v1`)

## 1. Overview & Research Context
This project adapts the official **M2FTrans** architecture (*Modality-Masked Fusion Transformer for Incomplete Multi-Modality Brain Tumor Segmentation*, Shi et al., *IEEE Journal of Biomedical and Health Informatics*, 2024) to the **MyoPS380** benchmark protocol (`myops380_train304_test76_last_v1`), strictly conforming to the reference benchmark established in **SCAR** (`https://github.com/thanhquan123hi1/SCAR.git`) and the standardization rules specified in `Promt_train_test_no_val.txt`.

---

## 2. Architectural Formulation: 3D to 2D Adaptation
The original M2FTrans codebase was designed for 3D volumetric multi-modal brain MRI (BraTS, $128 \times 128 \times 128$, 4 modalities: FLAIR, T1ce, T1, T2). For MyoPS380, the benchmark consists of 2D short-axis slices ($128 \times 128$, 3 aligned modalities: bSSFP, LGE, T2w).

The architecture has been faithfully adapted into `M2FTrans2D`:
1. **Three Dedicated 2D Encoders**:
   - `c0_encoder`, `lge_encoder`, `t2_encoder` extract multi-scale 2D features $(x_1, x_2, x_3, x_4, x_5)$ at resolutions $128\times 128, 64\times 64, 32\times 32, 16\times 16, 8\times 8$.
   - Retains the paper's residual convolution structure with LeakyReLU and Instance Normalization.
2. **Bottleneck MaskedTransformer with Fusion Token**:
   - At stage 5 ($8 \times 8$, 256 feature dims), spatial feature maps from each modality are flattened into 64 spatial tokens.
   - 3 modality token blocks ($3 \times 64 = 192$ tokens) are concatenated with a learnable `fusion` token block (64 tokens) and learnable 2D positional embeddings `self.pos` ($256 \times 256$ total tokens).
   - `MaskedTransformer` applies `MaskedAttention` incorporating modality availability masking (`mask_gen_fusion`). The fusion token attends to all present modalities and itself.
3. **Weight_Attention Module**:
   - Extracts the cross-attention matrix from the fusion token to each modality.
   - Computes spatial attention maps for each modality and modulates skip connections $(x_1 \dots x_5)$ hierarchically via 2D upsampling.
4. **Multi-Scale Decoder with CT and RFM**:
   - Stages 5 and 4 use `MultiCrossToken` (`CT5`, `CT4`) to perform cross-attention between decoder features and encoder tokens.
   - Stages 3, 2, and 1 use Region-aware Feature Modulation (`RFM3`, `RFM2`, `RFM1` via `fusion_postnorm2d`).
   - Deep supervision PRM heads (`prm_fusion1` through `prm_fusion5`) predict auxiliary multi-scale segmentation maps.
5. **Auxiliary Unimodal Decoders (`Decoder_sep`)**:
   - Predicts individual modality segmentations during training to enforce independent modality discriminability.
6. **Canonical 4-Class Output**:
   - `0`: Background
   - `1`: Normal Myocardium
   - `2`: Edema (Edema-exclusive)
   - `3`: Scar

---

## 3. Conformance to `Promt_train_test_no_val.txt`

| Section / Requirement | Specification | M2FTrans Implementation |
|---|---|---|
| **Dataset Protocol** | `myops380_train304_test76_last_v1` | Ingests 304 train patients (1,972 slices) from locked `train.txt`. |
| **Validation Policy** | Strictly NO validation (`no_val=True`) | Zero validation split, no validation loader, no validation loop, no early stopping, no `best.pt`. |
| **Label Mapping** | Legacy to Canonical | Raw cache legacy `{0:bg, 1:myo, 2:scar, 3:edema}` mapped to canonical `{0:bg, 1:myo, 2:edema, 3:scar}` via LUT `[0, 1, 3, 2]` before training. |
| **Augmentation** | Exact SCAR `RandomGenerator` | I-MMSeg deterministic sequence: rot/flip (50%), rotate $\pm 20^\circ$ (25%), gaussian noise (12.5%), roll shift $\pm 5$px (50%), brightness $[0.8, 1.2]$ (50%), gamma $[0.8, 1.2]$ (50%), crop (50%), cubic zoom. |
| **Training Budget** | 300 epochs, 37,200 updates | Effective batch size 16 (`batch_size=16, accum_steps=1`), 124 batches/epoch (last batch has 4 slices), 37,200 optimizer steps. |
| **Optimizer** | AdamW | `base_lr = 0.0003333`, `weight_decay = 0.0001` across all parameters, `betas = (0.9, 0.999)`, `eps = 1e-8`. |
| **Scheduler** | Polynomial LR Decay (`power = 0.9`) | Step-level decay with zero-based iteration offset: $\text{lr} = \text{base\_lr} \times \left(\max\left(0, 1 - \frac{\max(0, u-1)}{N}\right)\right)^{0.9}$. |
| **Objective / Loss** | Native M2FTrans Composite Loss | $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{fuse}} + \mathcal{L}_{\text{sep}} + \mathcal{L}_{\text{prm}}$ where each term is $\text{WeightedCE} + \text{SoftDice}$. |
| **Checkpoint Naming** | `<model_id>_seed<seed>.pt` | Explicitly named `m2ftrans_seed<seed>.pt`. Saved atomically each epoch; marked `training_complete = True` only upon completing 300 epochs. |
| **Test Evaluation** | 76 cases from `test_vol.txt` | Single-pass forward + argmax, zero morphological postprocessing, canonical export to `test_predictions/<CaseID>_pred.npz`. |
| **Shared Evaluator** | SCAR `surface_distance.py` | Metric protocol `myops380_voxel_v3` computing Dice, IoU, HD95 (voxel), ASD (voxel) across 5 regions. |

---

## 4. File Layout

```
M2FTrans/
├── config.yml                             # Complete configuration file
├── PROJECT.md                             # Technical overview and compliance report
├── train.py                               # 300-epoch training CLI engine
├── predict.py                             # Test inference and prediction export CLI
├── evaluate.py                            # Shared evaluation CLI
├── dataset/
│   ├── __init__.py
│   ├── data_augmentation.py               # Exact SCAR RandomGenerator
│   ├── myops380_dataset.py                # Train (2D) and Test (3D) dataset loaders
│   └── splits/
│       └── myops380_no_val/
│           ├── train.txt                  # 1,972 slice names (304 patients)
│           ├── test_vol.txt               # 76 test patient IDs
│           └── split_metadata.json        # Manifest checksums and provenance
├── models/
│   ├── __init__.py
│   ├── layers_2d.py                       # 2D Convolutions, Normalizations, PRM heads, RFM
│   ├── blocks_2d.py                       # 2D DepthWiseConv, MultiMaskAttention, CrossBlock
│   └── m2ftrans_2d.py                     # Canonical M2FTrans2D architecture
├── utils/
│   ├── __init__.py
│   ├── criterions.py                      # DiceLoss, SoftmaxWeightedLoss, CompositeLoss
│   ├── lr_scheduler.py                    # AdamW and Polynomial decay scheduler
│   ├── paths.py                           # Dynamic path resolution
│   └── surface_distance.py                # Shared SCAR evaluation metric protocol
└── tests/
    ├── __init__.py
    ├── test_m2ftrans.py                   # Model forward, backward, parameter tests
    ├── test_e2e.py                        # Dataset, checkpointing, and evaluation tests
    └── test_edge_cases.py                 # Edge-cases, boundary values, error paths
```

---

## 5. CLI Execution Commands

### A. Run Full Benchmark Training (300 Epochs, Zero Validation)
```bash
python train.py --batch_size 16 --accum_steps 1 --max_epochs 300 --lr 0.0003333 --weight_decay 0.0001 --seed 1234
```
For lower VRAM hardware (effective batch size 16):
```bash
python train.py --batch_size 4 --accum_steps 4 --max_epochs 300 --lr 0.0003333 --weight_decay 0.0001 --seed 1234
```

### B. Resume Incomplete Run
```bash
python train.py --resume checkpoints/m2ftrans_seed1234.pt --batch_size 16 --accum_steps 1 --max_epochs 300
```

### C. Run Prediction on 76 Test Patients
```bash
python predict.py --checkpoint checkpoints/m2ftrans_seed1234.pt
```

### D. Run Shared Evaluation on Test Predictions
```bash
python evaluate.py --checkpoint checkpoints/m2ftrans_seed1234.pt
```

### E. Run Full Test Suite
```bash
python -m unittest discover -s tests -p "test_*.py"
```
