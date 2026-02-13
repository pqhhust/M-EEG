<div align="center">

# M-EEG: Multi-Institutional Multimodal EEG Benchmark

**A Multi-Institutional Multimodal EEG Benchmark for Foundation Model Generalization and Early Neurological Diagnosis**

[![Paper](https://img.shields.io/badge/Paper-ICML%202026%20(Under%20Review)-red)]()
[![Dataset](https://img.shields.io/badge/Dataset-M--EEG-blue)]()
[![License](https://img.shields.io/badge/License-Research%20Only-green)]()

*Implementation of EEG Foundation Model benchmarking on the M-EEG dataset*

**Anonymous Submission for ICML 2026**

</div>

---

## 📋 Overview

**M-EEG** is a **large-scale clinical EEG dataset** comprising **1,170 hours** of recordings from **6,081 patients** across two major hospitals. This repository provides:

- **Standardized benchmarking** of state-of-the-art EEG foundation models (CBraMOD, EEGPT)
- **Multimodal fusion** capabilities integrating EEG with blood-based biomarkers (BBB) and clinical notes
- **Unified evaluation protocols** across diverse neurological diagnostic tasks
- **Enhanced regional diversity** addressing geographic limitations in existing EEG datasets

### Key Contributions

1. **M-EEG Dataset**: Large-scale multimodal EEG corpus with synchronized blood biomarkers and clinical narratives from international clinical cohorts
2. **P-EEG**: Unified pretraining corpus combining M-EEG with public datasets (TUEG, NMT-Scalp) to enhance geographic and clinical diversity
3. **T-EEG**: Comprehensive task-oriented benchmark spanning motor imagery, sleep staging, seizure detection, and disease prediction
4. **Multimodal Models**: Demonstrated 27.64% improvement in Alzheimer's risk prediction through EEG-biomarker integration

---

## 🏗️ Repository Structure

```
M-EEG-experiment/
├── CBraMod/              # CBraMOD architecture implementation
├── EEGPT/                # EEGPT architecture implementation
├── SampleData.tar.gz     # Sample M-EEG data for reviewers
├── manuscript.pdf        # Paper manuscript (ICML 2026 submission)
└── README.md             # This file
```

---

## 📊 Dataset Overview

### M-EEG Components

| Component | Description | Size |
|-----------|-------------|------|
| **Hospital A** | Clinical EEG recordings | 947 patients, 290 hours, 22 channels @ 200Hz |
| **Hospital B** | Multimodal EEG + BBB + Notes | 5,134 patients, 880 hours, 44 channels @ 500Hz |
| **Total** | Combined corpus | 6,081 patients, 1,170 hours |

### Multimodal Subset (Hospital B)

- **Blood-based biomarkers (BBB)**: Complete blood count, lipid panel, liver enzymes, renal markers, electrolytes, glucose/HbA1c
- **Clinical notes**: De-identified MRI reports and diagnostic impressions
- **Temporal alignment**: Same-day synchronization between EEG, labs, and notes

### Benchmark Tasks (T-EEG)

| Task | Dataset | Classes | Evaluation |
|------|---------|---------|------------|
| Motor Imagery | BCIC-2a, A&MISP, ALS | 4 | Leave-one-subject-out |
| Sleep Staging | SleepEDF | 5 | 5-fold cross-validation |
| Seizure Detection | TUEV | 4 | Hold-out test set |
| Abnormal EEG | TUAB | 2 | Hold-out test set |
| Alzheimer's Risk | PEARL | 2 | Cross-validation |
| Epilepsy | M-EEG-EPI | 2 | 5-fold cross-validation |
| TIA Detection | M-EEG-TIA | 2 | 5-fold cross-validation |
| Parkinson's | M-EEG-PD | 2 | 3-fold cross-validation |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16GB+ RAM

### Quick Start for Reviewers

1. **Extract sample data**:
   ```bash
   tar -xvf SampleData.tar.gz
   ```

2. **Set up environment** (choose one model):
   ```bash
   cd CBraMod  # or EEGPT
   conda create -n m-eeg python=3.8
   conda activate m-eeg
   pip install -r requirements.txt
   ```

3. **Verify installation** with sample data:
   ```bash
   python scripts/load_sample_data.py --data_dir ../SampleData
   ```

**For Reviewers**: The sample data allows verification of data formats, preprocessing pipelines, and model architectures. Due to file size constraints, the sample contains a representative subset. Full experimental reproduction requires the complete dataset available through the access process below.

### Installation

Each model architecture has its own dependencies. We recommend using **separate virtual environments**:

#### CBraMOD

```bash
cd CBraMod
conda create -n cbramod python=3.8
conda activate cbramod
pip install -r requirements.txt
```

#### EEGPT

```bash
cd EEGPT
conda create -n eegpt python=3.8
conda activate eegpt
pip install -r requirements.txt
```

---

## 🔧 Usage

### Fine-tuning on Downstream Tasks

#### 1. Unimodal EEG Fine-tuning

Run from inside the respective model directory:

```bash
# CBraMOD (using P-EEG pretrained checkpoint)
cd CBraMod
python finetune_main.py \
    --foundation_dir ./checkpoints/pretrained_peeg \
    --datasets_dir /path/to/datasets \
    --downstream_dataset TUAB \
    --modality_mode eeg_only \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4

# EEGPT (using P-EEG pretrained checkpoint)
cd EEGPT
python finetune_main.py \
    --foundation_dir ./checkpoints/pretrained_peeg \
    --datasets_dir /path/to/datasets \
    --downstream_dataset TUAB \
    --modality_mode eeg_only \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4
```

#### 2. Multimodal Fine-tuning (EEG + BBB)

For tasks with blood biomarker data:

```bash
python finetune_main.py \
    --foundation_dir ./checkpoints/pretrained_peeg \
    --datasets_dir /path/to/datasets \
    --downstream_dataset pearl-msit \
    --modality_mode eeg_bbb \
    --fusion_method attention \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4
```

#### 3. Multimodal Fine-tuning (EEG + Clinical Notes)

For tasks with textual clinical data:

```bash
python finetune_main.py \
    --foundation_dir ./checkpoints/pretrained_peeg \
    --datasets_dir /path/to/datasets \
    --downstream_dataset meeg-epi \
    --modality_mode eeg_text \
    --text_encoder clinical-t5 \
    --epochs 50 \
    --batch_size 64 \
    --lr 1e-4
```

### Command-line Arguments

View all available options:

```bash
python finetune_main.py --help
```

**Key arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `--foundation_dir` | Path to pretrained checkpoint | Required |
| `--datasets_dir` | Root directory for datasets | Required |
| `--downstream_dataset` | Task name (see benchmark tasks above) | Required |
| `--modality_mode` | `eeg_only`, `eeg_bbb`, `eeg_text`, or `eeg_bbb_text` | `eeg_only` |
| `--fusion_method` | `concat` or `attention` | `attention` |
| `--epochs` | Number of training epochs | 50 |
| `--batch_size` | Training batch size | 64 |
| `--lr` | Learning rate | 1e-4 |
| `--dropout` | Dropout rate | 0.1 |
| `--weight_decay` | AdamW weight decay | 5e-2 |

---

## 📈 Benchmark Results

### Cross-Regional Generalization (Balanced Accuracy)

| Architecture | BCIC-2a (in-region) | A&MISP (out-of-region) | Gain from P-EEG |
|--------------|---------------------|------------------------|-----------------|
| CBraMOD (Base) | 0.4907 | 0.2604 | - |
| CBraMOD (P-EEG) | **0.4978** (+1.45%) | **0.2715** (+4.26%) | ✓ |
| EEGPT (Base) | 0.5051 | 0.2507 | - |
| EEGPT (P-EEG) | **0.5374** (+6.39%) | **0.2716** (+8.37%) | ✓ |

### Multimodal Disease Prediction (Balanced Accuracy)

| Task | Architecture | EEG-only | EEG + BBB | Improvement |
|------|--------------|----------|-----------|-------------|
| **Alzheimer's (PEARL-MSIT)** | CBraMOD | 0.5283 | **0.6743** | +27.64% |
| | EEGPT | 0.4615 | **0.5774** | +25.11% |
| **Epilepsy (M-EEG)** | CBraMOD | 0.5248 | **0.6280** | +19.67% |
| | EEGPT | 0.5144 | **0.6306** | +22.59% |
| **TIA (M-EEG)** | CBraMOD | 0.5266 | **0.5680** | +7.86% |
| | EEGPT | 0.5446 | **0.6263** | +15.00% |
| **Parkinson's (M-EEG)** | CBraMOD | 0.5556 | **0.6667** | +20.00% |
| | EEGPT | 0.6157 | **0.7667** | +24.53% |

*See paper for complete metrics (AUPR, AUROC, Cohen's κ, Weighted F1)*

---

## 📦 Sample Data for Reviewers

To facilitate review and verification of our methods, we provide **SampleData.tar** containing:

- **Preprocessed EEG samples** from M-EEG dataset in BIDS format
- **Example blood biomarker data** (anonymized)
- **Sample clinical notes** (de-identified)
- **Data loading scripts** and preprocessing utilities

### Extracting Sample Data

```bash
# Extract the sample data
tar -xvf SampleData.tar

# The extracted structure follows BIDS v1.8.0:
SampleData/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── phenotype/
│   ├── results.tsv          # Blood biomarker values
│   └── results.json         # Clinical notes metadata
└── sub-*/
    ├── sub-*_scans.tsv
    └── eeg/
        ├── sub-*_task-rest_eeg.edf
        ├── sub-*_task-rest_eeg.json
        └── sub-*_task-rest_channels.tsv
```

### Quick Start with Sample Data

```bash
# Example: Load and visualize sample EEG data
cd CBraMod  # or EEGPT
python scripts/load_sample_data.py --data_dir ../SampleData

# Example: Run preprocessing on sample data
python scripts/preprocess.py \
    --input_dir ../SampleData \
    --output_dir ./processed_samples
```

**Note**: The sample data contains a small subset of anonymized records for verification purposes only. The full M-EEG dataset will be available through the controlled-access process described below.

---

## 📄 Manuscript

The full paper manuscript is provided as **manuscript.pdf** for reviewer convenience. This document contains:

- Detailed methodology and experimental setup
- Complete results and ablation studies
- Supplementary materials and appendices
- BIDS dataset structure documentation

---

## 📦 Data Access

Due to patient privacy regulations and institutional review board (IRB) requirements, the M-EEG dataset will be available through a **controlled-access process** upon paper acceptance.

### Request Pipeline (Post-Acceptance)

1. **Review institutional requirements** 
2. **Submit standardized data request form** including:
   - Research proposal summary
   - Intended use case
   - Data security plan
   - IRB approval (if applicable)
3. **Institutional evaluation** (typically 5-7 days)
4. **Secure cloud environment access** upon approval

*Detailed access instructions will be provided upon publication.*

### Data Format

The dataset follows **Brain Imaging Data Structure (BIDS) v1.8.0**:

```
M-EEG/
├── dataset_description.json
├── participants.tsv
├── participants.json
├── phenotype/
│   ├── results.tsv          # Blood biomarker values
│   └── results.json         # Clinical notes
└── sub-xxxx/
    ├── sub-xxxx_scans.tsv
    └── eeg/
        ├── sub-xxxx_task-rest_eeg.edf
        ├── sub-xxxx_task-rest_eeg.json
        └── sub-xxxx_task-rest_channels.tsv
```

---

## 🔬 Preprocessing Pipeline

All datasets undergo standardized preprocessing:

1. **Band-pass filtering**: 0.3–75 Hz (clinical tasks), 1–50 Hz (BCI tasks)
2. **Notch filtering**: 50 Hz or 60 Hz (region-dependent)
3. **Channel selection**: Mapping to 19 standard 10-20 channels
4. **Resampling**: 200 Hz uniform sampling rate
5. **Artifact rejection**: Independent Component Analysis (ICA)
6. **Normalization**: Per-channel z-score normalization to [-1, 1]
7. **Segmentation**: Task-specific window lengths (4s, 10s, or 30s)

---

## 📝 Citation

If you use this code or the M-EEG dataset, please cite:

```bibtex
@inproceedings{meeg2026,
  title={A Multi-Institutional Multimodal EEG Benchmark for Foundation Model Generalization and Early Neurological Diagnosis},
  author={Anonymous Authors},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026},
  note={Under review}
}
```

---

## 🤝 Contributing

We welcome contributions! For anonymous review purposes, contribution guidelines will be provided upon paper acceptance.

*During the review period, please direct questions through the conference review system.*

---

## 📄 License

This project is released under a **Research-Only License**. The code and data are provided for academic research purposes only. Commercial use is prohibited without explicit permission.

---

## 🙏 Acknowledgments

- Data collection was performed in full compliance with institutional ethical guidelines
- We thank the participating hospitals and patients for their contributions
- Funding and affiliation details will be disclosed upon paper acceptance

---

## 📧 Contact

**Anonymous Submission - ICML 2026**

Contact information will be provided upon paper acceptance.

For questions during review, please use the conference review system.

---

## 🔗 Related Resources

- **TUEG Dataset**: [Temple University EEG Corpus](https://isip.piconepress.com/projects/tuh_eeg/)
- **BIDS Standard**: [Brain Imaging Data Structure](https://bids.neuroimaging.io/)
- **CBraMOD Paper**: [Criss-cross Brain Foundation Model](https://openreview.net/forum?id=NPNUHgHF2w)
- **EEGPT Paper**: [Pretrained Transformer for Universal EEG](https://openreview.net/forum?id=lvS2b8CjG5)

---

<div align="center">

**⚠️ CONFIDENTIAL - Under Review for ICML 2026**  
**Anonymous Submission - Do Not Distribute**

*This repository includes sample data and manuscript for reviewer verification*  
*Full repository details and contact information will be made available upon acceptance*

</div>
