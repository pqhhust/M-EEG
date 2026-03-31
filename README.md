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

## 🚀 Getting Started
### Installation

Each model architecture has its own dependencies. We recommend using **separate virtual environments**:

#### CBraMOD

```bash
cd CBraMod
conda create -n cbramod python=3.11
conda activate cbramod
pip install -r requirements.txt
```

#### EEGPT

```bash
cd EEGPT
conda create -n eegpt python=3.11
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

## 📦 Data Access

---

<div align="center">

**⚠️ CONFIDENTIAL - Under Review for ICML 2026**  
**Anonymous Submission - Do Not Distribute**

*This repository includes sample data and manuscript for reviewer verification*  
*Full repository details and contact information will be made available upon acceptance*

</div>
