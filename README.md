<div align="center">

# M-EEG-experiment

Fine-tuning for EEG foundation models.
Run each architecture **from its own folder** (`CBraMod/` or `EEGPT/`).

</div>

---

## 📁 Repository Layout

```
M-EEG-experiment/
├─ CBraMod/          # CBraMod model code + configs + utils + scripts (multiple files)
├─ EEGPT/            # EEGPT model code + configs + utils + scripts (multiple files)
└─ readme.md
```

## 🔨 Install (per-model dependencies)

### CBraMod
```bash
cd CBraMod
pip install -r requirements.txt
```

### EEGPT
```bash
cd EEGPT
pip install -r requirements.txt
```

> Recommended: use a **separate virtual environment per model** to avoid dependency conflicts.

---

## 🚀 Fine-tune

### 1) CBraMod (run inside `CBraMod/`)
```bash
cd CBraMod

python finetune_main.py
```

### 2) EEGPT (run inside `EEGPT/`)
```bash
cd EEGPT

python finetune_main.py 
```

---

## 🧩 CLI Options

To see all supported options for a given model:
```bash
python finetune_main.py -h
```

Common options you will typically use:
- `--foundation_dir`: path to the pretrained checkpoint
- `--datasets_dir`: path to the preprocessed downstream dataset folder
- `--downstream_dataset`: downstream task/dataset name (e.g., `pearl`, `epilepsy-bbb`, `HOS-108`)
- `--modality_mode`: modality setting (multimodality or EEG only)
- training knobs: `--epochs`, `--batch_size`, `--lr`, `--dropout`, `--weight_decay`

---

## ✅ Notes

- Always run commands **from inside the model folder** (important for relative imports/config paths).
- If your paths differ, update `--datasets_dir` and `--foundation_dir` accordingly.
