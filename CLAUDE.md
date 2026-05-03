# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project performs **semantic segmentation of breathing audio signals** using a 1D U-NET + Bidirectional LSTM model. Each audio timestep is classified into one of three classes: background (0), inhalation (1), or exhalation (2). The model achieves >80% F1-score on both train and validation sets.

**Dataset:** HF_Lung_V1 breathing audio dataset (WAV files + CSV annotations).

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

### Step 1 — Preprocessing

```bash
python date_generation.py
```

Reads WAV files and per-file label TXT files from `train/` and `test/` directories at the repo root. Label files use the format `TYPE HH:MM:SS.mmm HH:MM:SS.mmm` (types: `I`=inhalation, `E`=exhalation; `D` and `Rhonchi` are skipped). Outputs pickle files to `output/filtered_500/3_split/signal/` and `output/filtered_500/3_split/segmented_label/`. Output directories are created automatically.

### Step 2 — Training

```bash
jupyter notebook unet1d.ipynb
```

Data paths are already configured to the local `output/` directory. Run all cells in order. Best model is saved to `best_unet1d_filter_model.keras`.

## Architecture

### Preprocessing (`date_generation.py`)

1. Decimate from 4000 Hz → 500 Hz (factor 8, IIR filter)
2. High-pass Butterworth filter (order=10, cutoff=80 Hz)
3. Min-Max normalization to [0, 1]
4. Split into 7.5-second segments (3,750 samples) with 50% overlap → 3 segments per file
5. Create per-sample class labels aligned to segment boundaries

### Model (`unet1d.ipynb`)

Input shape: `(3750, 1)` — 7.5 seconds mono at 500 Hz  
Output shape: `(3750, 3)` — per-sample softmax probabilities

```text
Encoder:  Conv1D(64)×2 → Pool → Conv1D(128)×2 → Pool
Bottleneck: Conv1D(128)×2 + Dropout(0.3)
Decoder:  ConvTranspose + skip concat → Conv1D(128)×2 → ConvTranspose + skip concat → Conv1D(64)×2
Temporal: Bidirectional LSTM(64) → Bidirectional LSTM(32)
Head:     Conv1D(3, activation='softmax')
```

**Loss:** Multiclass Dice loss (handles class imbalance better than cross-entropy).  
**Optimizer:** Adam with `ReduceLROnPlateau` (factor=0.5, patience=10).  
**Training:** 330 epochs, batch size 4, 90/10 train-val split (1,390 / 155 segments).

### Model Variants

- [UNet.txt](UNet.txt) — simple 2-block encoder/decoder, no skip connections in decoder
- [UNet+LSTM.txt](UNet+LSTM.txt) — full architecture with skip connections, Batch Norm, Dropout, and Bi-LSTM (this is what the notebook trains)
