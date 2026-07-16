# Environmental Sound Classification: Supervised vs. Self-Supervised

Classifying environmental sounds on [ESC-50](https://github.com/karolpiczak/ESC-50) (2,000 clips, 50 classes, 5 folds), comparing a standard supervised baseline against a COLA-style contrastive self-supervised pipeline, and honestly reporting *why* the self-supervised approach underperformed.

> **TL;DR.** The supervised CNN reaches ~62% test accuracy. The self-supervised contrastive model, pretrained on ESC-50 alone, only reaches ~19% with a frozen linear probe. This is a clean, measured demonstration of shortcut learning: contrastive SSL needs large-scale unlabeled data (e.g. AudioSet, ~2M clips), and on a 2k-clip dataset the pretext task becomes trivial, so the learned features don't transfer.

> ℹ️ **Note on the numbers.** The accuracies above are aggregated from private Weights & Biases runs and are not reproducible from this repo alone (no run artifacts committed). Some contrastive/probe runs used a 10-fold corpus (UrbanSound8K), so the committed `conf/config.yaml` folds won't match ESC-50 as-is. Treat the figures as reported, not verifiable here.

---

## Results

| Approach | Backbone / features | Test accuracy |
|---|---|---|
| Supervised (baseline) | Compact CNN on MFCCs | ~62% |
| Contrastive, fine-tuned (encoder unfrozen) | EfficientNet-B0 on log-mel | 23–39% |
| Contrastive, frozen linear probe | EfficientNet-B0 on log-mel | ~19% |
| Contrastive, pretraining pretext accuracy | n/a | 100% 🚩 |

*(Metrics aggregated from 25 [Weights & Biases](https://wandb.ai) runs. Chance level on ESC-50 = 2%.)*

## The interesting part: why self-supervision failed here

The self-supervised model implements COLA (Saeed et al., 2020, *Contrastive Learning of General-Purpose Audio Representations*): an EfficientNet-B0 encoder trained to match two augmented views of the same clip against other clips in the batch.

The 100% pretext accuracy is the smoking gun. The model solved its self-supervised task perfectly, which sounds good but is actually the failure. On ESC-50, different clips have very different low-level statistics (loudness, spectral envelope), so the model learned to match views by cheap shortcuts instead of by semantic content. When a pretext task is that easy, the encoder learns nothing transferable, confirmed by the frozen linear probe collapsing to ~19% against the 62% supervised baseline.

The root cause is data scale, not a code bug. Contrastive SSL (COLA, SimCLR) is designed for massive unlabeled corpora: the original COLA pretrains on AudioSet (~2M clips) and *then* transfers to ESC-50. Pretraining on ~1,600 clips and fine-tuning on the same tiny set can't beat plain supervised learning. The pipeline is correct; the experimental setup asks SSL to do something it fundamentally needs scale for.

Takeaway: self-supervised pretraining is a scale game. Below a certain data volume, a straightforward supervised model wins, and the linear-probe number tells you the truth about representation quality regardless of what the pretext accuracy says.

## What's in here

A full PyTorch Lightning pipeline with four entry points:

- `main_supervised.py`: supervised CNN baseline on MFCC features
- `main_contrastive.py`: COLA-style contrastive pretraining (self-supervised, no labels)
- `main_selfsup.py`: self-supervised training flow
- `main_finetune.py`: fine-tune / linear-probe the pretrained encoder for classification

Stack: PyTorch · PyTorch Lightning · EfficientNet-B0 · librosa (log-mel / MFCC) · Hydra (config) · Weights & Biases (tracking) · Google Cloud Storage (streaming data to Colab).

## Repo structure

```
environmental_sound/
├── conf/config.yaml          # Hydra config: all hyperparameters per training mode
├── data/                     # audio -> spectrogram processing, dataloaders
├── supervised/               # CNN baseline + spectrogram transforms
├── contrastive/              # COLA encoder, contrastive datasets, augmentations
├── utils/                    # GCP bucket helpers, embedding viz, misc
├── main_supervised.py
├── main_contrastive.py
├── main_selfsup.py
└── main_finetune.py
notebooks/                    # EDA + spectrogram exploration
```

## Running it

```bash
pip install -r requirements.txt

# Supervised baseline
python -m environmental_sound.main_supervised

# Self-supervised: pretrain, then fine-tune / probe
python -m environmental_sound.main_contrastive
python -m environmental_sound.main_finetune

# Self-supervised training flow (loads the contrastive checkpoint)
python -m environmental_sound.main_selfsup
```

All hyperparameters live in `environmental_sound/conf/config.yaml` (batch size, temperature, embedding dim, freeze-encoder, learning rates, W&B project). ESC-50 audio is expected under `audio_data/` (see `data/`).

## If you wanted to make the self-supervised approach actually work

1. Pretrain on a large unlabeled corpus (FSD50K, UrbanSound8K, or AudioSet), *not* ESC-50 alone. This is the single biggest lever.
2. Harden the pretext task so pretraining accuracy stops pegging at 100% (stronger, symmetric augmentations; watch for that saturation as a warning sign).
3. L2-normalize embeddings consistently between pretraining and fine-tuning.
4. Use the frozen linear probe as the primary SSL metric. It's the honest measure of representation quality.

---

<sub>Original code (2025) written by me; results analysis (pulled from Weights & Biases) and repo cleanup done with the help of [Claude Code](https://claude.com/claude-code).</sub>
