# Transfer Learning for Hierarchical Image Classification

## Background

Modern image recognition often involves hierarchical structure: a coarse label (e.g., **superclass** such as bird/dog/reptile) and a fine-grained **subclass** (e.g., hawk, chihuahua, mud turtle). 
This project studies how transfer learning strategies adapt to such hierarchical, low-resolution (64×64) images while also handling **open-set** conditions where test-time categories may not appear in training. We treat the task as a **multi-task problem** with a shared visual backbone and two classification heads—one for superclasses and one for subclasses. We compare a scratch-trained CNN against **ResNet-18** and **EfficientNet-B0**, and we systematically vary fine-tuning regimes (linear/partial/full) to understand when adaptation of pretrained features is most beneficial. We add **Mixup** to improve generalization and apply a **confidence threshold** at inference to route low-confidence predictions to a special “novel” class, enabling a simple form of novelty detection. These design choices are motivated by the challenge of fine-grained recognition under limited data and potential distribution shift (report in docs/)

---

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Python 3.10+ recommended
```

## Setup & Data

This project expects a small, labeled dataset of RGB images at 64×64 resolution. Each image has two labels:
- a superclass index (coarse category),
- a subclass index (fine category belonging to that superclass).


During training, images are resized to 64×64, randomly flipped/rotated/jittered, converted to tensors, and normalized to mean/std 0.5. 
These lightweight augmentations improve robustness without violating labels. 
The notebook handles an ~90/10 train/validation split and includes class-name↔index mapping utilities. 
The dataset characteristics, imbalance considerations, and open-set setting are detailed in the paper.

## Models (Dual-Head)

All models share the same two-head layout over a shared feature extractor:
- Head A: logits for superclasses (e.g., 4 outputs if using 3 seen + 1 novel)
- Head B: logits for subclasses (e.g., 88 seen + 1 novel)

We evaluate three backbones:
1. CNN Baseline (small 3-block ConvNet, trained from scratch)
2. ResNet-18 (ImageNet-pretrained)
3. EfficientNet-B0 (ImageNet-pretrained)

For transfer models we compare:
- Linear probing (freeze backbone; train only the heads / small projection)
- Partial fine-tuning (unfreeze the final block)
- Full fine-tuning (unfreeze all layers)

Loss is CE_super + CE_sub. We apply weight decay and Mixup (using a Beta-distributed mixing coefficient, often clamped to lam = max(lam, 1-lam) to avoid extreme mixes). 
The modeling choices and rationale follow the report’s methodology and align with best practices for small, fine-grained datasets

## Training

Open the notebook and run cells sequentially:
```bash
jupyter notebook notebooks/NNDL_Final_Project.ipynb
```
Inside the notebook you can select a backbone and a fine-tuning regime. Typical optimizer settings are Adam with a small learning rate (e.g., 1e-4) and weight decay (e.g., 1e-4). 
The notebook logs per-epoch training/validation loss and accuracy for both heads and plots curves for quick comparison across strategies.

## Evaluation & Results

Validation tracks:
- Superclass accuracy (coarse categories)
- Subclass accuracy (fine-grained categories)
- Seen vs. Novel accuracy (using sets derived from training labels)

The study consistently found that full fine-tuning of pretrained backbones yields the strongest results on fine-grained subclass recognition (pdf in docs/)
In particular, ResNet-18 (full) reached ~88.7% subclass accuracy, outperforming all other variants; EfficientNet-B0 (full) also performed strongly at ~80.6% subclass accuracy. The CNN baseline trained from scratch achieved a competitive ~76.2% on subclasses. Linear probing underperformed substantially (e.g., ResNet-18 and EfficientNet linear variants below ~40% subclass accuracy), indicating that feature adaptation is crucial at this resolution and label granularity. Superclass accuracy saturated across models (≥96% and up to ~99%), confirming that coarse categories are comparatively easier than fine-grained subclasses. Because the training split did not include truly novel categories, novel-class accuracies were near 0% on validation, as expected. 
Novelty detection is therefore evaluated primarily through the confidence-threshold routing logic at test time.
