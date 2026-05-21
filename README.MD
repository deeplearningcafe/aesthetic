# Anime Aesthetic Classifier & ELO Ranking Toolkit

A highly optimized pipeline for curation, annotation, feature caching,
evaluation, and ELO ranking of anime-style datasets.

---

## Architecture Overview

```
 ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
 │  Step 1: Sample │ ──> │ Step 2: Annotate│ ──> │  Step 3: Cache  │
 │ (run_sampling)  │     │ (run_labeling)  │     │ (Gradio/Script) │
 └─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
 ┌─────────────────┐     ┌─────────────────┐              │
 │  Step 6: Rank   │ <── │ Step 5: Train   │ <────────────┘
 │   (run_elo)     │     │ (run_training)  │
 └─────────────────┘     └─────────────────┘
```

The system employs a two-tier aesthetic evaluation system:
1. **Linear Classifier (Pre-Filter):** An $O(N)$ multi-class MLP trained to
   categorize images into four buckets: `Worst` (0), `Worse` (1), `Better`
   (2), and `Masterpiece` (3). Used to drop low-quality images.
2. **Pairwise ELO Arena (Ranker):** A pairwise comparison classifier trained
   on head-to-head human/synthetic selections. Simulates a Swiss-system
   tournament on high-quality candidates to generate a continuous gradient
   of aesthetic scores.

---

## Repository Structure

The package uses a modern modular structure utilizing the `src/` layout:

```text
├── configs/
│   └── config.yaml             # Shared dataset & pipeline parameters
├── aesthetic/
│   ├── scripts/                # Entrypoints for execution
│   │   ├── run_sampling.py     # Curation & downloading
│   │   ├── run_labeling.py     # Modular Gradio App
│   │   ├── run_training.py     # Unified MLP Trainer
│   │   └── run_elo.py          # ELO Pipeline Phases 1-5
│   ├── setup.py                # Package installer
│   └── src/
│       └── aesthetic/
│           ├── elo/            # Tournament & analysis engines
│           ├── labeling/       # UI/UX elements and tab managers
│           ├── sampling/       # Curation adapters and filters
│           └── training/       # Datasets, architectures, trainers
```

---

## Installation

Install the repository in editable mode along with its dependencies:

```bash
pip install -e .
```

---

## Workflow Guide

### Step 1: Sampling & Curation

Filter and prepare raw metadata from Danbooru parquets to create balanced
preliminary candidate sets:

```bash
python aesthetic/scripts/run_sampling.py \
    --config configs/config.yaml \
    --total_samples 200000
```

---

### Step 2: Multi-Paradigm Labeling

Launch the unified annotation suite, featuring keyboard shortcuts (`1-4` for
single classes, `A`/`D` for pairwise selections) and integrated caching:

```bash
python aesthetic/scripts/run_labeling.py \
    --images_folder "/your-data" \
    --final_tiers_csv "reports/final_tiers.csv" \
    --prior_knowledge_csv "cleaned_prior_knowledge.csv" \
    --pairs_output "aesthetic/labels_pairs.json"
```

The interface exposes four specialized operational modules:
- **Single Labeling:** Map raw images into one of the four aesthetic bins.
- **Pair Labeling (Elo):** Select between side-by-side candidates.
- **Data Preparation (Cache):** Pre-extract and cache SwinV2 features.
- **Soft Pairs Generation:** Programmatically pair class 1 and class 3 images.

---

### Step 3: Feature Caching

To avoid loading the Vision Transformer repeatedly, pre-extract and cache the
SwinV2 embeddings ($1024$-dimensional vectors) directly to HDF5 shards:

```bash
python -c '
from aesthetic.training.cache import FeatureCacher
cacher = FeatureCacher(model_dir="model/wd_swinv2")
cacher.cache_dataset(
    json_path="aesthetic/labels_pairs.json",
    output_h5="aesthetic_output/cached_features.h5",
    output_meta="aesthetic_output/cached_meta.json",
    global_path="//Data/evangelion",
    batch_size=32
)
'
```

---

### Step 4: Model Training

Train either the four-class gating classifier or the pairwise discriminator on
top of your cached embeddings:

```bash
# Train the Pairwise Classifier
python aesthetic/scripts/run_training.py \
    --h5_train aesthetic_output/cached_features.h5 \
    --meta_train aesthetic_output/cached_meta.json \
    --pairs_json aesthetic/labels_pairs.json \
    --model_type pair \
    --epochs 30

# Train the Four-Class Classifier
python aesthetic/scripts/run_training.py \
    --h5_train aesthetic_output/cached_features.h5 \
    --meta_train aesthetic_output/cached_meta.json \
    --model_type four_class \
    --epochs 30
```

---

### Step 5: The ELO Pipeline

Evaluate unlabeled candidates through the complete ELO workflow:

```bash
python aesthetic/scripts/run_elo.py \
    --phases 1 2 3 4 5 \
    --input_csv "image_aesthetic_labels.csv" \
    --pair_model "models/pair_classifier_best.pth" \
    --threshold 1600.0 \
    --rounds 100
```

The execution loop processes five sequential stages:
1. **Phase 1 (Pre-Filter):** Filters out low-scoring metadata (scores < 2).
2. **Phase 2 (Caching):** Extracts features of the high-quality subset.
3. **Phase 3 (Arena Simulation):** Initializes player scores at $1200$ and
   runs $K$-rounds of Swiss-system pairings. Match results are determined
   by pairwise model prediction probabilities.
4. **Phase 4 (Binning & Thresholding):** Extracts selected samples that
   cross the defined ELO cutoff (e.g., $>1600$).
5. **Phase 5 (Visual Diagnosis):** Automatically renders diagnostic plots:
   - `quantiles_grid.png`: Displays samples grouped by ELO quantiles.
   - `similar_elos.png`: Shows neighbor groupings with tight ratings.
   - `elo_gaps.png`: Pairs high-scoring masterpieces with low-scoring inputs.
```

---

## Notes

*   **Embeddings Choice**: The feature extraction pipeline uses a SwinV2 model used for tagging danbooru images. You can adapt it to use other feature extractors by modifying `load_feature_extractor` and ensuring the feature processing pipeline is compatible.

## References
- Danbooru2024 dataset by p1atdev :https://huggingface.co/datasets/p1atdev/danbooru-2024
- Swinv2 tagger by SmilingWolf :https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3
- The idea is based on the aesthetic classifier of waifu diffusion project: https://github.com/waifu-diffusion/aesthetic

## Author

[aipracticecafe-codeberg](https://codeberg.org/aipracticecafe)
[aipracticecafe-github](https://github.com/deeplearningcafe)


## License
This project is licensed under the MIT license. Details are in the [LICENSE](LICENSE.txt) file. I don't own the danbooru dataset nor the tagger model, its license can be found in each respective huggingface page.
