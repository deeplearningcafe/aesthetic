# Anime Aesthetic Classifier Toolkit

## Table of Contents

- [Features](#Features)
- [Repository Structure](#Repository-Structure)
- [Prerequisites](#Prerequisites)
- [Workflow and Usage](#Workflow-and-Usage)
- [Customization and Notes](#Customization-and-Notes)
- [References](#References)

This repository provides a set of tools to help you create an aesthetic classifier for anime-style images. It includes scripts for:
1.  **Filtering and Sampling Data**: Processing a large dataset (e.g., Danbooru) to select a diverse and manageable subset of images for labeling, based on scores, favorite counts, and ratings.
2.  **Image Labeling**: A Gradio-based web application for manually labeling images into four aesthetic categories.
3.  **Classifier Training**: Extracting image features using a SwinV2 model and training a Multi-Layer Perceptron (MLP) classifier on these features.

The primary goal is to facilitate the creation of a labeled dataset and train a model that can predict aesthetic scores for new images. The tools are designed to be relatively self-contained.

## Features

*   **Data Filtering (`filtering.py`)**:
    *   Filters Danbooru-style datasets from Parquet files.
    *   Removes deleted/banned images and MD5 duplicates.
    *   Handles parent/child image relationships.
    *   Allows exclusion of images based on an external ID list.
    *   Probabilistic skipping of images based on specified tags.
    *   Samples images into 4 preliminary aesthetic buckets ("worst", "worse", "better", "best") based on `score` and `fav_count` percentiles to ensure a diverse set for labeling.
    *   Distributes samples within buckets according to SFW ratings (general, sensitive, questionable, explicit).
    *   Advanced sampling to improve diversity (parent group aware, ID proximity).
    *   Downloads sampled images into class-structured folders.
*   **Image Labeling (`labeling_app.py`)**:
    *   User-friendly Gradio interface for labeling.
    *   Four aesthetic classes: "worst" (0), "worse" (1), "better" (2), "best" (3).
    *   Keyboard shortcuts (1-4) for faster labeling.
    *   Saves labels progressively to a CSV file (`image_path`, `label`).
    *   Resumes labeling from the last saved point.
    *   Tracks labeling progress and time statistics.
    *   Utility to migrate labels from an old JSON format.
    *   Utility to export final CSV labels to a JSON format (compatible with the training script).
*   **Classifier Training (`aesthetic.py`)**:
    *   Loads a pre-trained SwinV2 model for feature extraction.
    *   Preprocesses images (resize, pad, normalize, BGR conversion).
    *   Supports data augmentation during feature extraction (creating multiple versions per image).
    *   Caches extracted features efficiently in HDF5 format with JSON metadata.
    *   Trains an MLP classifier on the extracted features.
    *   Supports feature-level augmentations during training (Mixup, Gaussian noise).
    *   Handles class imbalance with class weights.
    *   Implements early stopping based on validation F1-score.
    *   Saves the trained model and training history.

## Repository Structure

*   `filtering.py`: Script for dataset filtering and image downloading.
*   `labeling_app.py`: Gradio application for image labeling.
*   `aesthetic.py`: Script for feature extraction and classifier training.
*   `add_folder_labels.py`: Script for adding images from a folder to the train json file labeling all of the images with the same class.
*   `model/wd_swinv2/` (Example): Directory where you should place the SwinV2 model files (`model.safetensors`, `config.json`).
*   `data/` (Example): Directory where `filtering.py` might download images and `labeling_app.py` might read them from.
*   `aesthetic/` (Example): Directory where `aesthetic.py` might save features, metadata, and the trained classifier.

## Prerequisites

*   Python 3.8+
*   Key Python libraries:
    *   `pandas`
    *   `numpy`
    *   `gradio`
    *   `torch`, `torchvision`
    *   `timm`
    *   `h5py`
    *   `Pillow` (PIL)
    *   `scikit-learn`
    *   `requests`
    *   `transformers` (for image_utils, image_transforms)
    It's recommended to set up a virtual environment and install dependencies.
*   **SwinV2 Model**: For `aesthetic.py`, you need the SwinV2 model files (`model.safetensors` and `config.json`). You can obtain these from Hugging Face, the model I found best that can be runned easily with 6GB VRAM is `https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3`. Place them in a directory (e.g., `model/wd_swinv2/` as used by default in `aesthetic.py`).
Thanks to `https://huggingface.co/SmilingWolf` for sharing this booru tagger models.
*   **Danbooru dataset**: For `filtering_v2.py`, you need the parquet files (`train-00034-of-00035.parquet`). You can obtain these from Hugging Face at `https://huggingface.co/datasets/p1atdev/danbooru-2024`. Place them inside the repository as used by default in `filtering_v2.py`.
Thanks to `https://huggingface.co/p1atdev` for sharing this danbooru dataset in dataframe format.

## Workflow and Usage

### Step 1: Data Filtering and Downloading (`filtering.py`)

This script prepares a dataset for labeling by filtering a large image collection and downloading a selected subset.

1.  **Prepare your data**: You'll need a Danbooru-like dataset, with columns like `id`, `file_url`, `score`, `fav_count`, `rating`, `tag_string`.
2.  **Configure `filtering.py`**:
    *   Open `filtering.py` and modify the `if __name__ == "__main__":` block.
    *   Set `df = pd.read_parquet("your_dataset.parquet")` to load your data.
    *   Optionally, provide an `exclude_df` if you have a CSV of image IDs to exclude. Useful in case you want to sample from the same dataframe.
    *   Adjust `total_images`, `random_seed`, `tag_to_skip`, and `custom_rating_percentages` as needed.
    *   Set `download = True` and `output_dir` (e.g., `"data/my_anime_dataset"`) for downloading images. Recommended to first set `download = False` to check if the buckets have the desired distributions, as in case of running out of samples from the desired rating, other ratings are used.
3.  **Run the script**:
    ```bash
    python filtering.py
    ```
4.  **Output**:
    *   A CSV file (e.g., `filtered_data_train_xx_v2.csv`) containing information about the sampled images, including an `aesthetic_class` column (0-3) representing the initial bucket.
    *   Downloaded images in `output_dir/{aesthetic_class}/`, e.g., `data/my_anime_dataset/0/image1.jpg`, `data/my_anime_dataset/1/image2.jpg`, etc. These `aesthetic_class` subfolders (0-3) are based on the initial `score`/`fav_count` bucketing.

### Step 2: Image Labeling (`labeling_app.py`)

Use this Gradio app to manually assign the final aesthetic labels (0-3) to the images prepared in Step 1.

1.  **Configure `labeling_app.py`**:
    *   Open `labeling_app.py` and modify the `if __name__ == "__main__":` block.
    *   Set `IMAGES_FOLDER` to the `output_dir` used in `filtering.py` (e.g., `"./data/my_anime_dataset"`). The app expects images to be in subfolders like `IMAGES_FOLDER/0/`, `IMAGES_FOLDER/1/`, etc.
    *   Set `OUTPUT_CSV_FILE` to where you want to save your labels (e.g., `"aesthetic_labels.csv"`).
    *   If you have labels in an old JSON format, set `OLD_JSON_FILE` to its path for migration. Otherwise, set it to `None`.
2.  **Run the app**:
    ```bash
    python labeling_app.py
    ```
    The app will launch in your browser (usually at `http://127.0.0.1:1234`).
3.  **Labeling**:
    *   Images will be displayed one by one.
    *   Click "Worst" (0), "Worse" (1), "Better" (2), or "Best" (3) to assign a label.
    *   Labels are saved to the CSV file as you go. You can stop and resume later.
4.  **Output**:
    *   A CSV file (e.g., `aesthetic_labels.csv`) with columns `image_path` and `label` (the manually assigned 0-3 score).
    *   When you close the app (Ctrl+C in terminal), it will also export the labels to a JSON file with a `.final.json` suffix (e.g., `aesthetic_labels.final.json`). This JSON file is used by the training script.

### Step 3: Feature Extraction and Classifier Training (`aesthetic.py`)

This script first extracts features from your labeled images and then trains the aesthetic classifier.

**Prerequisites for this step:**
*   Ensure you have the SwinV2 model files (`model.safetensors`, `config.json`) in the directory specified by `--model_dir` (default: `model/wd_swinv2/`).
*   You need the `.final.json` label file generated by `labeling_app.py`.

**3.a. Feature Extraction**

This process extracts deep features from images and caches them.

1.  **Command**:
    ```bash
    python aesthetic.py \
        --json_path path/to/your/labels.final.json \
        --model_dir path/to/your/swinv2_model_folder \
        --output_dir ./aesthetic_output \
        --global_path ./data/my_anime_dataset \
        --extract_batch_size 32 \
        --num_workers 4 \
        --device cuda \
        --num_augmentations 1 
    ```
    *   `--json_path`: Path to the `.final.json` file from `labeling_app.py`.
    *   `--model_dir`: Directory containing SwinV2 `model.safetensors` and `config.json`.
    *   `--output_dir`: Where HDF5 features and metadata will be saved.
    *   `--global_path`: If image paths in your JSON are relative (e.g., `0/image.jpg`), this should be the base path to them (e.g., the `IMAGES_FOLDER` used in `labeling_app.py`). If paths in JSON are absolute or directly usable from the script's CWD, this might not be strictly needed or can be `.`.
    *   `--num_augmentations`: Set to `1` for no augmentation during extraction, or `>1` to extract features for the original image plus augmented versions. Which are saved in the h5 file as new samples.
    *   Use `--skip_extraction` if features (`cached_features_train.h5`, `cached_features_meta_train.json`) already exist in `--output_dir` and you want to reuse them (e.g., for re-training the classifier).

2.  **Output**:
    *   `./aesthetic_output/cached_features_train.h5`: HDF5 file with image features.
    *   `./aesthetic_output/cached_features_meta_train.json`: JSON metadata linking image keys to HDF5 indices and labels.
    *   (If you prepare separate validation data, you'd run this step for your validation JSON as well, outputting `cached_features_val.h5` and `cached_features_meta_val.json`).

**3.b. Classifier Training**

This trains the MLP on the cached features.

1.  **Command**:
    ```bash
    python aesthetic.py \
        --json_path path/to/your/labels.final.json \
        --model_dir path/to/your/swinv2_model_folder \
        --output_dir ./aesthetic_output \
        --global_path ./data/my_anime_dataset \
        --skip_extraction \
        --train_batch_size 1024 \
        --epochs 100 \
        --lr 1e-4 \
        --wd 1e-3 \
        --dropout 0.3 \
        --hidden_dims 512 \
        --num_classes 4 \
        --patience 15 \
        --use_mixup \
        --use_noise \
        --noise_std 0.05 \
        --num_augmentations 1 # Should match what was used for extraction if reusing features
    ```
    *   Most path arguments are the same as for extraction.
    *   `--skip_extraction` is crucial here to use the features generated in 3.a.
    *   Adjust training hyperparameters (`--train_batch_size`, `--epochs`, `--lr`, etc.) as needed.
    *   `--num_augmentations` here should typically match what was used during feature extraction if you are using those augmented features for training. If feature extraction created N augmentations, the dataset will be N times larger.
    *   **Validation**: The script expects pre-extracted HDF5/JSON files for validation data if used (named `cached_features_val.h5` and `cached_features_meta_val.json` in the `--output_dir`). You would need to prepare a separate validation JSON from your labeling efforts and run the feature extraction (3.a) on it.
    *   **Class Weights**: The script attempts to calculate class weights from the training data. You can see this logic in `aesthetic.py`'s `main()` function.

2.  **Output**:
    *   `./aesthetic_output/aesthetic_classifier.pth`: The trained PyTorch model.
    *   `./aesthetic_output/aesthetic_classifier_history.json`: JSON file with training metrics (loss, accuracy, F1).

## Customization and Notes

*   **Aesthetic Classes**: The system is built around 4 classes: "worst" (0), "worse" (1), "better" (2), "best" (3). If you need different classes, you'll need to modify `labeling_app.py` and `aesthetic.py` (especially `num_classes`).
*   **Dataset Format**: `filtering.py` is tailored for Danbooru-like datasets. You might need to adapt it for other sources.
*   **Model Choice**: `aesthetic.py` uses a SwinV2 model. You can adapt it to use other feature extractors by modifying `load_feature_extractor` and ensuring the feature processing pipeline is compatible.
*   **Hyperparameters**: The provided commands include example hyperparameters. You'll likely need to tune these for your specific dataset and goals.
*   **File Paths**: Pay close attention to file paths, especially the `global_path` argument in `aesthetic.py` and the `IMAGES_FOLDER` in `labeling_app.py`, to ensure images are correctly located.

## References
- Danbooru2024 dataset by p1atdev :https://huggingface.co/datasets/p1atdev/danbooru-2024
- Swinv2 tagger by SmilingWolf :https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3
- The idea is based on the aesthetic classifier of waifu diffusion project: https://github.com/waifu-diffusion/aesthetic

## Author
[aipracticecafe](https://github.com/deeplearningcafe)

## License
This project is licensed under the MIT license. Details are in the [LICENSE](LICENSE.txt) file. I don't own the danbooru dataset nor the tagger model, its license can be found in each respective huggingface page.