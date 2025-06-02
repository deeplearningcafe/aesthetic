import os
import json
import h5py
import torch
import numpy as np
import timm
from tqdm import tqdm
from PIL import Image, ImageFile
from typing import Dict, List, Tuple, Optional, Generator, Union
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.transforms import v2
from torch import optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
)
from transformers.image_transforms import (
    rescale,
    to_channel_dimension_format,
    normalize,
)
import random
torch.manual_seed(46)
random.seed(46)
np.random.seed(46)


import logging
import sys
from contextlib import contextmanager
import argparse
import gc

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_IMG_SIZE = (448, 448)
DEFAULT_MODEL_DIR = 'model/wd_swinv2'
DEFAULT_OUTPUT_DIR = 'aesthetic/latents'
DEFAULT_CLASSIFIER_HIDDEN_DIMS = 512
DEFAULT_CLASSIFIER_DROPOUT = 0.1
DEFAULT_GLOBAL_PATH =  'FOLDER-PATH'
DEFAULT_NUM_AUGMENTATIONS = 1
# --- Utility Functions ---

@contextmanager
def open_h5_file(file_path: str, mode: str = 'r') -> Generator[h5py.File, None, None]:
    """Context manager for safely opening and closing H5 files."""
    h5_file = None
    try:
        # Ensure directory exists for write modes
        if mode in ['w', 'a', 'w-', 'x']:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        h5_file = h5py.File(file_path, mode)
        yield h5_file
    except Exception as e:
        logger.error(f"Error opening H5 file {file_path} in mode {mode}: {e}")
        raise # Re-raise the exception after logging
    finally:
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception as e:
                logger.error(f"Error closing H5 file {file_path}: {e}")


# --- Model Loading ---

def load_feature_extractor(
    model_dir: str
) -> Tuple[Optional[nn.Module], Optional[Dict], Optional[Tuple[int, int]]]:
    """
    Loads the Swin V2 feature extractor model and its config.

    Args:
        model_dir: Directory containing model.safetensors and config.json.

    Returns:
        Tuple of (model, config, input_size_hw) or (None, None, None).
    """
    checkpoint_path = os.path.join(model_dir, 'model.safetensors')
    config_path = os.path.join(model_dir, 'config.json')

    if not all(os.path.exists(p) for p in [model_dir, checkpoint_path, config_path]):
        logger.error(f"Model directory, checkpoint, or config not found in {model_dir}")
        return None, None, None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded config from {config_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {config_path}")
        return None, None, None
    except IOError as e:
        logger.error(f"Error reading config file {config_path}: {e}")
        return None, None, None

    model_name = config.get('architecture', 'swinv2_base_window8_256')
    num_classes = config.get('num_classes', 0) # Original classes, not ours
    model_kwargs = config.get('model_args', {})
    input_size_cfg = config.get('pretrained_cfg', {}).get('input_size')

    if not isinstance(input_size_cfg, (list, tuple)) or len(input_size_cfg) != 3:
        logger.warning(f"Invalid input_size in config: {input_size_cfg}. Using default.")
        input_size_hw = DEFAULT_IMG_SIZE
    else:
        # Assuming format is (C, H, W)
        input_size_hw = (input_size_cfg[1], input_size_cfg[2])

    logger.info(f"Attempting to load model: {model_name}")
    try:
        model = timm.create_model(
            model_name,
            checkpoint_path=checkpoint_path,
            num_classes=num_classes, # Load with original head, we only use features
            **model_kwargs
        )
        model.eval() # Set to evaluation mode
        logger.info("Feature extractor model loaded successfully.")
        return model, config, input_size_hw
    except Exception as e:
        logger.error(f"Error creating model {model_name}: {e}", exc_info=True)
        return None, None, None

def get_feature_dimension(model: nn.Module, input_size_hw: Tuple[int, int]) -> int:
    """Gets the output dimension of the feature extractor."""
    try:
        dummy_input = torch.randn(1, 3, input_size_hw[0], input_size_hw[1])
        device = next(model.parameters()).device # Use model's current device
        dummy_input = dummy_input.to(device)
        with torch.no_grad():
            features = model.forward_features(dummy_input)
            pooled_features = model.head.global_pool(features)
        return pooled_features.shape[1]
    except Exception as e:
        logger.error(f"Could not determine feature dimension: {e}")
        raise

def resize_with_padding(
    image: np.ndarray,
    size: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    resample = PILImageResampling.BILINEAR,
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    """
    Resizes image while maintaining aspect ratio and padding to fill target size.
    Correctly handles BGR conversion expected by the model.
    
    Args:
        image: Input image as numpy array (assumed RGB initially if from PIL)
        size: Target size as (height, width)
        color: Background color for padding as RGB tuple
        resample: PIL resampling filter
        data_format: Output channel dimension format
        input_data_format: Input channel dimension format
        
    Returns:
        Resized, padded, and BGR-converted image as numpy array
    """
    # For transformations, keep same data format as input unless specified
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    # Convert to PIL for resizing if it's a numpy array
    # Keep track if we need to rescale back later
    do_rescale_back = False
    if not isinstance(image, Image.Image):
        # Check if image needs rescaling for PIL conversion (0-1 range)
        if is_scaled_image(image):
            do_rescale_back = True
            image = image * 255
        # Ensure it's uint8 for PIL
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        # Create PIL image, inferring mode if necessary
        image = Image.fromarray(image)

    # Get original dimensions
    original_width, original_height = image.size
    height, width = size

    # Calculate aspect ratio for resize
    ratio = min(width / original_width, height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    # Resize maintaining aspect ratio
    resized_image = image.resize(
        (new_width, new_height), resample=resample
    )

    # Create new image with solid background (using RGBA for paste)
    new_image_rgba = Image.new("RGBA", (width, height), color + (255,))

    # Paste resized image at the center
    offset = ((width - new_width) // 2, (height - new_height) // 2)
    # Ensure resized image is RGBA for pasting with alpha
    resized_image_rgba = resized_image.convert("RGBA")
    new_image_rgba.paste(resized_image_rgba, offset, resized_image_rgba)

    # Convert final image back to RGB (discard alpha)
    new_image_rgb = new_image_rgba.convert("RGB")

    # Convert PIL image (RGB) to numpy array (float32)
    image_array = np.asarray(new_image_rgb, dtype=np.float32)
    
    # *** CHANGE START: Added BGR conversion ***
    # Convert RGB to BGR as expected by the model's original preprocessing
    image_array = image_array[:, :, ::-1]
    # *** CHANGE END ***

    # Add channel dimension if needed (e.g., for grayscale - unlikely here)
    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=-1)
    
    # Convert to desired channel format (e.g., channels_first for PyTorch)
    # Input is now channels_last after numpy conversion and BGR swap
    image_array = to_channel_dimension_format(
        image_array, data_format, input_channel_dim=ChannelDimension.LAST
    )
    
    # Restore original scale (0-1) if needed
    if do_rescale_back:
        image_array = image_array / 255.0

    return image_array


# --- Datasets ---

class ImagePathDataset(Dataset):
    """
    Dataset for loading image paths and labels from a JSON file.
    Applies data augmentation to create multiple versions of each image.
    """
    def __init__(
        self,
        json_path: str,
        processor_config: Dict,
        global_path: str=None,
        num_augmentations: int = 1, 
    ):
        self.json_path = json_path
        self.global_path = global_path
        # self.transform = transform
        # self.img_size = img_size # Store for error placeholder
        self.size_dict = processor_config["size"]
        self.color_tuple = tuple(processor_config["color"])
        self.image_mean = processor_config["image_mean"]
        self.image_std = processor_config["image_std"]
        self.rescale_factor = processor_config["rescale_factor"]
        self.resample_filter = processor_config["resample"]
        self.num_augmentations = max(1, num_augmentations)

        try:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
            self.image_paths = list(self.data.keys())
            logger.info(f"Loaded {len(self.image_paths)} image paths from {json_path}")
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {json_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data from {json_path}: {e}")
            raise

        if self.num_augmentations > 1:
            self.augmentation_transform = v2.Compose([
            # Geometric transformations
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15),
            # Affine includes scale, translate, shear. Keep moderate.
            v2.RandomAffine(
                degrees=0, translate=(0.075, 0.075), scale=(0.9, 1.1),
                shear=10
            ),
            # Perspective distortion
            v2.RandomPerspective(
                distortion_scale=0.15, p=0.3,
                interpolation=v2.InterpolationMode.BILINEAR
            ),
            # Color transformations (applied with probability)
            v2.RandomApply([
                v2.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                )
            ], p=0.8),
            # Grayscale (low probability)
            v2.RandomGrayscale(p=0.1),
            # Blurring (applied with probability)
            v2.RandomApply([
                v2.GaussianBlur(kernel_size=3)
            ], p=0.3),
            # Add more transforms here if needed, e.g., RandomErasing
            v2.RandomErasing(p=0.2, scale=(0.02, 0.75), ratio=(0.5, 2)),
            ])
        else:
            self.augmentation_transform = None 


    def preprocess_image(self, image):
        
        image_array = np.array(image.convert("RGB"))

        # --- Preprocessing Pipeline ---
        # 1. Resize with padding and convert to BGR numpy array
        #    Output format will be channels_first for PyTorch
        processed_image = resize_with_padding(
            image=image_array,
            size=(self.size_dict["height"], self.size_dict["width"]),
            color=self.color_tuple,
            resample=self.resample_filter,
            data_format=ChannelDimension.FIRST # PyTorch expects channels first
        )
        
        # 2. Rescale values to [0,1] if they aren't already
        #    resize_with_padding handles rescaling back if input was 0-1,
        #    so here we ensure it's rescaled *from* 0-255 to 0-1 if needed.
        if not is_scaled_image(processed_image):
            processed_image = rescale(
                processed_image,
                scale=self.rescale_factor,
                data_format=ChannelDimension.FIRST
            )
        
        # 3. Normalize with mean and std
        processed_image = normalize(
            processed_image, 
            mean=self.image_mean,
            std=self.image_std,
            data_format=ChannelDimension.FIRST
        )
        
        # 4. Convert final processed numpy array to PyTorch tensor
        #    Add batch dimension and move to target device
        img_tensor = torch.tensor(processed_image).float()#.unsqueeze(0).to(device)
        return img_tensor

    def __len__(self) -> int:
        # The total length is the number of original images times the
        # number of versions (original + augmentations) per image.
        return len(self.image_paths) * self.num_augmentations

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int, str]]:
        """
        Returns (image_tensor, label, img_path) or None if error.
        Applies augmentation based on the index.
        """
        if idx >= self.__len__():
            raise IndexError("Dataset index out of range.")

        # Determine which original image and which augmentation this index refers to
        original_idx = idx // self.num_augmentations
        augmentation_idx = idx % self.num_augmentations # 0 is original

        img_path = self.image_paths[original_idx]
        label = self.data[img_path]

        img_path_open= img_path
        if self.global_path:
            img_path_corrected = img_path.replace('\\', os.sep)
            img_path_open = os.path.join(
                                self.global_path, img_path_corrected
                            )


        if not os.path.exists(img_path_open):
            logger.warning(f"Image file not found: {img_path_open}. Skipping.")
            # Return None or raise error? Returning None allows DataLoader to skip.
            # We need to return something the collate_fn can handle,
            # or implement a custom collate_fn.
            # For simplicity, let's return None and handle in extraction loop.
            # However, DataLoader default collate doesn't like None.
            # A better approach is to return a placeholder or skip in the loop.
            # Let's assume the extraction loop handles potential None returns.
            # *Correction*: Let's raise an error here, should be handled upstream
            # or filter missing files beforehand. For now, log and return None.
            return None # Signal an error

        try:
            image = Image.open(img_path_open).convert('RGB')
            # Apply augmentation if this is not the original version (aug_idx > 0)
            augmented_image = image
            if augmentation_idx > 0 and self.augmentation_transform:
                try:
                    augmented_image = self.augmentation_transform(image)
                except Exception as aug_e:
                    logger.warning(
                        f"Error applying augmentation {augmentation_idx} to "
                        f"{img_path}: {aug_e}. Using original."
                    )
                    # Fallback to original if augmentation fails
                    augmented_image = image


            # Apply the standard preprocessing (resizing, normalization, etc.)
            # to the original or the augmented image.
            image_tensor = self.preprocess_image(augmented_image)

            # Create a unique key for metadata: append suffix for augmented versions
            unique_img_key = img_path
            if augmentation_idx > 0:
                unique_img_key = f"{img_path}_aug_{augmentation_idx}"

            return image_tensor, label, unique_img_key

        except (IOError, OSError, Image.DecompressionBombError) as e:
            logger.warning(f"Error loading/processing image {img_path}: {e}. Skipping.")
            return None # Signal an error
        except Exception as e:
            logger.error(
                f"Unexpected error processing image {img_path}: {e}",
                exc_info=True
            )
            return None # Signal an error

class FeatureDataset(Dataset):
    """
    Dataset for loading pre-extracted features from a single HDF5 dataset,
    indexed by a JSON metadata file.
    """
    def __init__(self, h5_path: str, meta_path: str):
        self.h5_path = h5_path
        self.meta_path = meta_path

        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        try:
            with open(meta_path, 'r') as f:
                metadata = json.load(f)

            # --- Updated Metadata Handling ---
            if "dataset_info" not in metadata or \
               "sample_mapping" not in metadata:
                raise ValueError(
                    "Metadata JSON format incorrect. Missing "
                    "'dataset_info' or 'sample_mapping'."
                )

            self.h5_feature_dataset_path = metadata["dataset_info"].get(
                "h5_feature_path", "features" # Default to "features" if missing
            )
            self.sample_mapping = metadata["sample_mapping"]
            self.image_keys = list(self.sample_mapping.keys())
            if not self.image_keys:
                logger.warning(f"Metadata file {meta_path} contains no samples.")
            else:
                logger.info(
                    f"Loaded metadata for {len(self.image_keys)} features. "
                    f"H5 dataset path: '{self.h5_feature_dataset_path}'"
                )

        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {meta_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading metadata from {meta_path}: {e}")
            raise

        # Optional: Verify H5 file contains the dataset path on init
        try:
            with open_h5_file(self.h5_path, 'r') as h5_file:
                if self.h5_feature_dataset_path not in h5_file:
                    raise KeyError(
                        f"HDF5 file '{self.h5_path}' does not contain "
                        f"dataset '{self.h5_feature_dataset_path}'."
                    )
                # Check if number of samples matches metadata (optional)
                h5_len = h5_file[self.h5_feature_dataset_path].shape[0]
                meta_len = len(self.image_keys)
                if h5_len != meta_len:
                    logger.warning(
                        f"H5 dataset '{self.h5_feature_dataset_path}' length "
                        f"({h5_len}) does not match metadata length ({meta_len})."
                    )
        except Exception as e:
            logger.error(f"Error verifying HDF5 file structure: {e}")
            # Decide if this should be a fatal error
            # raise # Uncomment to make it fatal


    def __len__(self) -> int:
        return len(self.image_keys)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int]]:
        """Returns (feature_tensor, label) or None if error."""
        if idx >= len(self.image_keys):
            raise IndexError("Index out of range")

        
        img_key = self.image_keys[idx]
        meta_info = self.sample_mapping.get(img_key)

        if meta_info is None:
            logger.warning(f"Metadata not found for key: {img_key}. Skipping.")
            return None # Indicate failure to load

        try:
            h5_index = meta_info['h5_index']
            label = meta_info['label'] # Assuming label is directly usable

            # Open H5 file here for thread/process safety with DataLoader workers
            with open_h5_file(self.h5_path, 'r') as h5_file:
                # Access the main dataset using the path from metadata
                features_dataset = h5_file[self.h5_feature_dataset_path]
                # Retrieve the specific feature vector by its index
                feature_np = features_dataset[h5_index]
                # Convert numpy array to torch tensor
                feature = torch.from_numpy(feature_np)

            # Ensure label is in a usable format (e.g., int or tensor)
            # This might need adjustment based on how labels are used later
            if isinstance(label, list): # Example: handle one-hot labels if needed
                 label_tensor = torch.tensor(label, dtype=torch.float32)
                 return feature, label_tensor
            else: # Assume integer label
                 return feature, int(label)

        except KeyError as e:
            logger.warning(
                f"KeyError accessing data for key {img_key}: {e}. "
                f"Maybe missing 'h5_index' or dataset path "
                f"'{self.h5_feature_dataset_path}' incorrect?"
            )
            return None # Indicate failure
        except IndexError:
             logger.warning(
                 f"H5 index {h5_index} out of bounds for key {img_key} in "
                 f"dataset '{self.h5_feature_dataset_path}'. "
                 f"H5 file might not match metadata."
             )
             return None # Indicate failure
        except Exception as e:
            logger.error(
                f"Error reading feature index {h5_index} (key: {img_key}) "
                f"from {self.h5_path} dataset "
                f"'{self.h5_feature_dataset_path}': {e}"
            )
            return None # Indicate failure


# --- Feature Extraction ---

def extract_and_cache_features(
    model: nn.Module,
    json_path: str,
    output_h5_path: str,
    output_meta_path: str,
    processor_config: Dict,
    batch_size: int = 32,
    device: str = 'cpu',
    num_workers: int = 4,
    global_path: str=None,
    h5_compression: Optional[str] = "gzip", 
    num_augmentations: int = 1, 
) -> int:
    """
    Extracts features using a model and caches them efficiently into a single
    resizable HDF5 dataset.

    Args:
        model: The loaded feature extractor model.
        json_path: Path to the input JSON with image paths and labels.
        output_h5_path: Path to save the HDF5 feature file.
        output_meta_path: Path to save the JSON metadata file.
        input_size_hw: Target image size (height, width).
        batch_size: Processing batch size.
        device: The torch device to use for inference.
        num_workers: Number of data loading workers.
        global_path: Path of the folder where the images are stored
        h5_compression: Compression algorithm for HDF5 ('gzip', 'lzf', None).
        num_augmentations: Number of versions (original + aug) per image.

    Returns:
        Number of features successfully cached.
    """
    dataset = ImagePathDataset(
        json_path,
        processor_config=processor_config,
        global_path=global_path,
        num_augmentations=num_augmentations
    )
    # Custom collate_fn to filter out None values from dataset errors
    def collate_fn_filter_none(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: # If the whole batch failed
            return None
        # Default collate behavior for the filtered batch
        return torch.utils.data.dataloader.default_collate(batch)

    # Use custom collate_fn to handle loading errors
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
        collate_fn=collate_fn_filter_none
    )

    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    model = torch.compile(model)

    metadata = {}
    feature_count = 0
    processed_batches = 0
    total_batches = len(dataloader)
    feature_dim = None # Will be determined from the first batch
    h5_features_ds = None # HDF5 dataset handle

    logger.info(f"Starting feature extraction for {len(dataset)} images.")
    logger.info(f"Output H5: {output_h5_path}")
    logger.info(f"Output Meta JSON: {output_meta_path}")

    with open_h5_file(output_h5_path, 'w') as h5f:
        with torch.no_grad():
            pbar = tqdm(dataloader, total=total_batches, desc="Extracting Features")
            for batch in pbar:
                if batch is None:
                    logger.warning(f"Skipping None batch {processed_batches+1}")
                    processed_batches += 1
                    continue

                # Unpack the batch (may be empty if all items failed)
                images, labels, unique_keys = batch

                if images.numel() == 0: # Skip empty batches
                    logger.warning(f"Skipping empty batch {processed_batches+1}")
                    processed_batches += 1
                    continue

                images = images.to(device)

                try:
                    features = model.forward_features(images)
                    pooled_features = model.head.global_pool(features)
                    features_np = pooled_features.cpu().float().numpy()

                    num_in_batch = features_np.shape[0]
                    if num_in_batch == 0: # Should not happen if check above works
                        processed_batches += 1
                        continue

                    # --- HDF5 Dataset Handling (Create or Resize) ---
                    if h5_features_ds is None:
                        # First valid batch: Create the dataset
                        feature_shape = features_np.shape[1:] # Get shape (C,) or (C, H, W) etc.
                        logger.info(f"Detected feature shape: {feature_shape}")
                        h5_features_ds = h5f.create_dataset(
                            "features",
                            # Initial shape is (0, *feature_dims)
                            shape=(0,) + feature_shape,
                            # Max shape allows infinite samples
                            maxshape=(None,) + feature_shape,
                            dtype=np.float32, # Store as float32
                            # Chunking by sample improves read performance later
                            chunks=(1,) + feature_shape,
                            compression=h5_compression
                        )
                        logger.info(f"Created resizable H5 dataset 'features'")
                    # else:
                    current_size = h5_features_ds.shape[0]
                    new_size = current_size + num_in_batch
                    h5_features_ds.resize(new_size, axis=0)

                    # --- Write Data and Update Metadata ---
                    start_index = current_size
                    end_index = current_size + num_in_batch
                    h5_features_ds[start_index:end_index] = features_np

                    for i in range(num_in_batch):
                        img_key = unique_keys[i] # This key includes _aug_ suffix
                        current_h5_index = start_index + i
                        metadata[img_key] = {
                            # Store the index within the H5 dataset
                            "h5_index": current_h5_index,
                            # Ensure label is a standard Python type
                            "label": labels[i].item() if hasattr(labels[i], 'item') else labels[i]
                        }

                    # Update total count *after* successful write
                    feature_count += num_in_batch
                    pbar.set_postfix({"cached": feature_count})

                except Exception as e:
                    logger.error(
                        f"Error processing batch {processed_batches+1}: {e}",
                        exc_info=True # Provides traceback
                    )
                    # Decide how to handle: skip batch, stop? Currently skips.
                    # Note: If error occurs after resize but before write,
                    # the H5 file might have empty allocated space.

                finally:
                    # Ensure batch counter increments even if error occurred
                    processed_batches += 1
                    # Optional: Clear CUDA cache periodically if memory is tight
                    if processed_batches % 50 == 0:
                        # --- Memory Management ---
                        del pooled_features, features, labels
                        del features_np
                        if device == "cuda":
                            torch.cuda.empty_cache()
                        gc.collect() # Optional: Force garbage collection

        # --- Explicitly flush changes to disk before closing ---
        if h5_features_ds is not None:
            logger.info(f"Flushing HDF5 file {output_h5_path}...")
            h5f.flush()
            logger.info("HDF5 file flushed.")
        else:
            logger.warning("No features were processed, HDF5 file might be empty.")


    # --- Save Metadata ---
    try:
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(output_meta_path), exist_ok=True)
        with open(output_meta_path, 'w') as f:
            # Add overall dataset info to metadata if desired
            full_metadata = {
                "dataset_info": {
                    "total_cached_features": feature_count,
                    "feature_shape": list(feature_shape) if feature_shape else None,
                    "h5_feature_path": "features", # Path within H5 file
                    "source_json": json_path,
                    "num_augmentations_per_image": num_augmentations,
                },
                "sample_mapping": metadata # Dict mapping path to index/label
            }
            json.dump(full_metadata, f, indent=4)
        logger.info(f"Successfully cached {feature_count} features.")
        logger.info(f"Metadata saved to {output_meta_path}")
    except IOError as e:
        logger.error(f"Error saving metadata to {output_meta_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error saving metadata: {e}")

    # --- Final Logging ---
    original_count = len(dataset) # Use len(dataset) for potential filtered items
    if feature_count < original_count:
        logger.warning(
            f"Processed {feature_count} features, but dataset "
            f"reported {original_count} items. Some may have failed loading."
        )
    if device == 'cuda':
        logger.info(
            f"Max CUDA memory allocated: "
            f"{torch.cuda.max_memory_allocated() / (1024**3):.2f} GB"
        )
    torch.cuda.empty_cache()

    try:
        del pooled_features, features, labels
        del features_np
    except Exception as e:
        logger.warning(f"Couldn't clean features before ending")
    gc.collect() # Optional: Force garbage collection

    return feature_count

# --- Classifier Model ---

class AestheticClassifier(torch.nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 4,
        hidden_dims: int = DEFAULT_CLASSIFIER_HIDDEN_DIMS,
        dropout_rate: float = DEFAULT_CLASSIFIER_DROPOUT
        ):
        super(AestheticClassifier, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(feature_dim, hidden_dims)
        self.bn1 = nn.BatchNorm1d(hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims//2)
        self.bn2 = nn.BatchNorm1d(hidden_dims//2)
        self.fc3 = nn.Linear(hidden_dims//2, num_classes)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout_rate)

    def forward(self,  x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.fc1(x))
        x = self.relu(x)
        x = self.bn2(self.fc2(x))
        x = self.drop(self.relu(x))

        x = self.fc3(x)
        return x

def to_one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converts integer labels to one-hot vectors."""
    return torch.nn.functional.one_hot(
        labels, num_classes=num_classes
    ).float()


def train_classifier(
    h5_features_path: str,
    meta_path: str,
    h5_features_path_val: str,
    meta_path_val: str,
    feature_dim: int,
    num_classes: int = 4,
    hidden_dims: List[int] = DEFAULT_CLASSIFIER_HIDDEN_DIMS,
    dropout_rate: float = DEFAULT_CLASSIFIER_DROPOUT,
    num_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    device: torch.device = torch.device('cpu'),
    save_path: str = 'output/aesthetic_classifier.pth',
    # val_split: float = 0.2,
    early_stopping_patience: int = 5,
    num_workers: int = 2,
    # Add Augmentation/Regularization Params
    use_mixup: bool = False,
    mixup_alpha: float = 0.2,
    use_noise: bool = False,
    noise_std: float = 0.05,
    class_weights_data: Optional[List[float]] = None
) -> Optional[nn.Module]:
    """
    Trains the aesthetic classifier on cached features with optional
    feature-level augmentations.
    """

    # Ensure save directory exists
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)


    # Create datasets using the temporary metadata files
    train_dataset = FeatureDataset(h5_features_path, meta_path)
    val_dataset = FeatureDataset(h5_features_path_val, meta_path_val)

    logger.info(f"Training set size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation set size: {len(val_dataset)}")
    else:
            logger.warning("No validation set created.")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == 'cuda'),
        drop_last=True if use_mixup else False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == 'cuda'),
    ) if val_dataset else None


    # Model, Loss, Optimizer
    model = AestheticClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate
    ).to(device)
    model = torch.compile(model)

    # Prepare class weights for the loss function if provided
    weights_tensor = None
    if class_weights_data:
        if len(class_weights_data) == num_classes:
            weights_tensor = torch.tensor(
                class_weights_data, dtype=torch.float
            ).to(device)
            logger.info(f"Using class weights for loss: {weights_tensor.tolist()}")
        else:
            logger.warning(
                f"Length of class_weights_data ({len(class_weights_data)}) "
                f"does not match num_classes ({num_classes}). Ignoring weights."
            )

    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    best_val_metric = -1.0 # Use F1 or Accuracy
    patience_counter = 0
    history = {'train_loss': [], 'val_accuracy': [], 'val_f1': [], 'val_loss': []}

    # Log augmentation settings
    logger.info("Starting classifier training...")
    logger.info(f"Using Mixup: {use_mixup}, Alpha: {mixup_alpha if use_mixup else 'N/A'}")
    logger.info(f"Using Feature Noise: {use_noise}, StdDev: {noise_std if use_noise else 'N/A'}")
    logger.info(f"Dropout Rate: {dropout_rate}")
    logger.info(f"Weight Decay: {weight_decay}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_steps = 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Train")
        for features, labels in pbar_train:
            if features.numel() == 0: continue # Skip empty batches from collate_fn

            features = features.to(device)
            labels = labels.to(device)

            mixed_features = features
            target_labels = labels # Default to original labels

            if use_mixup and mixup_alpha > 0:
                # Generate lambda from Beta distribution
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                # Get permutation indices
                batch_s = features.size(0)
                index = torch.randperm(batch_s).to(device)

                # Mix features
                mixed_features = lam * features + (1 - lam) * features[index, :]

                # Convert labels to one-hot and mix them (soft labels)
                labels_onehot = to_one_hot(labels, num_classes).to(device)
                labels_shuffled_onehot = labels_onehot[index, :]
                target_labels = lam * labels_onehot + (1 - lam) * labels_shuffled_onehot

            if use_noise and noise_std > 0:
                # Add Gaussian noise to the (potentially mixed) features
                noise = torch.randn_like(mixed_features) * noise_std
                # Apply noise only if mixup wasn't used, or apply to mixed feats
                if not use_mixup:
                    mixed_features = features + noise
                else:
                    mixed_features = mixed_features + noise # Add noise to mixed features


            optimizer.zero_grad()
            outputs = model(mixed_features)
            loss = criterion(outputs, target_labels)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            print(f"Grad norm {norm.item()}")
            optimizer.step()

            running_loss += loss.item()
            train_steps += 1
            pbar_train.set_postfix({"loss": running_loss / train_steps})

        epoch_train_loss = running_loss / train_steps if train_steps > 0 else 0.0
        history['train_loss'].append(epoch_train_loss)
        logger.info(f"Epoch {epoch+1} Train Loss: {epoch_train_loss:.4f}")

        # Validation Phase
        if val_loader:
            model.eval()
            val_preds, val_true, val_loss = [], [], 0
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Val")
            with torch.no_grad():
                for features, labels in pbar_val:
                    if features.numel() == 0: continue

                    features = features.to(device)
                    outputs = model(features)
                    loss = criterion(outputs, labels.to(device))
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

            if not val_true:
                logger.warning("Validation set yielded no samples.")
                accuracy, f1 = 0.0, 0.0
            else:
                accuracy = accuracy_score(val_true, val_preds)
                # Use weighted F1 for potentially imbalanced classes
                f1 = f1_score(val_true, val_preds, average='weighted', zero_division=0)
                conf_matrix = confusion_matrix(val_true, val_preds)

                val_loss = val_loss/len(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(accuracy)
                history['val_f1'].append(f1)

                logger.info(f"Epoch {epoch+1} Val loss {val_loss:.4f} Val Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                logger.debug(f"Confusion Matrix:\n{conf_matrix}")

            # Use F1 score for scheduler and early stopping
            current_val_metric = f1
            scheduler.step(current_val_metric)

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                logger.info(f"New best F1: {best_val_metric:.4f}. Saving model...")
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_metric': best_val_metric,
                        'feature_dim': feature_dim,
                        'num_classes': num_classes,
                        'hidden_dims': hidden_dims,
                        'dropout_rate': dropout_rate,
                    }, save_path)
                except Exception as e:
                    logger.error(f"Error saving model: {e}")
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"Val F1 did not improve. Patience: {patience_counter}/{early_stopping_patience}")

            if patience_counter >= early_stopping_patience:
                logger.info("Early stopping triggered.")
                break
        else:
            # No validation, save model every epoch or based on train loss?
            # For simplicity, just save the last epoch if no validation
            logger.info("No validation set. Saving model from last epoch.")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    # ... include other relevant info
                    'feature_dim': model.feature_dim,
                    'num_classes': model.num_classes,
                }, save_path)
            except Exception as e:
                logger.error(f"Error saving model: {e}")


    # Save training history
    history_path = save_path.replace('.pth', '_history.json')
    try:
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        logger.info(f"Training history saved to {history_path}")
    except IOError as e:
        logger.error(f"Error saving training history: {e}")

    # Load the best model state if validation was performed
    if val_loader and os.path.exists(save_path):
        try:
            checkpoint = torch.load(save_path, map_location=device)

            # Create a new state dict to hold the adjusted keys
            adjusted_state_dict = {}
            has_orig_mod_prefix = False
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith("_orig_mod."):
                    # Strip the prefix
                    new_key = key[len("_orig_mod."):]
                    adjusted_state_dict[new_key] = value
                    has_orig_mod_prefix = True
                else:
                    # Keep keys without the prefix as-is (e.g., if compile failed)
                    adjusted_state_dict[key] = value

            if has_orig_mod_prefix:
                logger.info(
                    "Adjusted state_dict keys by removing '_orig_mod.' prefix "
                    "for compatibility with uncompiled model loading."
                )
            # --- *** THE FIX ENDS HERE *** ---

            # Re-create a *fresh, uncompiled* model instance
            # Use parameters from the checkpoint for consistency
            final_model = AestheticClassifier(
                feature_dim=checkpoint['feature_dim'],
                num_classes=checkpoint['num_classes'],
                hidden_dims=checkpoint.get(
                    'hidden_dims', DEFAULT_CLASSIFIER_HIDDEN_DIMS
                ),
                dropout_rate=checkpoint.get(
                    'dropout_rate', DEFAULT_CLASSIFIER_DROPOUT
                )
            ).to(device)

            # Load the *adjusted* state dict
            final_model.load_state_dict(adjusted_state_dict)

            logger.info(f"Loaded best model from {save_path} with metric: {checkpoint['best_val_metric']:.4f}")
            return final_model
        except Exception as e:
            logger.error(f"Error loading best model state dict: {e}")
            # Fallback to returning the model as it was at the end of training
            return model
    elif os.path.exists(save_path): # Case: No validation, last model saved
         try:
            checkpoint = torch.load(save_path, map_location=device)
            # Assuming the current model instance is the one saved
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from last epoch saved at {save_path}")
            return model
         except Exception as e:
            logger.error(f"Error loading saved model state dict: {e}")
            return model # Return the model in its current state
    else:
        logger.warning("No model checkpoint found to load.")
        return model # Return the model as is


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract features and train an aesthetic classifier."
    )
    # Paths
    parser.add_argument(
        '--model_dir', type=str, default=DEFAULT_MODEL_DIR,
        help="Directory containing SwinV2 model files (config.json, model.safetensors)."
    )
    parser.add_argument(
        '--json_path', type=str, required=True,
        help="Path to input JSON file (image_path -> class_label)."
    )
    parser.add_argument(
        '--global_path', type=str, default=DEFAULT_GLOBAL_PATH,
        help="Directory to save features, metadata, and trained model."
    )

    parser.add_argument(
        '--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help="Directory to save features, metadata, and trained model."
    )
    # Feature Extraction
    parser.add_argument(
        '--skip_extraction', action='store_true',
        help="Skip feature extraction if H5/meta files exist."
    )
    parser.add_argument(
        '--extract_batch_size', type=int, default=32,
        help="Batch size for feature extraction."
    )
    parser.add_argument(
        '--device', type=str, default='cuda', choices=['cuda', 'cpu'],
        help="Device to use ('cuda' or 'cpu')."
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help="Number of worker processes for DataLoaders."
    )
    parser.add_argument(
        '--num_augmentations', type=int, default=DEFAULT_NUM_AUGMENTATIONS,
        help="Number of versions per image (1=original only, >1=original+augs)."
    )

    # Training
    parser.add_argument(
        '--num_classes', type=int, default=4,
        help="Number of aesthetic classes for the classifier."
    )
    parser.add_argument(
        '--hidden_dims', type=int,
        default=DEFAULT_CLASSIFIER_HIDDEN_DIMS,
        help="Hidden layer dimensions for the classifier MLP."
    )
    parser.add_argument(
        '--dropout', type=float, default=DEFAULT_CLASSIFIER_DROPOUT,
        help="Dropout rate for the classifier."
    )
    parser.add_argument(
        '--epochs', type=int, default=30, help="Number of training epochs."
    )
    parser.add_argument(
        '--train_batch_size', type=int, default=64,
        help="Batch size for classifier training."
    )
    parser.add_argument(
        '--lr', type=float, default=0.001, help="Learning rate."
    )
    parser.add_argument(
        '--wd', type=float, default=1e-5, help="Weight decay."
    )
    parser.add_argument(
        '--val_split', type=float, default=0.2,
        help="Fraction of data to use for validation (0 to disable)."
    )
    parser.add_argument(
        '--patience', type=int, default=5,
        help="Early stopping patience (epochs)."
    )
    parser.add_argument(
        '--use_mixup', action='store_true',
        help="Enable Mixup feature augmentation during training."
    )
    parser.add_argument(
        '--mixup_alpha', type=float, default=0.2,
        help="Alpha parameter for the Beta distribution in Mixup."
    )
    parser.add_argument(
        '--use_noise', action='store_true',
        help="Enable Feature Noise Injection during training."
    )
    parser.add_argument(
        '--noise_std', type=float, default=0.05,
        help="Standard deviation for Gaussian noise injection."
    )


    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda"

    # --- 1. Load Feature Extractor ---
    logger.info(f"Loading feature extractor from: {args.model_dir}")
    feature_extractor, config, input_size_hw = load_feature_extractor(args.model_dir)
    if feature_extractor is None or input_size_hw is None:
        logger.error("Failed to load feature extractor model. Exiting.")
        sys.exit(1)
    feature_extractor.to(device) # Move model to device

    # Determine feature dimension
    try:
        feature_dim = get_feature_dimension(feature_extractor, input_size_hw)
        logger.info(f"Determined feature dimension: {feature_dim}")
    except Exception as e:
        logger.error(f"Could not get feature dimension: {e}. Exiting.")
        sys.exit(1)

    # --- 2. Feature Extraction (Optional) ---
    output_h5_path = os.path.join(args.output_dir, 'cached_features_train.h5')
    output_meta_path = os.path.join(args.output_dir, 'cached_features_meta_train.json')
    output_h5_path_val = os.path.join(args.output_dir, 'cached_features_val.h5')
    output_meta_path_val = os.path.join(args.output_dir, 'cached_features_meta_val.json')

    processor_config = {
        "size": {"height": 448, "width": 448},
        "color": [255, 255, 255], # Padding color (RGB)
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "rescale_factor": 1 / 255.0,
        "resample": PILImageResampling.BILINEAR, # Use constant
    }


    perform_extraction = True
    if args.skip_extraction and os.path.exists(output_h5_path) and os.path.exists(output_meta_path):
        logger.info("Skipping feature extraction as files exist.")
        perform_extraction = False
        # Basic check: ensure metadata isn't empty if skipping
        try:
            with open(output_meta_path, 'r') as f:
                meta = json.load(f)
            if not meta:
                 logger.warning(f"Metadata file {output_meta_path} is empty. Extraction might be needed.")
                 perform_extraction = True # Force extraction if meta is empty
            # Optional: Check if num_augmentations matches requested
            meta_augs = meta.get("dataset_info", {}).get("num_augmentations_per_image", 1)
            if meta_augs != args.num_augmentations:
                logger.warning(
                    f"Existing metadata has num_augmentations={meta_augs}, "
                    f"but requested {args.num_augmentations}. Re-extracting."
                )
                perform_extraction = True

        except Exception as e:
            logger.warning(f"Could not verify existing metadata file {output_meta_path}: {e}. Re-extracting.")
            perform_extraction = True

    if perform_extraction:
        logger.info("Starting feature extraction and caching...")
        num_cached = extract_and_cache_features(
            model=feature_extractor,
            json_path=args.json_path,
            output_h5_path=output_h5_path,
            output_meta_path=output_meta_path,
            processor_config=processor_config,
            batch_size=args.extract_batch_size,
            device=device,
            num_workers=args.num_workers,
            global_path=args.global_path,
            num_augmentations=args.num_augmentations
        )
        if num_cached == 0:
            logger.error("No features were cached. Cannot proceed to training.")
            sys.exit(1)
    else:
        logger.info(f"Using existing features from {output_h5_path}")

    # dataset = FeatureDataset(output_h5_path, output_meta_path)
    # # Use custom collate_fn to handle loading errors
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=args.extract_batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     pin_memory=(device == 'cuda'),
    #     prefetch_factor=4 if args.num_workers > 0 else None,
    #     persistent_workers=True if args.num_workers > 0 else False,
    # )
    # batch = next(iter(dataloader))
    # print(f"Image shape {batch[0].shape} and labels {batch[1]}")

    # --- 3. Train Classifier ---
    logger.info("Starting classifier training...")
    classifier_save_path = os.path.join(args.output_dir, 'aesthetic_classifier.pth')

    train_class_counts = [1888, 2122, 2395, 962]
    total_train_samples = sum(train_class_counts)
    num_classes_actual = len(train_class_counts) # Should match args.num_classes

    if num_classes_actual != args.num_classes:
        logger.warning(
            f"Number of classes from counts ({num_classes_actual}) "
            f"differs from args.num_classes ({args.num_classes}). "
            "Ensure this is intended."
        )
        # Proceeding with num_classes_actual for weight calculation
        # but CrossEntropyLoss will use args.num_classes for output layer size.

    class_weights_list = []
    if total_train_samples > 0 and num_classes_actual > 0:
        for count in train_class_counts:
            if count > 0:
                weight = total_train_samples / (num_classes_actual * count)
                class_weights_list.append(weight)
            else:
                # Handle case where a class might have 0 samples (though unlikely here)
                class_weights_list.append(1.0) # Default weight
        logger.info(f"Calculated class weights: {class_weights_list}")
    else:
        logger.warning("Could not calculate class weights (no samples or classes).")

    trained_classifier = train_classifier(
        h5_features_path=output_h5_path,
        meta_path=output_meta_path,
        h5_features_path_val=output_h5_path_val,
        meta_path_val=output_meta_path_val,
        feature_dim=feature_dim,
        num_classes=args.num_classes,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout,
        num_epochs=args.epochs,
        batch_size=args.train_batch_size,
        lr=args.lr,
        weight_decay=args.wd,
        device=device,
        save_path=classifier_save_path,
        # val_split=args.val_split if use_validation else 0.0,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        use_mixup=args.use_mixup,
        mixup_alpha=args.mixup_alpha,
        use_noise=args.use_noise,
        noise_std=args.noise_std,
        class_weights_data=class_weights_list
    )

    if trained_classifier:
        logger.info(f"Classifier training finished. Model saved to {classifier_save_path}")
    else:
        logger.error("Classifier training failed.")
        sys.exit(1)


    logger.info("Script finished successfully.")

if __name__ == "__main__":
    main()

    """
    python aesthetic/aesthetic.py  --json_path aesthetic/aesthetic_labels_train.final.json --num_augmentations 2 --extract_batch_size 32 --epochs 100 --train_batch_size 1024 --use_mixup --dropout 0.3 --wd 0.1  --patience 50 --lr 1e-4
    python aesthetic/aesthetic.py  --json_path aesthetic/aesthetic_labels_train.final.json --num_augmentations 1 --extract_batch_size 32 --epochs 100 --train_batch_size 2048 --use_mixup --dropout 0.3 --wd 0.1  --patience 50 --lr 1e-4
    python aesthetic/aesthetic.py  --json_path aesthetic/aesthetic_labels_train.final.json --skip_extraction --train_batch_size 1024 --use_mixup --use_noise --dropout 0.3 --wd 1e-3  --num_augmentations 2 --patience 15 --lr 1e-4
    python aesthetic/aesthetic.py  --json_path aesthetic/aesthetic_labels_train.final.json --skip_extraction --epochs 100 --train_batch_size 1024 --use_mixup --dropout 0.3 --wd 1e-3  --num_augmentations 2 --patience 50 --lr 1e-4
    """