import json
import torch
import random
from torch.utils.data import Dataset
import h5py
import os
import logging
from contextlib import contextmanager
import sys
from typing import Optional, Generator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


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
            with open(meta_path, "r") as f:
                metadata = json.load(f)

            # --- Updated Metadata Handling ---
            if "dataset_info" not in metadata or "sample_mapping" not in metadata:
                raise ValueError(
                    "Metadata JSON format incorrect. Missing "
                    "'dataset_info' or 'sample_mapping'."
                )

            self.h5_feature_dataset_path = metadata["dataset_info"].get(
                "h5_feature_path",
                "features",  # Default to "features" if missing
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
            with open_h5_file(self.h5_path, "r") as h5_file:
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

    def __getitem__(self, idx: int) -> Optional[tuple[torch.Tensor, int]]:
        """Returns (feature_tensor, label) or None if error."""
        if idx >= len(self.image_keys):
            raise IndexError("Index out of range")

        img_key = self.image_keys[idx]
        meta_info = self.sample_mapping.get(img_key)

        if meta_info is None:
            logger.warning(f"Metadata not found for key: {img_key}. Skipping.")
            return None  # Indicate failure to load

        try:
            h5_index = meta_info["h5_index"]
            label = meta_info["label"]  # Assuming label is directly usable

            # Open H5 file here for thread/process safety with DataLoader workers
            with open_h5_file(self.h5_path, "r") as h5_file:
                # Access the main dataset using the path from metadata
                features_dataset = h5_file[self.h5_feature_dataset_path]
                # Retrieve the specific feature vector by its index
                feature_np = features_dataset[h5_index]
                # Convert numpy array to torch tensor
                feature = torch.from_numpy(feature_np)

            # Ensure label is in a usable format (e.g., int or tensor)
            # This might need adjustment based on how labels are used later
            if isinstance(label, list):  # Example: handle one-hot labels if needed
                label_tensor = torch.tensor(label, dtype=torch.float32)
                return feature, label_tensor
            else:  # Assume integer label
                return feature, int(label)

        except KeyError as e:
            logger.warning(
                f"KeyError accessing data for key {img_key}: {e}. "
                f"Maybe missing 'h5_index' or dataset path "
                f"'{self.h5_feature_dataset_path}' incorrect?"
            )
            return None  # Indicate failure
        except IndexError:
            logger.warning(
                f"H5 index {h5_index} out of bounds for key {img_key} in "
                f"dataset '{self.h5_feature_dataset_path}'. "
                f"H5 file might not match metadata."
            )
            return None  # Indicate failure
        except Exception as e:
            logger.error(
                f"Error reading feature index {h5_index} (key: {img_key}) "
                f"from {self.h5_path} dataset "
                f"'{self.h5_feature_dataset_path}': {e}"
            )
            return None  # Indicate failure


class PairFeatureDataset(Dataset):
    """Dataset for loading H5 features for pair comparisons."""

    def __init__(self, h5_path: str, meta_path: str, pairs_json: str):
        self.h5_path = h5_path

        with open(meta_path, "r") as f:
            metadata = json.load(f)
            self.meta = metadata.get("sample_mapping", metadata)

        with open(pairs_json, "r") as f:
            self.pairs_data = json.load(f)

        self.valid_pairs = []
        for key, pair in self.pairs_data.items():
            p1, p2 = pair["left_path"], pair["right_path"]
            if p1 in self.meta and p2 in self.meta:
                self.valid_pairs.append(
                    {
                        "idx1": self.meta[p1]["h5_index"],
                        "idx2": self.meta[p2]["h5_index"],
                        "label": pair["winner"],
                    }
                )

    def __len__(self) -> int:
        return len(self.valid_pairs)

    def __getitem__(self, idx: int):
        pair = self.valid_pairs[idx]
        with open_h5_file(self.h5_path, "r") as h5f:
            ds = h5f["features"]
            emb1 = torch.from_numpy(ds[pair["idx1"]])
            emb2 = torch.from_numpy(ds[pair["idx2"]])

        return emb1, emb2, pair["label"]


def split_train_val_pairs(
    input_json: str, train_json: str, val_json: str, val_ratio: float = 0.1
):
    """
    Utility to split the generated labels_pairs.json into training and validation sets.
    This creates the validation dataset needed for the Pair MLP training loop.
    """
    with open(input_json, "r") as f:
        pairs = json.load(f)

    pair_items = list(pairs.items())
    random.shuffle(pair_items)

    val_size = int(len(pair_items) * val_ratio)
    val_pairs = dict(pair_items[:val_size])
    train_pairs = dict(pair_items[val_size:])

    with open(train_json, "w") as f:
        json.dump(train_pairs, f, indent=2)

    with open(val_json, "w") as f:
        json.dump(val_pairs, f, indent=2)

    print(
        f"Split {len(pair_items)} pairs into {len(train_pairs)} train and {len(val_pairs)} validation pairs."
    )


@contextmanager
def open_h5_file(file_path: str, mode: str = "r") -> Generator[h5py.File, None, None]:
    """Context manager for safely opening and closing H5 files."""
    h5_file = None
    try:
        if mode in ["w", "a", "w-", "x"]:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        h5_file = h5py.File(file_path, mode)
        yield h5_file
    except Exception as e:
        logger.error(f"Error opening H5 file {file_path} in mode {mode}: {e}")
        raise
    finally:
        if h5_file is not None:
            try:
                h5_file.close()
            except Exception as e:
                logger.error(f"Error closing H5 file {file_path}: {e}")
