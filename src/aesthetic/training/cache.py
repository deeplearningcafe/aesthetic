import os
import gc
import json
import h5py
import torch
import sys
from torch.utils.data import DataLoader
from transformers.image_utils import PILImageResampling
from aesthetic.training.dataset.image_dataset import ImagePathDataset
from aesthetic.training.models.tagger import (
    load_feature_extractor,
    get_feature_dimension,
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class FeatureCacher:
    """Handles extracting features using SwinV2 and caching them to H5."""

    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device
        self.model = None

    def load_model(self):
        """Loads the SwinV2 model into memory."""
        if self.model is not None:
            return

        self.model, config, input_size_hw = load_feature_extractor(self.model_dir)
        if self.model is None or input_size_hw is None:
            logger.error("Failed to load feature extractor model. Exiting.")
            sys.exit(1)

        try:
            feature_dim = get_feature_dimension(self.model, input_size_hw)
            logger.info(f"Determined feature dimension: {feature_dim}")
        except Exception as e:
            logger.error(f"Could not get feature dimension: {e}. Exiting.")
            sys.exit(1)

        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)

    def unload_model(self):
        """Frees up VRAM after caching is complete."""
        if self.model is not None:
            del self.model
            self.model = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

    def cache_dataset(
        self,
        json_path: str,
        output_h5: str,
        output_meta: str,
        global_path: str = None,
        batch_size: int = 32,
        progress_callback=None,
    ):
        """Extracts features and saves them to an H5 file."""
        self.load_model()

        processor_config = {
            "size": {"height": 448, "width": 448},
            "color": [255, 255, 255],
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1 / 255.0,
            "resample": PILImageResampling.BILINEAR,
        }

        dataset = ImagePathDataset(
            json_path=json_path,
            processor_config=processor_config,
            global_path=global_path,
            num_augmentations=1,
        )

        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            if not batch:
                return None
            return torch.utils.data.dataloader.default_collate(batch)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=4,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        os.makedirs(os.path.dirname(output_h5), exist_ok=True)

        metadata = {}
        feature_count = 0

        with h5py.File(output_h5, "w") as h5f:
            h5_ds = None

            with torch.no_grad():
                for i, batch in enumerate(loader):
                    if batch is None:
                        continue

                    images, labels, paths = batch
                    images = images.to(self.device)

                    features = self.model.forward_features(images)
                    pooled = self.model.head.global_pool(features)
                    features_np = pooled.cpu().numpy()

                    num_in_batch = features_np.shape[0]

                    if h5_ds is None:
                        feat_shape = features_np.shape[1:]
                        h5_ds = h5f.create_dataset(
                            "features",
                            shape=(0,) + feat_shape,
                            maxshape=(None,) + feat_shape,
                            dtype="float32",
                            compression="gzip",
                        )

                    current_size = h5_ds.shape[0]
                    h5_ds.resize(current_size + num_in_batch, axis=0)
                    h5_ds[current_size:] = features_np

                    for j in range(num_in_batch):
                        metadata[paths[j]] = {
                            "h5_index": current_size + j,
                            "label": int(labels[j].item()),
                        }

                    feature_count += num_in_batch

                    if progress_callback:
                        progress_callback(i + 1, len(loader))

        full_meta = {
            "dataset_info": {"h5_feature_path": "features"},
            "sample_mapping": metadata,
        }
        with open(output_meta, "w") as f:
            json.dump(full_meta, f, indent=2)

        self.unload_model()
        return feature_count
