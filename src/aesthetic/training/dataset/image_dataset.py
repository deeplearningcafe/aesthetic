import json
import torch
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import v2
from transformers.image_transforms import (
    rescale,
    normalize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    infer_channel_dimension_format,
    is_scaled_image,
)
import os
import logging
from typing import Tuple, Optional, Union

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)


def resize_with_padding(
    image: np.ndarray,
    size: Tuple[int, int],
    color: Tuple[int, int, int] = (255, 255, 255),
    resample=PILImageResampling.BILINEAR,
    data_format: Optional[ChannelDimension] = None,
    input_data_format: Optional[Union[str, ChannelDimension]] = None,
):
    if input_data_format is None:
        input_data_format = infer_channel_dimension_format(image)
    data_format = input_data_format if data_format is None else data_format

    do_rescale_back = False
    if not isinstance(image, Image.Image):
        if is_scaled_image(image):
            do_rescale_back = True
            image = image * 255
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        image = Image.fromarray(image)

    original_width, original_height = image.size
    height, width = size

    ratio = min(width / original_width, height / original_height)
    new_width = int(original_width * ratio)
    new_height = int(original_height * ratio)

    resized_image = image.resize((new_width, new_height), resample=resample)

    new_image_rgba = Image.new("RGBA", (width, height), color + (255,))

    offset = ((width - new_width) // 2, (height - new_height) // 2)
    resized_image_rgba = resized_image.convert("RGBA")
    new_image_rgba.paste(resized_image_rgba, offset, resized_image_rgba)

    new_image_rgb = new_image_rgba.convert("RGB")

    image_array = np.asarray(new_image_rgb, dtype=np.float32)
    image_array = image_array[:, :, ::-1]

    if image_array.ndim == 2:
        image_array = np.expand_dims(image_array, axis=-1)

    image_array = to_channel_dimension_format(
        image_array, data_format, input_channel_dim=ChannelDimension.LAST
    )

    if do_rescale_back:
        image_array = image_array / 255.0

    return image_array


class ImagePathDataset(Dataset):
    """
    Dataset for loading image paths and labels from a JSON file.
    Applies data augmentation to create multiple versions of each image.
    Supports both 4-class dicts and Pairs JSON formats.
    """

    def __init__(
        self,
        json_path: str,
        processor_config: dict,
        global_path: str = None,
        num_augmentations: int = 1,
    ):
        self.json_path = json_path
        self.global_path = global_path
        self.size_dict = processor_config["size"]
        self.color_tuple = tuple(processor_config["color"])
        self.image_mean = processor_config["image_mean"]
        self.image_std = processor_config["image_std"]
        self.rescale_factor = processor_config["rescale_factor"]
        self.resample_filter = processor_config["resample"]
        self.num_augmentations = max(1, num_augmentations)

        try:
            with open(json_path, "r") as f:
                raw_data = json.load(f)

            self.data = {}
            for k, v in raw_data.items():
                # Detect Pair format vs 4-class format
                if isinstance(v, dict) and "left_path" in v and "right_path" in v:
                    self.data[v["left_path"]] = -1  # Dummy label for feature caching
                    self.data[v["right_path"]] = -1
                else:
                    self.data[k] = v

            self.image_paths = list(self.data.keys())
            logger.info(f"Loaded {len(self.image_paths)} unique image paths.")
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
            self.augmentation_transform = v2.Compose(
                [
                    v2.RandomHorizontalFlip(p=0.5),
                    v2.RandomRotation(degrees=15),
                    v2.RandomAffine(
                        degrees=0, translate=(0.075, 0.075), scale=(0.9, 1.1), shear=10
                    ),
                    v2.RandomPerspective(
                        distortion_scale=0.15,
                        p=0.3,
                        interpolation=v2.InterpolationMode.BILINEAR,
                    ),
                    v2.RandomApply(
                        [
                            v2.ColorJitter(
                                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                            )
                        ],
                        p=0.8,
                    ),
                    v2.RandomGrayscale(p=0.1),
                    v2.RandomApply([v2.GaussianBlur(kernel_size=3)], p=0.3),
                    v2.RandomErasing(p=0.2, scale=(0.02, 0.75), ratio=(0.5, 2)),
                ]
            )
        else:
            self.augmentation_transform = None

    def preprocess_image(self, image):
        image_array = np.array(image.convert("RGB"))

        processed_image = resize_with_padding(
            image=image_array,
            size=(self.size_dict["height"], self.size_dict["width"]),
            color=self.color_tuple,
            resample=self.resample_filter,
            data_format=ChannelDimension.FIRST,
        )

        if not is_scaled_image(processed_image):
            processed_image = rescale(
                processed_image,
                scale=self.rescale_factor,
                data_format=ChannelDimension.FIRST,
            )

        processed_image = normalize(
            processed_image,
            mean=self.image_mean,
            std=self.image_std,
            data_format=ChannelDimension.FIRST,
        )

        img_tensor = torch.tensor(processed_image).float()
        return img_tensor

    def __len__(self) -> int:
        return len(self.image_paths) * self.num_augmentations

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int, str]]:
        if idx >= self.__len__():
            raise IndexError("Dataset index out of range.")

        original_idx = idx // self.num_augmentations
        augmentation_idx = idx % self.num_augmentations

        img_path = self.image_paths[original_idx]
        label = self.data[img_path]

        img_path_open = img_path
        if self.global_path:
            img_path_corrected = img_path.replace("\\", os.sep)
            img_path_open = os.path.join(self.global_path, img_path_corrected)

        if not os.path.exists(img_path_open):
            logger.warning(f"Image file not found: {img_path_open}. Skipping.")
            return None

        try:
            image = Image.open(img_path_open).convert("RGB")
            augmented_image = image
            if augmentation_idx > 0 and self.augmentation_transform:
                try:
                    augmented_image = self.augmentation_transform(image)
                except Exception as aug_e:
                    logger.warning(
                        f"Error applying augmentation {augmentation_idx} to "
                        f"{img_path}: {aug_e}. Using original."
                    )
                    augmented_image = image

            image_tensor = self.preprocess_image(augmented_image)

            unique_img_key = img_path
            if augmentation_idx > 0:
                unique_img_key = f"{img_path}_aug_{augmentation_idx}"

            return image_tensor, label, unique_img_key

        except (IOError, OSError, Image.DecompressionBombError) as e:
            logger.warning(f"Error loading/processing image {img_path}: {e}. Skipping.")
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error processing image {img_path}: {e}", exc_info=True
            )
            return None
