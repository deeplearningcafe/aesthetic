import json
from pathlib import Path
from typing import Optional, Generator, Callable

# Define image file extensions
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"
}
DEFAULT_GLOBAL_PATH =  'FOLDER-PATH'

def is_image_file(path: Path) -> bool:
    """Check if a file has an image extension."""
    return path.suffix.lower() in IMAGE_SUFFIXES

def dirwalk(
    path: Path, 
    condition: Optional[Callable] = None
) -> Generator[Path, None, None]:
    """Walk through directory and yield files that meet the condition."""
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, condition)
        elif condition is None or condition(p):
            yield p


def add_images_to_json(
    image_dir_str: str, 
    json_file_path_str: str, 
    label: int = 3
):
    """
    Adds image paths from a directory to a JSON file with a specified label.

    The image paths in the JSON are relative to DEFAULT_GLOBAL_PATH.
    All images in the given directory receive the same label.

    Args:
        image_dir_str: Path to the directory containing images.
        json_file_path_str: Path to the JSON file to update.
        label: The label to assign to the images (default is 3).
    """
    image_dir = Path(image_dir_str)
    json_file_path = Path(json_file_path_str)
    
    global_base_path = Path(DEFAULT_GLOBAL_PATH)

    if not image_dir.is_dir():
        print(f"Error: Image directory '{image_dir}' not found.")
        return

    data = {}
    if json_file_path.exists() and json_file_path.stat().st_size > 0:
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            print(
                f"Warning: '{json_file_path}' contains invalid JSON. "
                f"Starting with an empty dataset."
            )
        except Exception as e:
            print(
                f"Error reading '{json_file_path}': {e}. "
                f"Starting with an empty dataset."
            )
    
    images_added_count = 0
    # Assuming dirwalk and is_image_file are defined from your util code
    for image_path_obj in dirwalk(image_dir, is_image_file):
        try:
            # Resolve to an absolute path to handle various input forms
            absolute_image_path = image_path_obj.resolve()
            
            # Calculate path relative to the global base path
            relative_path = absolute_image_path.relative_to(global_base_path)
            
            # Format path string to use backslashes, matching example JSON
            relative_path_str = str(relative_path).replace('/', '\\')
            
            data[relative_path_str] = label
            images_added_count += 1
        except ValueError:
            # This occurs if absolute_image_path is not under global_base_path
            print(
                f"Warning: Image '{absolute_image_path}' is not under "
                f"'{global_base_path}'. Skipping."
            )
        except Exception as e:
            print(
                f"Error processing image '{image_path_obj}': {e}. Skipping."
            )

    if images_added_count > 0:
        try:
            # Ensure parent directory for json_file_path exists
            json_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print(
                f"Successfully added {images_added_count} images from "
                f"'{image_dir}' to '{json_file_path}' with label {label}."
            )
        except Exception as e:
            print(f"Error writing to '{json_file_path}': {e}")
    else:
        if image_dir.is_dir(): # Only print if dir was valid but no images
            print(f"No new images found or added from '{image_dir}'.")

if __name__ == "__main__":
    img_dir = 'FOLDER-PATH'
    json_path = 'FILE-PATH.final.json'
    label = 3
    print(add_images_to_json(img_dir, json_path, label))
