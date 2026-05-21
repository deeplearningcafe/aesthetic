import pandas as pd
import re
from typing import List, Optional
import numpy as np
import os


def apply_target_filters(
    df: pd.DataFrame,
    character_list: Optional[List[str]] = None,
    artist_list: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filters the dataset upfront based on target characters and artists."""
    if df.empty or (not character_list and not artist_list):
        return df

    final_mask = pd.Series(False, index=df.index)

    if character_list and "tag_string_character" in df.columns:
        escaped_tags = [re.escape(t) for t in character_list]
        char_pattern = r"(?:^|\s)(" + "|".join(escaped_tags) + r")(?:$|\s)"
        char_mask = df["tag_string_character"].str.contains(
            char_pattern, regex=True, na=False
        )
        final_mask = final_mask | char_mask

    if artist_list and "tag_string_artist" in df.columns:
        escaped_artists = [re.escape(t) for t in artist_list]
        artist_pattern = r"(?:^|\s)(" + "|".join(escaped_artists) + r")(?:$|\s)"
        artist_mask = df["tag_string_artist"].str.contains(
            artist_pattern, regex=True, na=False
        )
        final_mask = final_mask | artist_mask

    return df[final_mask]


def filter_dataset(
    df: pd.DataFrame,
    total_images: int,
    ratings_percentage: dict[str, float],
    skip_tags: Optional[dict[str, float]],
    exclude_df: Optional[pd.DataFrame] = None,
    random_seed: int = 42,
    id_proximity_threshold: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Filters dataset to select a diverse subset based on scores and ratings."""
    np.random.seed(random_seed)
    rng = np.random.RandomState(random_seed)

    df.dropna(subset=["file_url"], inplace=True)

    base_required_cols = [
        "id",
        "file_url",
        "score",
        "fav_count",
        "rating",
        "tag_string",
    ]

    optional_feature_cols = ["is_deleted", "is_banned", "parent_id", "md5"]

    missing_base_cols = [c for c in base_required_cols if c not in df.columns]
    if missing_base_cols:
        raise ValueError(f"Core required columns {missing_base_cols} not found.")

    cols_to_select = list(base_required_cols)
    for col in optional_feature_cols:
        if col in df.columns and col not in cols_to_select:
            cols_to_select.append(col)

    df = df[cols_to_select].copy()

    if "is_deleted" in df.columns:
        df = df[~df["is_deleted"].fillna(False)]
    if "is_banned" in df.columns:
        df = df[~df["is_banned"].fillna(False)]

    if "md5" in df.columns:
        df.sort_values("score", ascending=False, inplace=True)
        df.drop_duplicates(subset=["md5"], keep="first", inplace=True)
        df.sort_index(inplace=True)

    if "parent_id" in df.columns and "id" in df.columns:
        parent_ids = set(df.loc[df["parent_id"].notna(), "parent_id"].unique())
        all_ids = set(df["id"])
        ids_to_remove = parent_ids.intersection(all_ids)
        if ids_to_remove:
            df = df[~df["id"].isin(ids_to_remove)]

    if exclude_df is not None and not exclude_df.empty:
        ids_to_exclude = set(exclude_df["id"].unique())
        df = df[~df["id"].isin(ids_to_exclude)]

    if skip_tags:
        rows_to_keep = pd.Series(True, index=df.index)
        for tag, probability in skip_tags.items():
            if probability == 0.0:
                continue
            mask = df["tag_string"].str.contains(f"\\b{tag}\\b", case=False, na=False)
            indices = df.index[mask]
            if len(indices) > 0:
                skip_rolls = rng.rand(len(indices))
                indices_to_skip = indices[skip_rolls <= probability]
                rows_to_keep.loc[indices_to_skip] = False
        df = df[rows_to_keep].copy()

    images_per_class = total_images // 4
    score_thresholds = np.percentile(df["score"].values, [0, 20, 60, 92, 100])
    fav_thresholds = np.percentile(df["fav_count"].values, [0, 20, 60, 92, 100])

    bucket_criteria = [
        (df["score"] < score_thresholds[1]) & (df["fav_count"] < fav_thresholds[1]),
        (df["score"] >= score_thresholds[1])
        & (df["score"] < score_thresholds[2])
        & (df["fav_count"] >= fav_thresholds[1])
        & (df["fav_count"] < fav_thresholds[2]),
        (df["score"] >= score_thresholds[2])
        & (df["score"] < score_thresholds[3])
        & (df["fav_count"] >= fav_thresholds[2])
        & (df["fav_count"] < fav_thresholds[3]),
        (df["score"] >= score_thresholds[3]) & (df["fav_count"] >= fav_thresholds[3]),
    ]

    stats = {i: {} for i in range(4)}
    sampled_dfs = []

    for bucket_id, criteria in enumerate(bucket_criteria):
        bucket_df = df[criteria].copy()
        if len(bucket_df) == 0 or images_per_class == 0:
            continue

        sampled = bucket_df.sample(
            min(len(bucket_df), images_per_class), random_state=rng
        )
        sampled["aesthetic_class"] = bucket_id
        sampled["aesthetic_name"] = ["worst", "worse", "better", "best"][bucket_id]
        sampled_dfs.append(sampled)
        stats[bucket_id]["sampled"] = len(sampled)

    if not sampled_dfs:
        return pd.DataFrame(
            columns=df.columns.tolist() + ["aesthetic_class", "aesthetic_name"]
        ), stats

    result_df = pd.concat(sampled_dfs, ignore_index=True)
    return result_df, stats


def download_images(
    df: pd.DataFrame, output_dir: str, max_workers: int = 8, timeout: int = 10
) -> list:
    import requests
    from concurrent.futures import ThreadPoolExecutor
    import urllib.parse
    import time

    os.makedirs(output_dir, exist_ok=True)
    for class_id in range(4):
        os.makedirs(os.path.join(output_dir, str(class_id)), exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    def download_image(row):
        try:
            url, class_id, img_id = row["file_url"], row["aesthetic_class"], row["id"]
            ext = os.path.splitext(urllib.parse.urlparse(url).path.lower())[1]
            if ext not in valid_extensions:
                return None

            save_path = os.path.join(output_dir, str(class_id), f"{img_id}{ext}")
            if os.path.exists(save_path):
                return save_path

            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            with open(save_path + ".tmp", "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            os.rename(save_path + ".tmp", save_path)
            return save_path
        except Exception as e:
            return None

    downloaded_paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, row) for _, row in df.iterrows()]
        for future in futures:
            if result := future.result():
                downloaded_paths.append(result)

    return downloaded_paths
