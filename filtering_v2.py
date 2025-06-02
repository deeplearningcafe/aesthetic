import pandas as pd
import numpy as np
import os
from typing import Tuple, List, Optional, Dict

def filter_dataset(
    df: pd.DataFrame, 
    total_images: int,
    ratings_percentage: Dict[str, float],
    skip_tags: Optional[Dict[str, float]],
    exclude_df: Optional[pd.DataFrame] = None,
    random_seed: int = 42,
    id_proximity_threshold: int = 5
) -> Tuple[pd.DataFrame, dict]:
    # --- START OF CHANGES: Column Handling and Initial Setup ---
    np.random.seed(random_seed)
    rng = np.random.RandomState(random_seed)

    df.dropna(subset=['file_url'], inplace=True) # Ensure file_url is present

    # Define base required columns for core functionality
    base_required_cols = [
        "id", "file_url", "score", "fav_count", "rating", 
        "tag_string"
    ]
    
    # Define optional columns for enhanced filtering features
    optional_feature_cols = [
        "is_deleted", "is_banned", "parent_id", "md5"
        # "has_children" is not directly used in the improved logic if
        # parent_id is present and reliable for identifying relationships.
    ]

    # Check for base required columns
    missing_base_cols = [
        col for col in base_required_cols if col not in df.columns
    ]
    if missing_base_cols:
        raise ValueError(
            f"Core required columns {missing_base_cols} not found in dataframe."
        )

    # Select columns: start with base, add optional ones if they exist
    cols_to_select = list(base_required_cols) # Make a copy
    for col in optional_feature_cols:
        if col in df.columns and col not in cols_to_select:
            cols_to_select.append(col)
    
    df = df[cols_to_select].copy()
    # --- END OF CHANGES: Column Handling and Initial Setup ---

    if not np.isclose(sum(ratings_percentage.values()), 1.0):
        raise ValueError(
            "Values in ratings_percentage must sum to approximately 1.0. "
            f"Current sum: {sum(ratings_percentage.values())}"
        )

    # --- START OF CHANGES: Pre-filtering Steps ---
    print(f"Initial dataframe size: {len(df)}")

    # STEP 1: Filter out deleted or banned images
    if "is_deleted" in df.columns:
        initial_count = len(df)
        df = df[~df["is_deleted"].fillna(False)]
        print(f"Removed {initial_count - len(df)} deleted items. "
              f"Size after: {len(df)}")
    if "is_banned" in df.columns:
        initial_count = len(df)
        df = df[~df["is_banned"].fillna(False)]
        print(f"Removed {initial_count - len(df)} banned items. "
              f"Size after: {len(df)}")

    # STEP 2: Handle MD5 duplicates (keep highest score)
    if "md5" in df.columns:
        initial_count = len(df)
        # Sort by md5 and score (descending), then keep the first (highest score)
        df.sort_values("score", ascending=False, inplace=True)
        df.drop_duplicates(subset=["md5"], keep="first", inplace=True)
        df.sort_index(inplace=True) # Restore original order if needed elsewhere
        print(f"Removed {initial_count - len(df)} MD5 duplicates. "
              f"Size after: {len(df)}")

    # STEP 3: Handle parent/child relationships: prefer children
    # Remove images that are parents IF their children are also in the DataFrame
    if "parent_id" in df.columns and "id" in df.columns:
        initial_count = len(df)
        # IDs of images that are listed as parents by other images in the df
        parent_ids_referenced_by_children = set(
            df.loc[df["parent_id"].notna(), "parent_id"].unique()
        )
        # Actual image IDs present in the df
        all_image_ids_in_df = set(df["id"])
        
        # IDs to remove: images that are parents AND are present in the df
        ids_of_parents_whose_children_are_present = \
            parent_ids_referenced_by_children.intersection(all_image_ids_in_df)
        
        if ids_of_parents_whose_children_are_present:
            df = df[~df["id"].isin(ids_of_parents_whose_children_are_present)]
            print(f"Removed {initial_count - len(df)} parent images whose "
                  f"children are present. Size after: {len(df)}")
    # --- END OF CHANGES: Pre-filtering Steps ---

    # STEP 4: Handle exclude_df (IDs from external list)
    if exclude_df is not None and not exclude_df.empty:
        # ... (this part remains unchanged from the previous AI's code) ...
        if 'id' not in exclude_df.columns:
            raise ValueError(
                "exclude_df must contain an 'id' column."
            )
        initial_count_before_exclude = len(df)
        ids_to_exclude = set(exclude_df['id'].unique())
        print(
            f"Excluding {len(ids_to_exclude)} IDs provided in exclude_df."
        )
        df = df[~df['id'].isin(ids_to_exclude)]
        num_excluded = initial_count_before_exclude - len(df)
        print(
            f"Removed {num_excluded} rows based on exclude_df IDs. "
            f"DataFrame size now: {len(df)}"
        )
    
    # STEP 5: Handle skip_tags filtering
    if skip_tags and isinstance(skip_tags, dict) and skip_tags:
        # ... (this part remains unchanged from the previous AI's code) ...
        if 'tag_string' not in df.columns: # Should be fine due to earlier checks
            raise ValueError(
                "Column 'tag_string' required for tag skipping."
            )
        initial_count = len(df)
        rows_to_keep = pd.Series(True, index=df.index)
        print("Processing tags to skip:")
        for tag, probability in skip_tags.items():
            if not (0.0 <= probability <= 1.0):
                print(
                    f"Warning: Invalid probability {probability} for tag '{tag}'. "
                    f"Skipping this tag."
                )
                continue
            if probability == 0.0:
                continue
            contains_tag_mask = df['tag_string'].str.contains(
                f'\\b{tag}\\b', case=False, na=False
            )
            indices_with_tag = df.index[contains_tag_mask]
            if len(indices_with_tag) > 0:
                skip_rolls = rng.rand(len(indices_with_tag))
                should_skip_mask = skip_rolls <= probability
                indices_to_skip = indices_with_tag[should_skip_mask]
                if len(indices_to_skip) > 0:
                    rows_to_keep.loc[indices_to_skip] = False
                    print(
                        f"  - Tag '{tag}' (Prob: {probability*100:.1f}%): "
                        f"{len(indices_to_skip)} rows marked for potential skip."
                    )
        df = df[rows_to_keep].copy()
        total_skipped = initial_count - len(df)
        print(f"\nTotal images skipped based on tags: {total_skipped}. "
              f"Size after: {len(df)}")


    # Calculate images per class
    images_per_class = total_images // 4

    # Create score and favorite buckets using percentiles
    # ... (this part remains unchanged) ...
    score_percentiles = [0, 20, 60, 92, 100]
    score_thresholds = np.percentile(df["score"].values, score_percentiles)
    fav_percentiles = [0, 20, 60, 92, 100]
    fav_thresholds = np.percentile(df["fav_count"].values, fav_percentiles)
    bucket_criteria = [
        (df["score"] < score_thresholds[1]) & 
        (df["fav_count"] < fav_thresholds[1]),
        (df["score"] >= score_thresholds[1]) & 
        (df["score"] < score_thresholds[2]) & 
        (df["fav_count"] >= fav_thresholds[1]) & 
        (df["fav_count"] < fav_thresholds[2]),
        (df["score"] >= score_thresholds[2]) & 
        (df["score"] < score_thresholds[3]) & 
        (df["fav_count"] >= fav_thresholds[2]) & 
        (df["fav_count"] < fav_thresholds[3]),
        (df["score"] >= score_thresholds[3]) & 
        (df["fav_count"] >= fav_thresholds[3])
    ]
    stats = { # ... (stats initialization unchanged) ...
        i: {
            "score_range": (
                f"< {score_thresholds[1]}" if i == 0 else
                f"> {score_thresholds[3]}" if i == 3 else
                f"{score_thresholds[i]} - {score_thresholds[i+1]}"
            ),
            "fav_range": (
                f"< {fav_thresholds[1]}" if i == 0 else
                f"> {fav_thresholds[3]}" if i == 3 else
                f"{fav_thresholds[i]} - {fav_thresholds[i+1]}"
            )
        } for i in range(4)
    }
    all_ratings = sorted(list(df["rating"].unique()))
    sampled_dfs = []
    
    for bucket_id, criteria in enumerate(bucket_criteria):
        bucket_df = df[criteria].copy()
        stats[bucket_id]["total_available"] = len(bucket_df)
        
        if len(bucket_df) == 0 or images_per_class == 0:
            # ... (unchanged) ...
            print(f"Warning: Bucket {bucket_id} has no images or target is 0.")
            continue
            
        available_ratings = sorted(list(bucket_df['rating'].unique()))
        if not available_ratings:
            # ... (unchanged) ...
            print(f"Warning: Bucket {bucket_id} has images but no ratings?")
            continue

        # Calculate desired number of samples per rating (unchanged logic)
        # ... (desired_targets, fractional_parts, remainder_to_distribute logic) ...
        desired_targets = {}
        fractional_parts = {}
        total_allocated_integer = 0
        relevant_ratings = [r for r in available_ratings if r in ratings_percentage]
        # ... (rest of target calculation logic from previous AI is fine) ...
        for rating in relevant_ratings:
            percentage = ratings_percentage[rating]
            exact_target = images_per_class * percentage
            int_target = int(exact_target)
            desired_targets[rating] = int_target
            fractional_parts[rating] = exact_target - int_target
            total_allocated_integer += int_target
        remainder_to_distribute = images_per_class - total_allocated_integer
        ratings_sorted_by_fractional = sorted(
            fractional_parts.keys(),
            key=lambda r: fractional_parts[r],
            reverse=True
        )
        if ratings_sorted_by_fractional:
            for i in range(remainder_to_distribute):
                rating_to_increment = ratings_sorted_by_fractional[
                    i % len(ratings_sorted_by_fractional)
                ]
                desired_targets[rating_to_increment] += 1
        
        # Adjust targets based on availability (unchanged logic)
        # ... (shortfall, temp_targets logic) ...
        shortfall = 0
        temp_targets = {}
        for rating in available_ratings:
            available_count = len(bucket_df[bucket_df['rating'] == rating])
            desired = desired_targets.get(rating, 0)
            take = min(desired, available_count)
            temp_targets[rating] = take
            shortfall += (desired - take)
        while shortfall > 0:
            added_in_pass = 0
            for rating in available_ratings: 
                available_count = len(bucket_df[bucket_df['rating'] == rating])
                current_take = temp_targets.get(rating, 0)
                if available_count > current_take:
                    temp_targets[rating] += 1
                    shortfall -= 1
                    added_in_pass += 1
                    if shortfall == 0: break
            if added_in_pass == 0:
                print(
                    f"Warning: Bucket {bucket_id} could not meet target "
                    f"{images_per_class}. Short by {shortfall} images."
                )
                break 
        final_targets = temp_targets

        # --- START OF CHANGES: Improved Sampling Logic per Rating ---
        sampled_bucket_dfs_for_rating = [] 
        for rating, target_count in final_targets.items():
            if target_count <= 0:
                continue

            rating_specific_df = bucket_df[
                bucket_df['rating'] == rating
            ].copy()
            
            if rating_specific_df.empty:
                continue

            samples_for_this_rating_list = []
            
            # Phase 1: Parent Group Aware Sampling
            if "parent_id" in rating_specific_df.columns:
                # Define parent_group: uses parent_id, or own id if no parent
                rating_specific_df["parent_group"] = \
                    rating_specific_df["parent_id"].fillna(
                        rating_specific_df["id"]
                    )
                
                # Get one best representative (highest score) from each group
                # Using .loc to avoid potential index issues with groupby().apply()
                group_reps_indices = rating_specific_df.groupby(
                    "parent_group"
                )["score"].idxmax()
                group_reps_df = rating_specific_df.loc[group_reps_indices]

                # Sort these representatives by score to pick the best groups first
                group_reps_df = group_reps_df.sort_values(
                    "score", ascending=False
                )
                
                # Take up to target_count from these representatives
                num_to_take_from_groups = min(len(group_reps_df), target_count)
                phase1_samples = group_reps_df.head(num_to_take_from_groups)
                samples_for_this_rating_list.append(phase1_samples)
                
                # Update remaining target and prepare for Phase 2 if needed
                remaining_target = target_count - len(phase1_samples)
                
                if remaining_target > 0:
                    # Exclude all images from parent_groups already sampled
                    sampled_parent_groups = set(phase1_samples["parent_group"])
                    eligible_for_phase2_df = rating_specific_df[
                        ~rating_specific_df["parent_group"].isin(
                            sampled_parent_groups
                        )
                    ]
                    # Also ensure we don't re-select exact IDs if somehow
                    # parent_group logic missed something (defensive)
                    eligible_for_phase2_df = eligible_for_phase2_df[
                        ~eligible_for_phase2_df["id"].isin(
                            set(phase1_samples["id"])
                        )
                    ]
                else:
                    eligible_for_phase2_df = pd.DataFrame() # No more needed
            else:
                # No parent_id info, all images go to Phase 2 (ID proximity)
                remaining_target = target_count
                eligible_for_phase2_df = rating_specific_df.copy()

            # Phase 2: ID Proximity Sampling for remaining target
            if remaining_target > 0 and not eligible_for_phase2_df.empty:
                available_for_phase2 = eligible_for_phase2_df.copy()
                phase2_samples_indices = []

                for _ in range(remaining_target):
                    if available_for_phase2.empty:
                        print(
                            f"Warning: Ran out of diverse images for bucket "
                            f"{bucket_id}, rating '{rating}' in Phase 2."
                        )
                        break
                    
                    sampled_row = available_for_phase2.sample(n=1, random_state=rng)
                    sampled_index = sampled_row.index[0]
                    sampled_id = sampled_row["id"].iloc[0]
                    phase2_samples_indices.append(sampled_index)

                    # Exclude based on ID proximity for next iterations in Phase 2
                    lower_bound = sampled_id - id_proximity_threshold
                    upper_bound = sampled_id + id_proximity_threshold
                    ids_to_drop_proximity = available_for_phase2[
                        (available_for_phase2['id'] >= lower_bound) &
                        (available_for_phase2['id'] <= upper_bound)
                    ].index
                    available_for_phase2.drop(ids_to_drop_proximity, inplace=True)
                
                if phase2_samples_indices:
                    # Use original df slice to get rows, not the modified one
                    samples_for_this_rating_list.append(
                        eligible_for_phase2_df.loc[phase2_samples_indices]
                    )
            
            if samples_for_this_rating_list:
                sampled_bucket_dfs_for_rating.append(
                    pd.concat(samples_for_this_rating_list)
                )
        # --- END OF CHANGES: Improved Sampling Logic per Rating ---

        if sampled_bucket_dfs_for_rating:
            sampled = pd.concat(sampled_bucket_dfs_for_rating)
            sampled["aesthetic_class"] = bucket_id
            sampled["aesthetic_name"] = ["worst", "worse", "better", "best"][bucket_id]
            sampled_dfs.append(sampled)
            stats[bucket_id]["sampled"] = len(sampled)
            # ... (rest of stats update unchanged) ...
            rating_dist = sampled["rating"].value_counts().to_dict()
            stats[bucket_id]["rating_distribution"] = {
                r: rating_dist.get(r, 0) for r in all_ratings
            }
        else:
            stats[bucket_id]["sampled"] = 0
            stats[bucket_id]["rating_distribution"] = {r: 0 for r in all_ratings}

    if not sampled_dfs: # Handle case where no images were sampled at all
        print("Warning: No images were sampled. Returning empty DataFrame.")
        return pd.DataFrame(columns=df.columns.tolist() + ["aesthetic_class", "aesthetic_name"]), stats

    result_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Print final stats (unchanged)
    # ... (stats printing logic) ...
    print(f"\nDataset filtering stats (Target per bucket: {images_per_class}):")
    total_sampled = 0
    final_rating_counts = {r: 0 for r in all_ratings}
    for bucket_id_stat in range(4): # Use a different var name to avoid conflict
        # Ensure stats entry exists before trying to access it
        if bucket_id_stat not in stats:
            print(f"\nClass {bucket_id_stat} ({(['worst', 'worse', 'better', 'best'][bucket_id_stat])}): No stats available (likely skipped).")
            continue

        class_name = ["worst", "worse", "better", "best"][bucket_id_stat]
        bucket_stats = stats[bucket_id_stat]
        total_sampled += bucket_stats.get('sampled', 0)
        
        print(f"\nClass {bucket_id_stat} ({class_name}):")
        print(f"  Score range: {bucket_stats['score_range']}")
        print(f"  Fav range: {bucket_stats['fav_range']}")
        print(f"  Available: {bucket_stats.get('total_available', 0)}") # Added get
        print(f"  Sampled: {bucket_stats.get('sampled', 0)}")
        
        print(f"  Rating distribution (in bucket sample):")
        bucket_rating_dist = bucket_stats.get("rating_distribution", {})
        if not bucket_rating_dist or sum(bucket_rating_dist.values()) == 0 :
             print("    None sampled or all counts zero")
        else:
            for rating_val in all_ratings: 
                count = bucket_rating_dist.get(rating_val, 0)
                if count > 0: 
                    print(f"    {rating_val}: {count}")
                    final_rating_counts[rating_val] += count
        
    print("-" * 30)
    print(f"Total images sampled: {total_sampled}")
    print("Overall rating distribution in final dataset:")
    for rating_val, count in final_rating_counts.items():
         print(f"  {rating_val}: {count}")
    print("-" * 30)

    return result_df, stats


def download_images(
    df: pd.DataFrame,
    output_dir: str,
    max_workers: int = 8,
    timeout: int = 10
) -> list:
    """
    Download images from the dataframe to the specified output directory.
    
    Args:
        df: Dataframe containing file_url column
        output_dir: Directory to save images
        max_workers: Number of parallel download workers
        timeout: Maximum time in seconds to wait for a download
    
    Returns:
        List of successfully downloaded image paths
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor
    import urllib.parse
    import time
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create class subdirectories
    for class_id in range(4):
        class_dir = os.path.join(output_dir, str(class_id))
        os.makedirs(class_dir, exist_ok=True)

    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# Function to download a single image
    def download_image(row):
        start_time = time.time()
        temp_file = None
        try:
            url = row["file_url"]
            class_id = row["aesthetic_class"]
            id = row["id"]
            
            # Extract filename from URL and check extension
            parsed_url = urllib.parse.urlparse(url)
            base_name  = os.path.basename(parsed_url.path)
            name, ext = os.path.splitext(base_name.lower())
            filename = f"{id}{ext}"

            if ext not in valid_extensions:
                print(f"Skipping invalid extension: {ext}")
                return None
            
            # Create save path with class folder
            save_path = os.path.join(output_dir, str(class_id), filename)
            temp_file = save_path + ".tmp"

            # Skip if file already exists
            if os.path.exists(save_path):
                print(f"Skipping existing file: {save_path}")
                return save_path
            
            # Download the image
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()
            
            # Download to temporary file first
            file_size = 0
            with open(temp_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    # Check if we've exceeded the timeout
                    if time.time() - start_time > timeout:
                        raise TimeoutError(
                            f"Download took longer than {timeout} seconds"
                        )
                    f.write(chunk)
                    file_size += len(chunk)
            
            # Rename to final filename
            os.rename(temp_file, save_path)
            
            # Calculate download speed
            elapsed_time = time.time() - start_time
            download_speed = file_size / elapsed_time / 1024  # KB/s
            
            print(
                f"Downloaded: {save_path} in {elapsed_time:.2f}s "
                f"({download_speed:.2f} KB/s)"
            )
            return save_path
        
        except TimeoutError as e:
            print(f"Timeout downloading from {url}: {e}")
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            return None
            
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
            return None
    
    # Download images in parallel
    downloaded_paths = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_image, row) for _, row in df.iterrows()]
        for future in futures:
            result = future.result()
            if result:
                downloaded_paths.append(result)
    
    return downloaded_paths


# --- Example Usage ---
if __name__ == "__main__":

    # TODO add the percentage of images per bucket, as the bucket 4 only has 15% of the data

    df = pd.read_parquet("train-00033-of-00035.parquet")
    print(f"Loaded dataframe with {len(df)} rows")
    exclude_df = pd.read_csv('filtered_data_train_33.csv', header=0, delimiter=',')
    total_images = 5000
    random_seed = 46

    tag_to_skip = {"2boys": 0.5, }
    # Define the desired rating distribution percentages
    # Example: 40% general, 30% sensitive, 20% questionable, 10% explicit
    custom_rating_percentages = {
        'g': 0.30,
        's': 0.30,
        'q': 0.25,
        'e': 0.15
    }
    # Ensure it sums to 1.0
    assert np.isclose(sum(custom_rating_percentages.values()), 1.0)


    print(f"Filtering dataset for {total_images} total images...")
    filtered_df, stats = filter_dataset(
        df, 
        total_images=total_images,
        ratings_percentage=custom_rating_percentages, 
        skip_tags=tag_to_skip,                  # Pass the tag
        exclude_df=exclude_df,
        random_seed=random_seed,
        id_proximity_threshold=5,
    )
    
    print(f"Filtered dataset has {len(filtered_df)} images")

    # Save filtered dataframe to CSV
    output_csv = "filtered_data_train_33_v2.csv"
    filtered_df.to_csv(output_csv, index=False)
    print(f"Saved filtered dataset to {output_csv}")
    
    download = True
    if download:
        output_dir = "data/train_33"
        print(f"Downloading images to {output_dir}...")
        downloaded = download_images(filtered_df, output_dir, max_workers=4)
        print(f"Downloaded {len(downloaded)} images successfully")
