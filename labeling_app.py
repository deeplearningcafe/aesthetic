import os
import json
import gradio as gr
from pathlib import Path
from typing import Callable, Generator, Optional
import time
import csv
import pandas as pd
import numpy as np
import yaml
import re


def dirwalk(path: Path, cond: Optional[Callable] = None) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, cond)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            yield p


class ImageLabeler:
    def __init__(self, images_folder, output_file="labels.csv"):
        """Initialize the image labeler with the folder path and output file."""
        self.images_folder = Path(images_folder)
        self.output_csv_file = output_file
        self.output_json_file = Path(output_file).with_suffix(".json")
        self.images = self._get_all_images()
        self.labels = {}
        self.current_index = 0
        if not self.images:
            print("Warning: No images found. Labeling cannot proceed.")
            return

        self._load_existing_labels()

    def _get_all_images(self):
        """
        Get all image paths from the specified folder and its subfolders.
        The directory structure should be: {base_dir}/{class_id}/images
        where class_id is 0, 1, 2, or 3.
        """
        from pathlib import Path

        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        all_images = []

        # Check if path exists and is a directory
        base_path = Path(self.images_folder)
        if not base_path.exists() or not base_path.is_dir():
            print(f"Warning: {self.images_folder} is not a valid directory")
            return []

        def is_valid_image(path):
            return path.is_file() and path.suffix.lower() in image_extensions

        for image_path in dirwalk(base_path, is_valid_image):
            all_images.append(str(image_path))

        all_images.sort()

        if all_images:
            class_counts = {}
            for img_path in all_images:
                # Extract class from path
                try:
                    path_parts = Path(img_path).parts
                    class_folder = path_parts[-2]
                    if class_folder in ["0", "1", "2", "3"]:
                        class_counts[class_folder] = (
                            class_counts.get(class_folder, 0) + 1
                        )
                except IndexError:
                    pass

            print(f"Found {len(all_images)} images across class folders:")
            for class_id, count in sorted(class_counts.items()):
                print(f"  Class {class_id}: {count} images")
        else:
            print(f"No images found in {self.images_folder} or its subfolders")

        return all_images

    def _load_existing_labels(self):
        """Load existing labels from CSV and set index to continue."""
        if os.path.exists(self.output_csv_file):
            try:
                with open(self.output_csv_file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    if header != ["image_path", "label"]:
                        print(f"Warning: Unexpected CSV header: {header}")

                    for row in reader:
                        if len(row) == 2:
                            image_path, label_str = row
                            if image_path in self.images:
                                try:
                                    self.labels[image_path] = int(label_str)
                                except ValueError:
                                    print(
                                        f"Warning: Invalid label '{label_str}'"
                                        f" for {image_path}. Skipping."
                                    )
                        else:
                            print(f"Warning: Skipping malformed row: {row}")

                # Find the highest index of labeled images to resume
                labeled_indices = []
                for labeled_path in self.labels:
                    if labeled_path in self.images:
                        try:
                            labeled_index = self.images.index(labeled_path)
                            labeled_indices.append(labeled_index)
                        except ValueError:
                            pass

                if labeled_indices:
                    self.current_index = max(labeled_indices) + 1
                    if self.current_index >= len(self.images):
                        self.current_index = 0
                        print("All images appear to be labeled based on the CSV file.")
                    else:
                        print(f"Resuming labeling from index {self.current_index}")
                else:
                    print("No previously labeled images found in CSV or paths differ.")
                    self.current_index = 0

            except FileNotFoundError:
                print("CSV file not found. Starting fresh.")
                self.labels = {}
                self.current_index = 0
            except StopIteration:
                print("CSV file is empty or contains only header. Starting fresh.")
                self.labels = {}
                self.current_index = 0
            except Exception as e:
                print(f"Error loading labels from CSV: {e}. Starting fresh.")
                self.labels = {}
                self.current_index = 0
        else:
            print("No existing CSV file found. Starting fresh.")
            self.labels = {}
            self.current_index = 0

    def _save_label(self, image_path, score):
        """Append the current label to the CSV output file."""
        file_exists = os.path.exists(self.output_csv_file)
        try:
            with open(self.output_csv_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists or os.path.getsize(self.output_csv_file) == 0:
                    writer.writerow(["image_path", "label"])
                writer.writerow([image_path, score])
        except IOError as e:
            print(f"Error saving label to CSV: {e}")

    def get_current_image(self):
        """Get the current image path based on index, skipping already labeled images."""
        if not self.images:
            return None

        # Find the next unlabeled image
        start_index = self.current_index
        while True:
            current_image = self.images[self.current_index]
            if current_image not in self.labels:
                return current_image

            self.current_index = (self.current_index + 1) % len(self.images)

            if self.current_index == start_index:
                return None

    def label_image(self, score):
        """Label the current image, save to CSV, and move to the next."""
        if not self.images or self.current_index >= len(self.images):
            progress = f"Finished or no images left. ({len(self.labels)} labeled)"
            return None, progress, len(self.labels)

        current_image = self.images[self.current_index]

        if current_image and current_image not in self.labels:
            self.labels[current_image] = score
            self._save_label(current_image, score)

            self.current_index = (self.current_index + 1) % len(self.images)
            next_image = self.get_current_image()

            total_labeled = len(self.labels)
            progress = (
                f"Image {self.current_index + 1}/{len(self.images)} "
                f"({total_labeled} labeled)"
            )

            return next_image, progress, total_labeled
        elif current_image in self.labels:
            print(
                f"Warning: Attempted to re-label already labeled image: {current_image}"
            )
            self.current_index = (self.current_index + 1) % len(self.images)
            next_image = self.get_current_image()
            total_labeled = len(self.labels)
            progress = (
                f"Image {self.current_index + 1}/{len(self.images)} "
                f"({total_labeled} labeled)"
            )
            return next_image, progress, total_labeled
        else:
            progress = f"Finished labeling. ({len(self.labels)} labeled)"
            return None, progress, len(self.labels)


class PairImageLabeler:
    def __init__(
        self,
        images_folder,
        final_tiers_csv,
        prior_knowledge_csv,
        output_file="labels_pairs.json",
        config_path="configs/config.yaml",
        artists_count=64,
        characters_count=32,
    ):
        self.images_folder = Path(images_folder)
        self.output_file = output_file
        self.config_path = config_path
        self.labels = {}
        self.pairs = []
        self.current_index = 0
        self.id_to_tier = {}
        self.artists_count = artists_count
        self.characters_count = characters_count

        self._load_existing_labels()
        self._generate_pairs(final_tiers_csv, prior_knowledge_csv)
        self._sync_index()

    def _load_existing_labels(self):
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, "r", encoding="utf-8") as f:
                    self.labels = json.load(f)
            except Exception:
                self.labels = {}

    def _generate_pairs(self, final_tiers_csv, prior_knowledge_csv):
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
        all_images = []
        for p in dirwalk(
            self.images_folder,
            lambda p: p.is_file() and p.suffix.lower() in image_extensions,
        ):
            all_images.append(str(p))

        id_to_path = {}
        for p in all_images:
            try:
                img_id = int(Path(p).stem)
                id_to_path[img_id] = p
            except ValueError:
                pass

        if not id_to_path or not final_tiers_csv or not prior_knowledge_csv:
            print("Missing images or CSV files for pair generation.")
            return

        try:
            df_tiers = pd.read_csv(final_tiers_csv)
            df_prior = pd.read_csv(prior_knowledge_csv, low_memory=False)
        except Exception as e:
            print(f"Error loading CSVs for pairs: {e}")
            return

        # Merge and filter to available images
        df = pd.merge(df_prior, df_tiers, on="id", how="inner")
        df = df[df["id"].isin(id_to_path.keys())].copy()

        self.id_to_tier = dict(zip(df["id"], df["final_tier"]))

        df["parent_group"] = df["parent_id"].fillna(df["id"])
        df = df.drop_duplicates(subset=["parent_group"])

        target_artists = []
        target_chars = []
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)
                target_artists = config.get("sampling", {}).get("artist_list", [])
                target_chars = config.get("sampling", {}).get("character_list", [])
                print(f"Loaded {len(target_artists)} artists and {len(target_chars)}.")
                print(
                    f"So final dataset is {len(target_artists) * self.artists_count + len(target_chars) * self.characters_count}"
                )
            except Exception as e:
                print(f"Error loading config for pairs: {e}")

        sampled_ids = set()
        rng = np.random.RandomState(42)

        def sample_group(group_df, target_n):
            if group_df.empty or target_n == 0:
                return []
            n_mp = target_n // 2
            n_gs = target_n - n_mp

            mp_df = group_df[group_df["final_tier"] == "masterpiece"]
            gs_df = group_df[group_df["final_tier"] == "good_score"]
            bs_df = group_df[group_df["final_tier"] == "bad_score"]

            n_mp = target_n // 2
            n_gs = target_n - n_mp
            n_bs = 0

            # if target tiers are short
            if len(mp_df) < n_mp:
                n_gs += n_mp - len(mp_df)
                n_mp = len(mp_df)

            if len(gs_df) < n_gs:
                shortfall = n_gs - len(gs_df)
                n_gs = len(gs_df)

                # excess to masterpiece
                mp_excess = len(mp_df) - n_mp
                if mp_excess > 0:
                    absorbed = min(shortfall, mp_excess)
                    n_mp += absorbed
                    shortfall -= absorbed

                n_bs += shortfall

            sampled = []
            if n_mp > 0 and not mp_df.empty:
                sampled.extend(
                    mp_df.sample(min(n_mp, len(mp_df)), random_state=rng)["id"].tolist()
                )
            if n_gs > 0 and not gs_df.empty:
                sampled.extend(
                    gs_df.sample(min(n_gs, len(gs_df)), random_state=rng)["id"].tolist()
                )
            if n_bs > 0 and not bs_df.empty:
                sampled.extend(
                    bs_df.sample(min(n_bs, len(bs_df)), random_state=rng)["id"].tolist()
                )

            if len(sampled) < target_n:
                remaining = target_n - len(sampled)
                leftover_df = group_df[~group_df["id"].isin(sampled)]
                if not leftover_df.empty:
                    sampled.extend(
                        leftover_df.sample(
                            min(remaining, len(leftover_df)), random_state=rng
                        )["id"].tolist()
                    )

            remaining_needed = target_n - len(sampled)
            if remaining_needed > 0:
                print(f"Remaining tags {remaining_needed}")

            return sampled

        # Artists Sampling
        if "tag_string_artist" in df.columns and target_artists:
            escaped_artists = [re.escape(t) for t in target_artists]
            artist_pattern = r"(?:^|\s)(?:" + "|".join(escaped_artists) + r")(?:$|\s)"

            is_artist_mask = df["tag_string_artist"].str.contains(
                artist_pattern, regex=True, na=False
            )

            df_artists = df[is_artist_mask].copy()
            df_artists["artist_tag"] = (
                df_artists["tag_string_artist"].fillna("").str.split(" ")
            )
            df_artists = df_artists.explode("artist_tag")

            df_artists = df_artists[df_artists["artist_tag"].isin(target_artists)]

            found_artists = 0
            for artist, group in df_artists.groupby("artist_tag"):
                group = group[~group["id"].isin(sampled_ids)]
                s_ids = sample_group(group, self.artists_count)
                sampled_ids.update(s_ids)
                found_artists += 1
            print(f"Created groups for {found_artists}")

        # Characters Sampling
        if "tag_string_character" in df.columns and target_chars:
            escaped_chars = [re.escape(t) for t in target_chars]
            char_pattern = r"(?:^|\s)(?:" + "|".join(escaped_chars) + r")(?:$|\s)"

            is_char_mask = df["tag_string_character"].str.contains(
                char_pattern, regex=True, na=False
            )

            df_chars = df[is_char_mask].copy()
            df_chars["char_tag"] = (
                df_chars["tag_string_character"].fillna("").str.split(" ")
            )
            df_chars = df_chars.explode("char_tag")

            df_chars = df_chars[df_chars["char_tag"].isin(target_chars)]

            found_chars = 0
            for char, group in df_chars.groupby("char_tag"):
                group = group[~group["id"].isin(sampled_ids)]
                s_ids = sample_group(group, self.characters_count)
                sampled_ids.update(s_ids)
                found_chars += 1
            print(f"Created groups for {found_chars}")

        sampled_list = list(sampled_ids)
        rng.shuffle(sampled_list)

        for i in range(0, len(sampled_list) - 1, 2):
            p1 = id_to_path[sampled_list[i]]
            p2 = id_to_path[sampled_list[i + 1]]
            self.pairs.append((p1, p2))

        print(f"Generated {len(self.pairs)} pairs for labeling.")

    def _sync_index(self):
        self.current_index = 0
        for i, pair in enumerate(self.pairs):
            pair_key = f"{pair[0]}|{pair[1]}"
            if pair_key not in self.labels:
                self.current_index = i
                break

    def get_current_pair(self):
        if self.current_index >= len(self.pairs):
            return None, None
        return self.pairs[self.current_index]

    def label_pair(self, winner_idx):
        if self.current_index >= len(self.pairs):
            return (
                None,
                None,
                f"Finished! ({len(self.labels)} labeled)",
                len(self.labels),
            )

        left_path, right_path = self.pairs[self.current_index]
        pair_key = f"{left_path}|{right_path}"

        winner_path = left_path if winner_idx == 0 else right_path
        winner_id = int(Path(winner_path).stem)
        selected_tier = self.id_to_tier.get(winner_id, "unknown")

        self.labels[pair_key] = {
            "winner": winner_idx,
            "left_path": left_path,
            "right_path": right_path,
            "selected_tier": selected_tier,
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)

        self.current_index += 1
        next_left, next_right = self.get_current_pair()
        progress = f"Pair {self.current_index + 1}/{len(self.pairs)} ({len(self.labels)} labeled)"
        return next_left, next_right, progress, len(self.labels)


def create_app(
    images_folder="./images",
    output_file="labels.csv",
    final_tiers_csv=None,
    prior_knowledge_csv=None,
    pairs_output="labels_pairs.json",
    config_path="configs/config.yaml",
):
    """Create and launch the Gradio interface for image labeling."""
    labeler = ImageLabeler(images_folder, output_file)
    pair_labeler = PairImageLabeler(
        images_folder, final_tiers_csv, prior_knowledge_csv, pairs_output, config_path
    )

    if not labeler.images and not pair_labeler.pairs:
        print(f"No images found in {images_folder}")
        return

    initial_image = labeler.get_current_image()
    initial_progress = (
        f"Image {labeler.current_index + 1}/{len(labeler.images)} "
        f"({len(labeler.labels)} labeled)"
    )

    def format_time(seconds):
        """Helper to format seconds into H:M:S"""
        if seconds < 0:
            seconds = 0
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    def label_callback(
        score, current_start_time, session_start_time, total_session_time, count_session
    ):
        end_time = time.time()
        time_spent_on_image = end_time - current_start_time

        next_image, base_progress, total_labeled_count = labeler.label_image(score)

        new_count_session = count_session + 1
        new_total_session_time = total_session_time + time_spent_on_image
        avg_time_session = (
            new_total_session_time / new_count_session if new_count_session > 0 else 0
        )
        total_elapsed_session = time.time() - session_start_time

        timer_str = f"Last: {time_spent_on_image:.2f}s"
        avg_time_str = f"Avg: {avg_time_session:.2f}s"
        session_stats_str = (
            f"Session Stats: Labeled: {new_count_session} | "
            f"Avg Time: {avg_time_session:.2f}s | "
            f"Total Time: {format_time(total_elapsed_session)}"
        )

        next_image_start_time = time.time()

        if next_image is None:
            base_progress = f"All {len(labeler.images)} images labeled!"
            timer_str = ""

        return (
            next_image,
            base_progress,
            timer_str,
            avg_time_str,
            session_stats_str,
            next_image_start_time,
            new_total_session_time,
            new_count_session,
        )

    with gr.Blocks(title="Aesthetic Labeler") as app:
        gr.Markdown("# Anime Image Aesthetic Labeler")

        image_start_time = gr.State(value=time.time())
        total_labeling_time = gr.State(0.0)
        images_labeled_session = gr.State(0)
        session_start_time = gr.State(value=time.time())

        with gr.Tabs():
            with gr.Tab("Single Labeling"):
                with gr.Row():
                    image_display = gr.Image(
                        label="Current Image",
                        value=initial_image,
                        show_download_button=False,
                        height=512,
                        interactive=False,
                    )

                with gr.Row():
                    progress_text = gr.Textbox(
                        label="Progress",
                        value=initial_progress,
                        interactive=False,
                        scale=3,
                    )
                    timer_text = gr.Textbox(
                        label="Image Time",
                        value="Last: 0.00s",
                        interactive=False,
                        scale=1,
                    )
                    avg_time_text = gr.Textbox(
                        label="Session Avg",
                        value="Avg: 0.00s",
                        interactive=False,
                        scale=1,
                    )

                with gr.Row():
                    btn_worst = gr.Button(
                        "Worst (1)", variant="stop", scale=1, elem_id="btn_worst"
                    )
                    btn_worse = gr.Button(
                        "Worse (2)", variant="secondary", scale=1, elem_id="btn_worse"
                    )
                    btn_better = gr.Button(
                        "Better (3)", variant="secondary", scale=1, elem_id="btn_better"
                    )
                    btn_best = gr.Button(
                        "Best (4)", variant="success", scale=1, elem_id="btn_best"
                    )

                session_stats_display = gr.Textbox(
                    label="Session Summary",
                    value="Session Stats: Labeled: 0 | Avg Time: 0.00s | Total Time: 00:00:00",
                    interactive=False,
                )

                outputs = [
                    image_display,
                    progress_text,
                    timer_text,
                    avg_time_text,
                    session_stats_display,
                    image_start_time,
                    total_labeling_time,
                    images_labeled_session,
                ]
                inputs = [
                    image_start_time,
                    session_start_time,
                    total_labeling_time,
                    images_labeled_session,
                ]

                btn_worst.click(
                    lambda *state: label_callback(0, *state),
                    inputs=inputs,
                    outputs=outputs,
                )
                btn_worse.click(
                    lambda *state: label_callback(1, *state),
                    inputs=inputs,
                    outputs=outputs,
                )
                btn_better.click(
                    lambda *state: label_callback(2, *state),
                    inputs=inputs,
                    outputs=outputs,
                )
                btn_best.click(
                    lambda *state: label_callback(3, *state),
                    inputs=inputs,
                    outputs=outputs,
                )

            with gr.Tab("Pair Labeling (Elo)"):
                initial_left, initial_right = pair_labeler.get_current_pair()
                initial_pair_progress = f"Pair {pair_labeler.current_index + 1}/{len(pair_labeler.pairs)} ({len(pair_labeler.labels)} labeled)"

                with gr.Row():
                    image_left = gr.Image(
                        label="Image 1 (Left)",
                        value=initial_left,
                        interactive=False,
                        height=512,
                    )
                    image_right = gr.Image(
                        label="Image 2 (Right)",
                        value=initial_right,
                        interactive=False,
                        height=512,
                    )

                with gr.Row():
                    progress_text_pairs = gr.Textbox(
                        label="Progress", value=initial_pair_progress, interactive=False
                    )

                with gr.Row():
                    btn_left = gr.Button(
                        "Left is Better (A)", variant="primary", elem_id="btn_left"
                    )
                    btn_right = gr.Button(
                        "Right is Better (D)", variant="primary", elem_id="btn_right"
                    )

                def pair_label_callback(winner_idx):
                    next_left, next_right, progress, _ = pair_labeler.label_pair(
                        winner_idx
                    )
                    return next_left, next_right, progress

                btn_left.click(
                    lambda: pair_label_callback(0),
                    outputs=[image_left, image_right, progress_text_pairs],
                )
                btn_right.click(
                    lambda: pair_label_callback(1),
                    outputs=[image_left, image_right, progress_text_pairs],
                )

        # --- Keyboard Shortcuts ---
        app.load(
            None,
            js="""
            function label_keydown(e) {
                if (document.activeElement.tagName === 'INPUT' ||
                    document.activeElement.tagName === 'TEXTAREA') {
                    return;
                }

                // Single Labeling Shortcuts
                if (['1', '2', '3', '4'].includes(e.key)) {
                    e.preventDefault();
                    const btnIds =['btn_worst', 'btn_worse', 'btn_better', 'btn_best'];
                    const btn = document.getElementById(btnIds[parseInt(e.key)-1]);
                    if (btn && btn.offsetParent !== null) {
                        btn.click();
                    }
                }

                // Pair Labeling Shortcuts
                if (e.key.toLowerCase() === 'a') {
                    const btnLeft = document.getElementById('btn_left');
                    if (btnLeft && btnLeft.offsetParent !== null) btnLeft.click();
                }
                if (e.key.toLowerCase() === 'd') {
                    const btnRight = document.getElementById('btn_right');
                    if (btnRight && btnRight.offsetParent !== null) btnRight.click();
                }
            }
            document.addEventListener('keydown', label_keydown);
            return () => {
                document.removeEventListener('keydown', label_keydown);
            }
        """,
        )

    return app


def migrate_json_to_csv(json_path, csv_path):
    """Migrates labels from a JSON file to a CSV file."""
    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}. No migration needed.")
        return False

    print(f"Migrating labels from {json_path} to {csv_path}...")
    try:
        with open(json_path, "r", encoding="utf-8") as fj:
            try:
                labels_dict = json.load(fj)
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file: {e}. Migration aborted.")
                return False

        with open(csv_path, "w", newline="", encoding="utf-8") as fc:
            writer = csv.writer(fc)
            writer.writerow(["image_path", "label"])  # Write header
            count = 0
            for image_path, label in labels_dict.items():
                writer.writerow([image_path, label])
                count += 1
        print(f"Successfully migrated {count} labels to {csv_path}.")
        return True
    except IOError as e:
        print(f"An error occurred during migration: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during migration: {e}")
        return False


def export_csv_to_json(csv_path, json_path):
    """Exports labels from a CSV file to a JSON file."""
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Cannot export.")
        return False

    print(f"Exporting labels from {csv_path} to {json_path}...")
    labels_dict = {}
    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as fc:
            reader = csv.reader(fc)
            header = next(reader)
            if header != ["image_path", "label"]:
                print(f"Warning: Unexpected CSV header: {header} during export.")

            count = 0
            for row in reader:
                if len(row) == 2:
                    image_path, label_str = row
                    try:
                        labels_dict[image_path] = int(label_str)
                        count += 1
                    except ValueError:
                        print(
                            f"Warning: Invalid label '{label_str}' for "
                            f"{image_path} found during export. Skipping."
                        )
                else:
                    print(f"Warning: Skipping malformed row during export: {row}")

        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(labels_dict, fj, indent=2)

        print(f"Successfully exported {count} labels to {json_path}.")
        return True
    except FileNotFoundError:
        print(f"CSV file not found at {csv_path}. Cannot export.")
        return False
    except StopIteration:
        print("CSV file is empty or contains only header. Exporting empty JSON.")
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump({}, fj, indent=2)
        return True
    except IOError as e:
        print(f"An error occurred during export: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during export: {e}")
        return False


if __name__ == "__main__":
    IMAGES_FOLDER = "FOLDER"
    OLD_JSON_FILE = None  # "aesthetic_labels_train.json"
    OUTPUT_CSV_FILE = "aesthetic/aesthetic_labels_pairs.csv"

    FINAL_TIERS_CSV = "final_tiers.csv"
    PRIOR_KNOWLEDGE_CSV = "cleaned_prior_knowledge.csv"
    PAIRS_OUTPUT_FILE = "aesthetic/labels_pairs.json"
    CONFIG_PATH = "configs/config.yaml"
    ARTISTS_COUNT = 32

    if OLD_JSON_FILE:
        if os.path.exists(OLD_JSON_FILE) and not os.path.exists(OUTPUT_CSV_FILE):
            migrated = migrate_json_to_csv(OLD_JSON_FILE, OUTPUT_CSV_FILE)
            if migrated:
                print(
                    f"Optional: You may want to rename or delete the old "
                    f"JSON file: {OLD_JSON_FILE}"
                )
            else:
                print(f"Migration failed. Please check the files and errors.")
                exit(1)

    app = create_app(
        IMAGES_FOLDER,
        OUTPUT_CSV_FILE,
        FINAL_TIERS_CSV,
        PRIOR_KNOWLEDGE_CSV,
        PAIRS_OUTPUT_FILE,
        CONFIG_PATH,
    )

    if app:
        print("\n--- Starting Gradio App ---")
        print(f"Images Folder: {IMAGES_FOLDER}")
        print(f"Labels CSV File: {OUTPUT_CSV_FILE}")
        print("Use keys 1 (Worst) to 4 (Best) for faster labeling.")
        print("Close the terminal or press Ctrl+C to stop the app.")

        try:
            app.launch(
                share=False,
                server_port=1234,
                inbrowser=True,
                prevent_thread_lock=True,
                debug=True,
                allowed_paths=[IMAGES_FOLDER],
            )
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Shutting down the app...")
        finally:
            print("\nApp closed.")
            final_json_path = Path(OUTPUT_CSV_FILE).with_suffix(".final.json")

            if not os.path.exists(OUTPUT_CSV_FILE):
                print(
                    f"CSV file {OUTPUT_CSV_FILE} not found. Skipping final JSON export."
                )
            else:
                final_json_path = Path(OUTPUT_CSV_FILE).with_suffix(".final.json")
                print(f"\nLabeling finished or app closed.")
                export_csv_to_json(OUTPUT_CSV_FILE, str(final_json_path))

    else:
        print("Could not create Gradio app. Check image folder and paths.")
