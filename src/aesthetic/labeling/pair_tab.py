import json
import random
import pandas as pd
import numpy as np
import yaml
import re
import os
import gradio as gr
from pathlib import Path
from aesthetic.labeling.utils import dirwalk


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
        num_pairs_per_image=1,
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
        self.num_pairs_per_image = num_pairs_per_image

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

    def _generate_pairs(
        self, final_tiers_csv, prior_knowledge_csv, num_pairs_per_image=1
    ):
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}
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

        # Handle dynamic number of pairs per image
        self.pairs = []
        if self.num_pairs_per_image <= 1:
            for i in range(0, len(sampled_list) - 1, 2):
                p1 = id_to_path[sampled_list[i]]
                p2 = id_to_path[sampled_list[i + 1]]
                self.pairs.append((p1, p2))
        else:
            # Pair each image with N random other images
            for i in range(len(sampled_list)):
                for _ in range(self.num_pairs_per_image):
                    j = rng.randint(0, len(sampled_list))
                    if i != j:
                        p1 = id_to_path[sampled_list[i]]
                        p2 = id_to_path[sampled_list[j]]
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

        # Safe extraction of ID
        try:
            winner_id = int(Path(winner_path).stem)
        except ValueError:
            winner_id = -1

        self.labels[pair_key] = {
            "winner": winner_idx,
            "left_path": left_path,
            "right_path": right_path,
            "selected_tier": self.id_to_tier.get(winner_id, "unknown"),
        }

        with open(self.output_file, "w", encoding="utf-8") as f:
            json.dump(self.labels, f, indent=2)

        self.current_index += 1
        next_left, next_right = self.get_current_pair()
        progress = f"Pair {self.current_index + 1}/{len(self.pairs)} ({len(self.labels)} labeled)"
        return next_left, next_right, progress, len(self.labels)

    def reshuffle_pairs(self):
        """Reshuffles the remaining unlabeled pairs."""
        unlabeled = [p for p in self.pairs if f"{p[0]}|{p[1]}" not in self.labels]
        random.shuffle(unlabeled)

        # Reconstruct pairs list: labeled first, then shuffled unlabeled
        labeled = [p for p in self.pairs if f"{p[0]}|{p[1]}" in self.labels]
        self.pairs = labeled + unlabeled
        self._sync_index()
        return "Pairs reshuffled successfully."


def build_pair_tab(pair_labeler: PairImageLabeler):
    """Builds the Gradio UI components for the pair labeling tab."""
    with gr.Tab("Pair Labeling (Elo)"):
        initial_left, initial_right = pair_labeler.get_current_pair()
        initial_progress = f"Pair {pair_labeler.current_index + 1}/{len(pair_labeler.pairs)} ({len(pair_labeler.labels)} labeled)"

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
                label="Progress", value=initial_progress, interactive=False
            )

        with gr.Row():
            btn_left = gr.Button(
                "Left is Better (A)", variant="primary", elem_id="btn_left"
            )
            btn_right = gr.Button(
                "Right is Better (D)", variant="primary", elem_id="btn_right"
            )
            reshuffle_btn = gr.Button(
                "Reshuffle Pairs", variant="secondary", elem_id="btn_reshuffle"
            )

        def pair_label_callback(winner_idx):
            next_left, next_right, progress, _ = pair_labeler.label_pair(winner_idx)
            return next_left, next_right, progress

        def reshuffle_callback():
            msg = pair_labeler.reshuffle_pairs()
            next_left, next_right = pair_labeler.get_current_pair()
            progress = f"Pair {pair_labeler.current_index + 1}/{len(pair_labeler.pairs)} ({len(pair_labeler.labels)} labeled) - {msg}"
            return next_left, next_right, progress

        btn_left.click(
            lambda: pair_label_callback(0),
            outputs=[image_left, image_right, progress_text_pairs],
        )
        btn_right.click(
            lambda: pair_label_callback(1),
            outputs=[image_left, image_right, progress_text_pairs],
        )

        reshuffle_btn.click(
            reshuffle_callback, outputs=[image_left, image_right, progress_text_pairs]
        )
