import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


class EloAnalyzer:
    def __init__(self, elo_json: str, global_path: str, output_dir: str):
        self.elo_json = elo_json
        self.global_path = global_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        with open(self.elo_json, "r") as f:
            elos = json.load(f)

        self.df = pd.DataFrame(list(elos.items()), columns=["image_path", "elo"])
        self.df = self.df.sort_values("elo", ascending=False).reset_index(drop=True)

    def _load_image(self, path: str) -> Image.Image:
        """Loads an image. Fails loudly if path is invalid."""
        full_path = os.path.join(self.global_path, path)
        return Image.open(full_path).convert("RGB")

    def plot_quantiles(self, num_quantiles: int = 5, samples_per_q: int = 4):
        """Plots a grid of images divided by ELO quantiles."""
        self.df["quantile"] = pd.qcut(self.df["elo"], q=num_quantiles, labels=False)

        fig, axes = plt.subplots(
            num_quantiles, samples_per_q, figsize=(15, 3 * num_quantiles)
        )
        fig.suptitle("ELO Quantiles (Top row = Best, Bottom row = Worst)", fontsize=16)

        # Reverse so quantile 4 (best) is at the top row (idx 0)
        for row_idx, q in enumerate(reversed(range(num_quantiles))):
            q_df = self.df[self.df["quantile"] == q]
            samples = q_df.sample(n=samples_per_q, replace=False)

            for col_idx, (_, row) in enumerate(samples.iterrows()):
                ax = axes[row_idx, col_idx]
                img = self._load_image(row["image_path"])
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"ELO: {row['elo']:.1f}", fontsize=10)

                if col_idx == 0:
                    ax.text(
                        -0.1,
                        0.5,
                        f"Q{q + 1}",
                        transform=ax.transAxes,
                        fontsize=14,
                        va="center",
                        ha="right",
                        rotation=90,
                    )

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "quantiles_grid.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved quantiles grid to {out_path}")

    def plot_similar_elos(self, num_anchors: int = 4, neighbors: int = 4):
        """Plots random anchor images alongside their closest ELO neighbors."""
        fig, axes = plt.subplots(num_anchors, neighbors, figsize=(15, 3 * num_anchors))
        fig.suptitle("Similar ELO Comparisons (Neighbors)", fontsize=16)

        # Pick random anchor indices spread across the dataset
        step = len(self.df) // num_anchors
        anchor_indices = [
            np.random.randint(i * step, (i + 1) * step - neighbors)
            for i in range(num_anchors)
        ]

        for row_idx, anchor_idx in enumerate(anchor_indices):
            # Take the anchor and its immediate neighbors in the sorted DF
            group = self.df.iloc[anchor_idx : anchor_idx + neighbors]

            for col_idx, (_, row) in enumerate(group.iterrows()):
                ax = axes[row_idx, col_idx]
                img = self._load_image(row["image_path"])
                ax.imshow(img)
                ax.axis("off")

                title = "Anchor\n" if col_idx == 0 else "Neighbor\n"
                title += f"ELO: {row['elo']:.1f}"
                ax.set_title(title, fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "similar_elos.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved similar ELOs grid to {out_path}")

    def plot_elo_gaps(self, num_pairs: int = 4):
        """Plots direct comparisons between High ELO and Low ELO images."""
        fig, axes = plt.subplots(num_pairs, 2, figsize=(10, 4 * num_pairs))
        fig.suptitle("High ELO vs Low ELO Comparisons", fontsize=16)

        # Compare top 20% vs bottom 20%
        top_20_idx = len(self.df) // 5
        bot_20_idx = 4 * len(self.df) // 5

        for i in range(num_pairs):
            high_row = self.df.iloc[np.random.randint(0, top_20_idx)]
            low_row = self.df.iloc[np.random.randint(bot_20_idx, len(self.df))]

            # Plot High ELO
            ax_high = axes[i, 0]
            img_high = self._load_image(high_row["image_path"])
            ax_high.imshow(img_high)
            ax_high.axis("off")
            ax_high.set_title(f"High ELO: {high_row['elo']:.1f}", fontsize=12)

            # Plot Low ELO
            ax_low = axes[i, 1]
            img_low = self._load_image(low_row["image_path"])
            ax_low.imshow(img_low)
            ax_low.axis("off")
            ax_low.set_title(f"Low ELO: {low_row['elo']:.1f}", fontsize=12)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "elo_gaps.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        print(f"Saved ELO gaps comparison to {out_path}")

    def run_all(self):
        self.plot_quantiles()
        self.plot_similar_elos()
        self.plot_elo_gaps()
