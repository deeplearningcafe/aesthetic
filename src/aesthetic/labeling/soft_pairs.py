import pandas as pd
import json
import random
from pathlib import Path


def generate_soft_pairs_from_4class(
    csv_path: str, output_json: str, num_pairs: int = 1000
):
    """
    Generates artificial pairs comparing Class 1 (worse) and Class 3 (best).
    The Class 3 image is always the winner.
    """
    df = pd.read_csv(csv_path)
    class_1 = df[df["label"] == 1]["image_path"].tolist()
    class_3 = df[df["label"] == 3]["image_path"].tolist()

    if not class_1 or not class_3:
        print("Not enough samples in Class 1 or Class 3 to generate pairs.")
        return

    pairs = {}
    for i in range(num_pairs):
        img_worse = random.choice(class_1)
        img_best = random.choice(class_3)

        # Randomize left/right position to avoid bias
        if random.random() > 0.5:
            left, right = img_worse, img_best
            winner = 1  # Right wins
        else:
            left, right = img_best, img_worse
            winner = 0  # Left wins

        pair_key = f"{left}|{right}"
        pairs[pair_key] = {
            "winner": winner,
            "left_path": left,
            "right_path": right,
            "selected_tier": "soft_generated",
        }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    print(f"Generated {len(pairs)} soft pairs at {output_json}")
