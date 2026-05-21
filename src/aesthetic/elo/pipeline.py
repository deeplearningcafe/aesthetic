import json
import torch
import h5py
import pandas as pd
import matplotlib.pyplot as plt
from aesthetic.training.cache import FeatureCacher
from aesthetic.training.models.pair_cls import PairClassifier


class PreFilter:
    def filter(self, input_csv: str, output_json: str) -> None:
        df = pd.read_csv(input_csv)

        if "label" in df.columns:
            valid = df[df["label"] >= 2]
            paths = valid["image_path"].tolist()
        elif "aesthetic_label" in df.columns:
            # only masterpiece
            valid = df[df["aesthetic_label"].isin(["best"])]
            paths = valid["relative_path"].tolist()
        else:
            raise ValueError("CSV must contain 'label' or 'aesthetic_label'")

        # FeatureCacher expects a dict of path -> label. Dummy label -1
        out_dict = {p: -1 for p in paths}
        with open(output_json, "w") as f:
            json.dump(out_dict, f, indent=2)
        print(f"Filtered {len(df)} down to {len(paths)} high-quality images.")


class FeatureExtractorCache:
    def __init__(self, model_dir: str, device: str = "cuda"):
        self.model_dir = model_dir
        self.device = device

    def extract(
        self,
        input_json: str,
        out_h5: str,
        out_meta: str,
        global_path: str = None,
        batch_size: int = 32,
    ) -> None:
        cacher = FeatureCacher(model_dir=self.model_dir, device=self.device)
        cacher.cache_dataset(
            json_path=input_json,
            output_h5=out_h5,
            output_meta=out_meta,
            global_path=global_path,
            batch_size=batch_size,
        )


class EloArena:
    def __init__(self, model_path: str, feature_dim: int = 1024, device: str = "cuda"):
        self.device = device
        self.model = PairClassifier(feature_dim=feature_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        self.model.requires_grad_(False)

    def simulate(
        self,
        h5_path: str,
        meta_path: str,
        out_json: str,
        num_rounds: int = 100,
        batch_size: int = 4096,
    ) -> None:
        with open(meta_path, "r") as f:
            meta = json.load(f)["sample_mapping"]

        keys = list(meta.keys())
        n_samples = len(keys)
        h5_indices = [meta[k]["h5_index"] for k in keys]

        print(f"Loading {n_samples} features into VRAM for O(1) batching...")
        with h5py.File(h5_path, "r") as f:
            h5_feat = f["features"]
            all_features = torch.from_numpy(h5_feat[:])

        features = all_features[h5_indices].to(self.device)
        elo_scores = torch.full((n_samples,), 1200.0, device=self.device)

        print(f"Starting {num_rounds} rounds of Swiss-system matchmaking...")
        for r in range(num_rounds):
            # K-factor decay for stabilization
            k_factor = (
                32 if r < num_rounds // 3 else (16 if r < 2 * num_rounds // 3 else 8)
            )

            # Swiss pairing: sort by ELO + noise to pair similar images
            noise = torch.randn(n_samples, device=self.device) * 50
            sort_idx = torch.argsort(elo_scores + noise)

            n_pairs = n_samples // 2
            p1_idx = sort_idx[0::2][:n_pairs]
            p2_idx = sort_idx[1::2][:n_pairs]

            for i in range(0, n_pairs, batch_size):
                b_p1 = p1_idx[i : i + batch_size]
                b_p2 = p2_idx[i : i + batch_size]

                emb1 = features[b_p1]
                emb2 = features[b_p2]

                with torch.no_grad():
                    logits = self.model(emb1, emb2)
                    p_A = torch.softmax(logits, dim=1)[:, 0]

                # Determine deterministic winner based on model probability
                S_A = (p_A > 0.5).float()
                S_B = 1.0 - S_A

                R_A = elo_scores[b_p1]
                R_B = elo_scores[b_p2]

                # Standard ELO Expected Score formula
                E_A = 1.0 / (1.0 + 10.0 ** ((R_B - R_A) / 400.0))
                E_B = 1.0 - E_A

                elo_scores[b_p1] = R_A + k_factor * (S_A - E_A)
                elo_scores[b_p2] = R_B + k_factor * (S_B - E_B)

        # Export sorted results
        results = {keys[i]: elo_scores[i].item() for i in range(n_samples)}
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Tournament complete. ELOs saved to {out_json}")


class DatasetSelector:
    def select(
        self,
        elo_json: str,
        threshold: float,
        out_csv: str,
        plot_path: str = "elo_dist.png",
    ) -> None:
        with open(elo_json, "r") as f:
            elos = json.load(f)

        df = pd.DataFrame(list(elos.items()), columns=["image_path", "elo"])

        # Debugging plot
        plt.figure(figsize=(10, 6))
        plt.hist(df["elo"], bins=50, color="skyblue", edgecolor="black")
        plt.axvline(threshold, color="red", linestyle="dashed", linewidth=2)
        plt.title("ELO Score Distribution")
        plt.xlabel("ELO Score")
        plt.ylabel("Frequency")
        plt.savefig(plot_path)
        plt.close()

        selected = df[df["elo"] >= threshold]
        selected.to_csv(out_csv, index=False)
        print(f"Dataset generated: {len(selected)} images above {threshold}.")
        print(f"Distribution plot saved to {plot_path}")
