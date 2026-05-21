# File: aesthetic/scripts/run_training.py
import os
import yaml
import argparse
from torch.utils.data import DataLoader, random_split
from aesthetic.training.models.pair_cls import PairClassifier
from aesthetic.training.models.four_cls import AestheticClassifier
from aesthetic.training.dataset.dataset import (
    PairFeatureDataset,
    FeatureDataset,
    split_train_val_pairs,
)
from aesthetic.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_train", required=True)
    parser.add_argument("--meta_train", required=True)
    parser.add_argument("--h5_val", required=False, default=None)
    parser.add_argument("--meta_val", required=False, default=None)
    parser.add_argument(
        "--pairs_json", required=False, help="Required if model_type is pair"
    )
    parser.add_argument(
        "--pairs_val_json",
        required=False,
        help="Required if using separate h5_val for pair model",
    )
    parser.add_argument("--model_type", choices=["pair", "four_class"], default="pair")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    val_dataset = None

    if args.model_type == "pair":
        if not args.pairs_json:
            raise ValueError("--pairs_json is required for pair model.")

        if args.h5_val and args.meta_val:
            if not args.pairs_val_json:
                raise ValueError("--pairs_val_json is required if providing h5_val.")
            train_dataset = PairFeatureDataset(
                h5_path=args.h5_train,
                meta_path=args.meta_train,
                pairs_json=args.pairs_json,
            )
            val_dataset = PairFeatureDataset(
                h5_path=args.h5_val,
                meta_path=args.meta_val,
                pairs_json=args.pairs_val_json,
            )
        else:
            print(
                f"No validation H5 provided. Splitting {args.pairs_json} "
                f"with ratio {args.val_ratio}..."
            )
            train_pairs_path = "train_pairs_tmp.json"
            val_pairs_path = "val_pairs_tmp.json"
            split_train_val_pairs(
                input_json=args.pairs_json,
                train_json=train_pairs_path,
                val_json=val_pairs_path,
                val_ratio=args.val_ratio,
            )

            train_dataset = PairFeatureDataset(
                h5_path=args.h5_train,
                meta_path=args.meta_train,
                pairs_json=train_pairs_path,
            )
            val_dataset = PairFeatureDataset(
                h5_path=args.h5_train,
                meta_path=args.meta_train,
                pairs_json=val_pairs_path,
            )

        model = PairClassifier(feature_dim=1024)
        save_path = "models/pair_classifier_best.pth"
    else:
        if args.h5_val and args.meta_val:
            train_dataset = FeatureDataset(args.h5_train, args.meta_train)
            val_dataset = FeatureDataset(args.h5_val, args.meta_val)
        else:
            full_dataset = FeatureDataset(args.h5_train, args.meta_train)
            val_size = int(len(full_dataset) * args.val_ratio)
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size]
            )

        model = AestheticClassifier(feature_dim=1024, num_classes=4)
        save_path = "models/four_class_classifier_best.pth"

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device="cuda",
    )

    trainer.train(epochs=args.epochs, save_path=save_path)


if __name__ == "__main__":
    main()
