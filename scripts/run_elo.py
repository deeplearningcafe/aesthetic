import argparse
from aesthetic.elo.pipeline import (
    PreFilter,
    FeatureExtractorCache,
    EloArena,
    DatasetSelector,
)
from aesthetic.elo.analyze_imgs import EloAnalyzer


def main():
    parser = argparse.ArgumentParser(description="ELO Ranking Pipeline")
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4, 5],
        help="Phases to run (1:Filter, 2:Cache, 3:Arena, 4:Bin, 5:Debug)",
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV from the 4-class classifier",
    )
    parser.add_argument("--filtered_json", type=str, default="filtered.json")
    parser.add_argument("--swin_dir", type=str, default="model/wd_swinv2")
    parser.add_argument("--global_path", type=str, default="")
    parser.add_argument("--h5_path", type=str, default="features_elo.h5")
    parser.add_argument("--meta_path", type=str, default="meta_elo.json")
    parser.add_argument(
        "--pair_model",
        type=str,
        required=True,
        help="Path to the trained PairClassifier .pth",
    )
    parser.add_argument("--elo_json", type=str, default="elos.json")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2000,
        help="Number of samples to be included in final dataset",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        help="Minimum ELO score to be included in final dataset",
    )
    parser.add_argument("--out_csv", type=str, default="hq_dataset.csv")
    parser.add_argument(
        "--rounds", type=int, default=100, help="Number of tournament rounds per image"
    )
    parser.add_argument(
        "--out_debug",
        type=str,
        default="elo_analysis",
        help="Directory to save the generated plots",
    )

    args = parser.parse_args()
    # TODO add mkdir for the outputs

    if 1 in args.phases:
        print("\n--- Phase 1: Pre-Filtering (The Gatekeeper) ---")
        pf = PreFilter()
        pf.filter(args.input_csv, args.filtered_json)

    if 2 in args.phases:
        print("\n--- Phase 2: Feature Extraction (Cache) ---")
        fe = FeatureExtractorCache(model_dir=args.swin_dir)
        fe.extract(args.filtered_json, args.h5_path, args.meta_path, args.global_path)

    if 3 in args.phases:
        print("\n--- Phase 3: ELO Arena (Head-to-Head Simulation) ---")
        arena = EloArena(model_path=args.pair_model)
        arena.simulate(
            args.h5_path, args.meta_path, args.elo_json, num_rounds=args.rounds
        )

    if 4 in args.phases:
        print("\n--- Phase 4: Binning and Dataset Selection ---")
        ds = DatasetSelector()
        ds.select(args.elo_json, args.out_csv, args.threshold, args.num_samples)

    if 5 in args.phases:
        print("\n--- Phase 5: Debugging Dataset ---")
        analyzer = EloAnalyzer(args.elo_json, args.global_path, args.out_debug)
        analyzer.run_all()


if __name__ == "__main__":
    main()
