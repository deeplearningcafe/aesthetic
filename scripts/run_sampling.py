import argparse
import yaml
import pandas as pd
from aesthetic.sampling.adapters import ParquetAdapter, PriorAdapter
from aesthetic.sampling.sampler import (
    apply_target_filters,
    filter_dataset,
    download_images,
)


def main():
    parser = argparse.ArgumentParser(description="Run Dataset Sampling")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--parquet_path", type=str, default=None, help="Override parquet path"
    )
    parser.add_argument(
        "--prior_csv", type=str, default=None, help="Override prior csv path"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Override download directory"
    )
    parser.add_argument(
        "--total_samples", type=int, default=None, help="Override total samples to pick"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    parquet_path = args.parquet_path or config.get("parquet_path", "data/parquets")
    prior_csv = args.prior_csv or config.get("prior_data", {}).get(
        "output_csv_path", ""
    )
    num_parquets = config.get("num_parquets", 1)

    print("Loading Parquet Data...")
    parquet_adapter = ParquetAdapter(parquet_path, num_parquets=num_parquets)
    df_parquet = parquet_adapter.read_data()

    print("Loading Prior Data...")
    prior_adapter = PriorAdapter(prior_csv)
    df_prior = prior_adapter.read_data()

    df = pd.concat([df_parquet, df_prior], ignore_index=True)
    df = df.drop_duplicates(subset="id", keep="last")
    print(f"Total loaded rows: {len(df)}")

    # 2. Apply Target Filters
    sampling_cfg = config.get("sampling", {})
    chars = sampling_cfg.get("character_list", [])
    artists = sampling_cfg.get("artist_list", [])
    df = apply_target_filters(df, chars, artists)
    print(f"Rows after target filtering: {len(df)}")

    # 3. Filter Dataset
    total_samples = args.total_samples or sampling_cfg.get("total_samples", 200000)
    filtered_df, stats = filter_dataset(
        df=df,
        total_images=total_samples,
        ratings_percentage=sampling_cfg.get("rating_distribution", {}),
        skip_tags=sampling_cfg.get("skip_tags", {}),
        random_seed=sampling_cfg.get("random_seed", 46),
    )

    # 4. Download Images
    download_cfg = config.get("download", {})
    output_dir = args.output_dir or config.get("download_dir", "data/downloaded")

    print(f"Downloading images to {output_dir}...")
    download_images(
        filtered_df, output_dir, max_workers=download_cfg.get("max_workers", 8)
    )


if __name__ == "__main__":
    main()
