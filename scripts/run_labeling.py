import os
import argparse
from aesthetic.labeling.app import create_app


def main():
    parser = argparse.ArgumentParser(description="Run Aesthetic Labeling App")
    parser.add_argument(
        "--images_folder", type=str, default="//Data/evangelion", help="Path to images"
    )
    parser.add_argument(
        "--output_csv", type=str, default="aesthetic/aesthetic_labels.csv"
    )
    parser.add_argument(
        "--final_tiers_csv",
        type=str,
        default="reports/evangelion_2026_05_04/final_tiers.csv",
    )
    parser.add_argument(
        "--prior_knowledge_csv", type=str, default="cleaned_prior_knowledge.csv"
    )
    parser.add_argument(
        "--pairs_output", type=str, default="aesthetic/labels_pairs.json"
    )
    parser.add_argument("--config_path", type=str, default="configs/config.yaml")
    parser.add_argument("--num_pairs_per_image", type=int, default=4)
    parser.add_argument(
        "--port", type=int, default=1234, help="Port to run the Gradio app"
    )
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio share link"
    )
    args = parser.parse_args()

    app = create_app(
        images_folder=args.images_folder,
        output_file=args.output_csv,
        final_tiers_csv=args.final_tiers_csv,
        prior_knowledge_csv=args.prior_knowledge_csv,
        pairs_output=args.pairs_output,
        config_path=args.config_path,
        num_pairs_per_image=args.num_pairs_per_image,
    )

    print("\n--- Starting Gradio App ---")
    print(f"Images Folder: {args.images_folder}")
    print("Use keys 1 (Worst) to 4 (Best) for single labeling, A/D for pair labeling.")

    try:
        app.launch(
            share=args.share,
            server_port=args.port,
            inbrowser=True,
            prevent_thread_lock=True,
            debug=True,
            allowed_paths=[args.images_folder, "aesthetic_output"],
        )
    except KeyboardInterrupt:
        print("\nShutting down the app...")


if __name__ == "__main__":
    main()
