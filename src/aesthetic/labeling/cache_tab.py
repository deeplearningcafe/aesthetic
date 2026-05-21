import os
import gradio as gr
from aesthetic.training.cache import FeatureCacher


def cache_ui_handler(
    model_dir,
    json_path,
    h5_path,
    meta_path,
    global_path,
    batch_size,
    progress=gr.Progress(),
):
    """Wrapper to handle UI progress updates for the caching process."""
    if not os.path.exists(json_path):
        return f"Error: JSON file '{json_path}' not found."

    try:
        cacher = FeatureCacher(model_dir=model_dir, device="cuda")

        def progress_cb(current, total):
            progress(current / total, desc=f"Extracting batch {current}/{total}")

        progress(0, desc="Loading SwinV2 Model...")
        count = cacher.cache_dataset(
            json_path=json_path,
            output_h5=h5_path,
            output_meta=meta_path,
            global_path=global_path if global_path.strip() else None,
            batch_size=int(batch_size),
            progress_callback=progress_cb,
        )
        return f"Successfully cached {count} features to {h5_path}!"
    except Exception as e:
        return f"Error during caching: {str(e)}"


def build_cache_tab():
    """Builds the Gradio UI components for the caching tab."""
    with gr.Tab("Data Preparation (Cache)"):
        gr.Markdown(
            "### Cache Dataset Features\n"
            "Extract SwinV2 image embeddings and save them to an H5 file to massively speed up training."
        )

        with gr.Row():
            with gr.Column():
                model_dir_input = gr.Textbox(
                    label="SwinV2 Model Directory", value="model/wd_swinv2"
                )
                json_input = gr.Textbox(
                    label="Input Labels JSON", value="aesthetic/labels_pairs.json"
                )
                global_path_input = gr.Textbox(
                    label="Global Images Path (Optional)", value=""
                )
                batch_size_input = gr.Number(label="Batch Size", value=32, precision=0)

            with gr.Column():
                h5_output = gr.Textbox(
                    label="Output H5 Path", value="aesthetic_output/cached_features.h5"
                )
                meta_output = gr.Textbox(
                    label="Output Meta JSON Path",
                    value="aesthetic_output/cached_meta.json",
                )

        cache_btn = gr.Button("Cache Dataset Features", variant="primary")
        cache_status = gr.Textbox(label="Status", interactive=False)

        cache_btn.click(
            fn=cache_ui_handler,
            inputs=[
                model_dir_input,
                json_input,
                h5_output,
                meta_output,
                global_path_input,
                batch_size_input,
            ],
            outputs=[cache_status],
        )
