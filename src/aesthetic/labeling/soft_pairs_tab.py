import gradio as gr
from aesthetic.labeling.soft_pairs import generate_soft_pairs_from_4class


def build_soft_pairs_tab():
    """Builds the Gradio UI components for generating soft pairs."""
    with gr.Tab("Soft Pairs Generation"):
        gr.Markdown(
            "### Generate Soft Pairs\n"
            "Generate artificial pairs comparing Class 1 (worse) and Class 3 (best) from your 4-class labeled data."
        )

        with gr.Row():
            with gr.Column():
                csv_input = gr.Textbox(
                    label="Input 4-Class CSV Path",
                    value="aesthetic/aesthetic_labels.csv",
                )
                json_output = gr.Textbox(
                    label="Output Pairs JSON Path", value="aesthetic/soft_pairs.json"
                )
                num_pairs_input = gr.Number(
                    label="Number of Pairs to Generate", value=1000, precision=0
                )

        generate_btn = gr.Button("Generate Soft Pairs", variant="primary")
        status_output = gr.Textbox(label="Status", interactive=False)

        def generate_handler(csv_path, json_path, num_pairs):
            try:
                generate_soft_pairs_from_4class(csv_path, json_path, int(num_pairs))
                return f"Successfully generated {int(num_pairs)} pairs at {json_path}"
            except Exception as e:
                return f"Error during generation: {str(e)}"

        generate_btn.click(
            fn=generate_handler,
            inputs=[csv_input, json_output, num_pairs_input],
            outputs=[status_output],
        )
