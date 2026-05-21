import gradio as gr
from aesthetic.labeling.single_tab import ImageLabeler, build_single_tab
from aesthetic.labeling.pair_tab import PairImageLabeler, build_pair_tab
from aesthetic.labeling.cache_tab import build_cache_tab
from aesthetic.labeling.soft_pairs_tab import build_soft_pairs_tab


def create_app(
    images_folder="./images",
    output_file="labels.csv",
    final_tiers_csv=None,
    prior_knowledge_csv=None,
    pairs_output="labels_pairs.json",
    config_path="configs/config.yaml",
    num_pairs_per_image=1,
):
    """Creates the Gradio interface by importing modular tabs."""
    labeler = ImageLabeler(images_folder, output_file)
    pair_labeler = PairImageLabeler(
        images_folder,
        final_tiers_csv,
        prior_knowledge_csv,
        pairs_output,
        config_path,
        num_pairs_per_image=num_pairs_per_image,
    )

    with gr.Blocks(title="Aesthetic Labeler") as app:
        gr.Markdown("# Anime Image Aesthetic Labeler")

        with gr.Tabs():
            build_single_tab(labeler)
            build_pair_tab(pair_labeler)
            build_cache_tab()
            build_soft_pairs_tab()

        # Keyboard Shortcuts
        app.load(
            None,
            js="""
            function label_keydown(e) {
                if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA') return;
                if (['1', '2', '3', '4'].includes(e.key)) {
                    e.preventDefault();
                    const btnIds =['btn_worst', 'btn_worse', 'btn_better', 'btn_best'];
                    const btn = document.getElementById(btnIds[parseInt(e.key)-1]);
                    if (btn && btn.offsetParent !== null) btn.click();
                }
                if (e.key.toLowerCase() === 'a') {
                    const btnLeft = document.getElementById('btn_left');
                    if (btnLeft && btnLeft.offsetParent !== null) btnLeft.click();
                }
                if (e.key.toLowerCase() === 'd') {
                    const btnRight = document.getElementById('btn_right');
                    if (btnRight && btnRight.offsetParent !== null) btnRight.click();
                }
            }
            document.addEventListener('keydown', label_keydown);
            return () => { document.removeEventListener('keydown', label_keydown); }
        """,
        )

    return app
