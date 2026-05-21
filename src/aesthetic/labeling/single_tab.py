import os
import csv
import time
import gradio as gr
from pathlib import Path
from aesthetic.labeling.utils import dirwalk, format_time


class ImageLabeler:
    def __init__(self, images_folder, output_file="labels.csv"):
        self.images_folder = Path(images_folder)
        self.output_csv_file = output_file
        self.images = self._get_all_images()
        self.labels = {}
        self.current_index = 0
        if not self.images:
            print("Warning: No images found. Labeling cannot proceed.")
            return
        self._load_existing_labels()

    def _get_all_images(self):
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"}
        all_images = []
        base_path = Path(self.images_folder)

        if not base_path.exists() or not base_path.is_dir():
            return []

        def is_valid_image(path):
            return path.is_file() and path.suffix.lower() in image_extensions

        for image_path in dirwalk(base_path, is_valid_image):
            all_images.append(str(image_path))
        all_images.sort()
        return all_images

    def _load_existing_labels(self):
        if os.path.exists(self.output_csv_file):
            try:
                with open(self.output_csv_file, "r", newline="", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    for row in reader:
                        if len(row) == 2:
                            self.labels[row[0]] = int(row[1])

                labeled_indices = [
                    self.images.index(p) for p in self.labels if p in self.images
                ]
                if labeled_indices:
                    self.current_index = max(labeled_indices) + 1
                    if self.current_index >= len(self.images):
                        self.current_index = 0
            except Exception as e:
                print(f"Error loading labels from CSV: {e}. Starting fresh.")
                self.labels = {}
                self.current_index = 0

    def _save_label(self, image_path, score):
        file_exists = os.path.exists(self.output_csv_file)
        with open(self.output_csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists or os.path.getsize(self.output_csv_file) == 0:
                writer.writerow(["image_path", "label"])
            writer.writerow([image_path, score])

    def get_current_image(self):
        if not self.images:
            return None
        start_index = self.current_index
        while True:
            current_image = self.images[self.current_index]
            if current_image not in self.labels:
                return current_image
            self.current_index = (self.current_index + 1) % len(self.images)
            if self.current_index == start_index:
                return None

    def label_image(self, score):
        if not self.images or self.current_index >= len(self.images):
            return None, f"Finished! ({len(self.labels)} labeled)", len(self.labels)

        current_image = self.images[self.current_index]
        if current_image not in self.labels:
            self.labels[current_image] = score
            self._save_label(current_image, score)

        self.current_index = (self.current_index + 1) % len(self.images)
        next_image = self.get_current_image()
        total = len(self.labels)
        progress = (
            f"Image {self.current_index + 1}/{len(self.images)} ({total} labeled)"
        )
        return next_image, progress, total


def build_single_tab(labeler: ImageLabeler):
    """Builds the Gradio UI components for the 4-class single labeling tab."""
    initial_image = labeler.get_current_image()
    initial_progress = f"Image {labeler.current_index + 1}/{len(labeler.images)} ({len(labeler.labels)} labeled)"

    with gr.Tab("Single Labeling"):
        image_start_time = gr.State(value=time.time())
        total_labeling_time = gr.State(0.0)
        images_labeled_session = gr.State(0)
        session_start_time = gr.State(value=time.time())

        with gr.Row():
            image_display = gr.Image(
                label="Current Image",
                value=initial_image,
                show_download_button=False,
                height=512,
                interactive=False,
            )

        with gr.Row():
            progress_text = gr.Textbox(
                label="Progress", value=initial_progress, interactive=False, scale=3
            )
            timer_text = gr.Textbox(
                label="Image Time", value="Last: 0.00s", interactive=False, scale=1
            )
            avg_time_text = gr.Textbox(
                label="Session Avg", value="Avg: 0.00s", interactive=False, scale=1
            )

        with gr.Row():
            btn_worst = gr.Button("Worst (1)", variant="stop", elem_id="btn_worst")
            btn_worse = gr.Button("Worse (2)", variant="secondary", elem_id="btn_worse")
            btn_better = gr.Button(
                "Better (3)", variant="secondary", elem_id="btn_better"
            )
            btn_best = gr.Button("Best (4)", variant="success", elem_id="btn_best")

        session_stats_display = gr.Textbox(
            label="Session Summary",
            value="Session Stats: Labeled: 0 | Avg Time: 0.00s | Total Time: 00:00:00",
            interactive=False,
        )

        def label_callback(
            score,
            current_start_time,
            session_start_time,
            total_session_time,
            count_session,
        ):
            end_time = time.time()
            time_spent = end_time - current_start_time
            next_image, base_progress, _ = labeler.label_image(score)

            new_count = count_session + 1
            new_total_time = total_session_time + time_spent
            avg_time = new_total_time / new_count if new_count > 0 else 0
            total_elapsed = time.time() - session_start_time

            timer_str = f"Last: {time_spent:.2f}s"
            avg_time_str = f"Avg: {avg_time:.2f}s"
            stats_str = f"Session Stats: Labeled: {new_count} | Avg Time: {avg_time:.2f}s | Total Time: {format_time(total_elapsed)}"

            if next_image is None:
                base_progress = f"All {len(labeler.images)} images labeled!"
                timer_str = ""

            return (
                next_image,
                base_progress,
                timer_str,
                avg_time_str,
                stats_str,
                time.time(),
                new_total_time,
                new_count,
            )

        outputs = [
            image_display,
            progress_text,
            timer_text,
            avg_time_text,
            session_stats_display,
            image_start_time,
            total_labeling_time,
            images_labeled_session,
        ]
        inputs = [
            image_start_time,
            session_start_time,
            total_labeling_time,
            images_labeled_session,
        ]

        btn_worst.click(
            lambda *s: label_callback(0, *s), inputs=inputs, outputs=outputs
        )
        btn_worse.click(
            lambda *s: label_callback(1, *s), inputs=inputs, outputs=outputs
        )
        btn_better.click(
            lambda *s: label_callback(2, *s), inputs=inputs, outputs=outputs
        )
        btn_best.click(lambda *s: label_callback(3, *s), inputs=inputs, outputs=outputs)
