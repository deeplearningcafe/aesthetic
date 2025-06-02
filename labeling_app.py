import os
import json
import gradio as gr
from pathlib import Path
from typing import Callable, Generator, Optional
import time
import csv

def dirwalk(path: Path, cond: Optional[Callable] = None) -> Generator[Path, None, None]:
    for p in path.iterdir():
        if p.is_dir():
            yield from dirwalk(p, cond)
        else:
            if isinstance(cond, Callable):
                if not cond(p):
                    continue
            yield p

class ImageLabeler:
    def __init__(self, images_folder, output_file="labels.csv"):
        """Initialize the image labeler with the folder path and output file."""
        self.images_folder = Path(images_folder)
        self.output_csv_file = output_file
        self.output_json_file = Path(output_file).with_suffix('.json')
        self.images = self._get_all_images()
        self.labels = {}
        self.current_index = 0
        # Optimization: Check if images list is empty early
        if not self.images:
            print("Warning: No images found. Labeling cannot proceed.")
            return

        self._load_existing_labels()
        
    def _get_all_images(self):
        """
        Get all image paths from the specified folder and its subfolders.
        The directory structure should be: {base_dir}/{class_id}/images
        where class_id is 0, 1, 2, or 3.
        """
        from pathlib import Path
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}
        all_images = []
        
        # Check if path exists and is a directory
        base_path = Path(self.images_folder)
        if not base_path.exists() or not base_path.is_dir():
            print(f"Warning: {self.images_folder} is not a valid directory")
            return []
        
        # Define condition for image files
        def is_valid_image(path):
            return path.is_file() and path.suffix.lower() in image_extensions
        
        # Walk through all directories and collect image paths
        for image_path in dirwalk(base_path, is_valid_image):
            all_images.append(str(image_path))
        
        # Sort images to ensure consistent ordering between runs
        all_images.sort()
        
        # Print summary of found images
        if all_images:
            class_counts = {}
            for img_path in all_images:
                # Extract class from path (assuming path format: base_dir/class_id/image)
                try:
                    path_parts = Path(img_path).parts
                    class_folder = path_parts[-2]  # Second-to-last part is class folder
                    if class_folder in ['0', '1', '2', '3']:
                        class_counts[class_folder] = class_counts.get(class_folder, 0) + 1
                except IndexError:
                    pass
                    
            print(f"Found {len(all_images)} images across class folders:")
            for class_id, count in sorted(class_counts.items()):
                print(f"  Class {class_id}: {count} images")
        else:
            print(f"No images found in {self.images_folder} or its subfolders")
        
        return all_images
        
    # def _load_existing_labels(self):
    #     """Load existing labels and set current index to continue from last labeled."""
    #     if os.path.exists(self.output_file):
    #         try:
    #             with open(self.output_file, 'r') as f:
    #                 self.labels = json.load(f)
                    
    #             # Find the highest index of labeled images to resume from there
    #             labeled_indices = []
    #             for labeled_path in self.labels:
    #                 if labeled_path in self.images:
    #                     labeled_index = self.images.index(labeled_path)
    #                     labeled_indices.append(labeled_index)
                
    #             if labeled_indices:
    #                 # Start from the image after the last labeled one
    #                 self.current_index = max(labeled_indices) + 1
    #                 # Handle case where we finished labeling all images
    #                 if self.current_index >= len(self.images):
    #                     self.current_index = 0
    #         except json.JSONDecodeError:
    #             self.labels = {}
    def _load_existing_labels(self):
        """Load existing labels from CSV and set index to continue."""
        if os.path.exists(self.output_csv_file):
            try:
                with open(self.output_csv_file, 'r', newline='',
                          encoding='utf-8') as f:
                    reader = csv.reader(f)
                    header = next(reader) # Skip header
                    if header != ['image_path', 'label']:
                        print(f"Warning: Unexpected CSV header: {header}")
                        # Attempt to load anyway assuming column order
                    
                    for row in reader:
                        if len(row) == 2:
                            image_path, label_str = row
                            if image_path in self.images:
                                try:
                                    # Convert label back to integer
                                    self.labels[image_path] = int(label_str)
                                except ValueError:
                                    print(f"Warning: Invalid label '{label_str}'"
                                          f" for {image_path}. Skipping.")
                        else:
                            print(f"Warning: Skipping malformed row: {row}")

                # Find the highest index of labeled images to resume
                labeled_indices = []
                for labeled_path in self.labels:
                    # Check if the labeled path is still in the current list
                    if labeled_path in self.images:
                        try:
                            labeled_index = self.images.index(labeled_path)
                            labeled_indices.append(labeled_index)
                        except ValueError:
                             # Image path from labels file not found in current scan
                             pass 

                if labeled_indices:
                    self.current_index = max(labeled_indices) + 1
                    if self.current_index >= len(self.images):
                        # All images were labeled previously
                        self.current_index = 0 
                        print("All images appear to be labeled based on "
                              "the CSV file.")
                    else:
                         print(f"Resuming labeling from index "
                               f"{self.current_index}")
                else:
                    print("No previously labeled images found in CSV or "
                          "paths differ.")
                    self.current_index = 0

            except FileNotFoundError:
                print("CSV file not found. Starting fresh.")
                self.labels = {}
                self.current_index = 0
            except StopIteration: # Handles empty file or only header
                print("CSV file is empty or contains only header. "
                      "Starting fresh.")
                self.labels = {}
                self.current_index = 0
            except Exception as e:
                print(f"Error loading labels from CSV: {e}. "
                      "Starting fresh.")
                self.labels = {}
                self.current_index = 0
        else:
            print("No existing CSV file found. Starting fresh.")
            self.labels = {}
            self.current_index = 0


    # def _save_labels(self):
        # """Save the current labels to the output file."""
        # with open(self.output_file, 'w') as f:
        #     json.dump(self.labels, f, indent=2)
    def _save_label(self, image_path, score):
        """Append the current label to the CSV output file."""
        file_exists = os.path.exists(self.output_csv_file)
        try:
            with open(self.output_csv_file, 'a', newline='',
                      encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header only if file is new/empty
                if not file_exists or os.path.getsize(self.output_csv_file) == 0:
                    writer.writerow(['image_path', 'label'])
                writer.writerow([image_path, score])
        except IOError as e:
            print(f"Error saving label to CSV: {e}")


    def get_current_image(self):
        """Get the current image path based on index, skipping already labeled images."""
        if not self.images:
            return None
            
        # Find the next unlabeled image
        start_index = self.current_index
        while True:
            current_image = self.images[self.current_index]
            if current_image not in self.labels:
                return current_image
                
            # Move to the next image
            self.current_index = (self.current_index + 1) % len(self.images)
            
            # If we've checked all images and come back to where we started, 
            # then all images are labeled
            if self.current_index == start_index:
            #     return self.images[self.current_index]  # Return current even if labeled
                return None
            
    def label_image(self, score):
        """Label the current image, save to CSV, and move to the next."""
        # Check if there are images and if current_index is valid
        if not self.images or self.current_index >= len(self.images):
             # Handle cases where get_current_image might return None
             # or index is out of bounds after loading
            progress = f"Finished or no images left. " \
                       f"({len(self.labels)} labeled)"
            return None, progress, len(self.labels)

        current_image = self.images[self.current_index]
        # if current_image:
        #     self.labels[current_image] = score
        #     self._save_labels()
        
        # # Move to the next image, skipping already labeled ones
        # self.current_index = (self.current_index + 1) % len(self.images)
        # next_image = self.get_current_image()
        
        # # Calculate progress based on total labels saved
        # total_labeled = len(self.labels)
        # progress = f"Image {self.current_index + 1}/{len(self.images)} " \
        #            f"({total_labeled} labeled)"
                   
        # return next_image, progress, total_labeled
        # Only proceed if the image hasn't been labeled already in this session
        # (get_current_image should skip already labeled ones based on loaded data)
        if current_image and current_image not in self.labels:
            self.labels[current_image] = score
            # Save this single label immediately
            self._save_label(current_image, score)

            # Move to the next *potential* index. get_current_image will handle skips.
            self.current_index = (self.current_index + 1) % len(self.images)
            next_image = self.get_current_image()

            # Calculate progress based on total labels saved (in memory count)
            total_labeled = len(self.labels)
            progress = f"Image {self.current_index + 1}/{len(self.images)} " \
                       f"({total_labeled} labeled)"

            return next_image, progress, total_labeled
        elif current_image in self.labels:
            # This case might happen if logic allows revisiting, but
            # get_current_image should prevent it. Log if it occurs.
            print(f"Warning: Attempted to re-label already labeled image: "
                  f"{current_image}")
            # Move to next without saving
            self.current_index = (self.current_index + 1) % len(self.images)
            next_image = self.get_current_image()
            total_labeled = len(self.labels)
            progress = f"Image {self.current_index + 1}/{len(self.images)} " \
                       f"({total_labeled} labeled)"
            return next_image, progress, total_labeled
        else: # current_image is None (shouldn't happen if get_current_image works)
            progress = f"Finished labeling. ({len(self.labels)} labeled)"
            return None, progress, len(self.labels)



def create_app(images_folder="./images", output_file="labels.json"):
    """Create and launch the Gradio interface for image labeling."""
    labeler = ImageLabeler(images_folder, output_file)
    
    if not labeler.images:
        print(f"No images found in {images_folder}")
        return
    
    # def label_callback(score, event=None):
    #     """Callback for labeling images."""
    #     return labeler.label_image(score)
    
        # --- State variables for timing and session stats ---
    # Using gr.State to maintain values between interactions
    initial_image = labeler.get_current_image()
    initial_progress = f"Image {labeler.current_index + 1}/{len(labeler.images)} " \
                       f"({len(labeler.labels)} labeled)"

    def format_time(seconds):
        """Helper to format seconds into H:M:S"""
        if seconds < 0: seconds = 0
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    # --- Updated label_callback function ---
    def label_callback(score, current_start_time, session_start_time,
                       total_session_time, count_session):
        """
        Callback for labeling images. Calculates time, updates stats,
        and returns new values for UI components and state.
        """
        # 1. Calculate time spent on the *current* image
        end_time = time.time()
        time_spent_on_image = end_time - current_start_time

        # 2. Call the labeler's core logic
        next_image, base_progress, total_labeled_count = labeler.label_image(score)

        # 3. Update session stats
        new_count_session = count_session + 1
        new_total_session_time = total_session_time + time_spent_on_image
        avg_time_session = (new_total_session_time / new_count_session
                            if new_count_session > 0 else 0)
        total_elapsed_session = time.time() - session_start_time

        # 4. Prepare display strings
        timer_str = f"Last: {time_spent_on_image:.2f}s"
        avg_time_str = f"Avg: {avg_time_session:.2f}s"
        session_stats_str = (
            f"Session Stats: Labeled: {new_count_session} | "
            f"Avg Time: {avg_time_session:.2f}s | "
            f"Total Time: {format_time(total_elapsed_session)}"
        )

        # 5. Get the start time for the *next* image
        next_image_start_time = time.time()

        # 6. Handle completion
        if next_image is None:
            base_progress = f"All {len(labeler.images)} images labeled!"
            timer_str = "" # No next image timer
            # Keep avg time display consistent

        return (
            next_image,               # Output for image_display
            base_progress,            # Output for progress_text
            timer_str,                # Output for timer_text
            avg_time_str,             # Output for avg_time_text
            session_stats_str,        # Output for session_stats_display
            next_image_start_time,    # Output for image_start_time state
            new_total_session_time,   # Output for total_labeling_time state
            new_count_session         # Output for images_labeled_session state
        )


    with gr.Blocks(title="Aesthetic Labeler") as app: # Added title
        gr.Markdown("# Anime Image Aesthetic Labeler")

        # --- State Initialization ---
        # Stores the timestamp when the current image was displayed
        image_start_time = gr.State(value=time.time())
        # Stores the total time spent actively labeling in this session
        total_labeling_time = gr.State(0.0)
        # Stores the number of images labeled in this session
        images_labeled_session = gr.State(0)
        # Stores the timestamp when the app session started
        session_start_time = gr.State(value=time.time())

        # --- UI Layout ---
        with gr.Row():
            image_display = gr.Image(
                label="Current Image",
                value=initial_image,
                show_download_button=False,
                height=512,
                # Optimization: Set interactive=False as it's display only
                interactive=False
            )

        # Row for Progress and Timing Info
        with gr.Row():
            progress_text = gr.Textbox(
                label="Progress",
                value=initial_progress,
                interactive=False,
                scale=3 # Give progress more space
            )
            timer_text = gr.Textbox(
                label="Image Time",
                value="Last: 0.00s", # Initial value
                interactive=False,
                scale=1 # Smaller space for timer
            )
            avg_time_text = gr.Textbox(
                label="Session Avg",
                value="Avg: 0.00s", # Initial value
                interactive=False,
                scale=1 # Smaller space for average
            )

        # Row for Labeling Buttons
        with gr.Row():
            # Using numeric values directly for clarity with keyboard shortcuts
            btn_worst = gr.Button("Worst (1)", variant="stop", scale=1)
            btn_worse = gr.Button("Worse (2)", variant="secondary", scale=1)
            btn_better = gr.Button("Better (3)", variant="secondary", scale=1)
            btn_best = gr.Button("Best (4)", variant="success", scale=1)

        # Display for Session Statistics at the bottom
        session_stats_display = gr.Textbox(
            label="Session Summary",
            value="Session Stats: Labeled: 0 | Avg Time: 0.00s | Total Time: 00:00:00",
            interactive=False
        )

        # --- Button Click Events ---
        # Define outputs including the state variables to be updated
        outputs = [
            image_display,
            progress_text,
            timer_text,
            avg_time_text,
            session_stats_display,
            image_start_time, # Pass back the new start time
            total_labeling_time, # Pass back updated total time
            images_labeled_session # Pass back updated count
        ]
        # Define inputs including the current state values needed for calculation
        inputs = [
            image_start_time,
            session_start_time,
            total_labeling_time,
            images_labeled_session
        ]

        btn_worst.click(lambda *state: label_callback(0, *state),
                        inputs=inputs, outputs=outputs)
        btn_worse.click(lambda *state: label_callback(1, *state),
                        inputs=inputs, outputs=outputs)
        btn_better.click(lambda *state: label_callback(2, *state),
                         inputs=inputs, outputs=outputs)
        btn_best.click(lambda *state: label_callback(3, *state),
                       inputs=inputs, outputs=outputs)

        # --- Keyboard Shortcuts ---
        # Kept the existing robust JS implementation
        app.load(None, js="""
            function label_keydown(e) {
                // Only trigger if no input elements are focused
                if (document.activeElement.tagName === 'INPUT' ||
                    document.activeElement.tagName === 'TEXTAREA') {
                    return;
                }
                // Prevent default actions for number keys 1-4
                if (['1', '2', '3', '4'].includes(e.key)) {
                    e.preventDefault();
                    // Find buttons more reliably using data-testid or class
                    const buttons = document.querySelectorAll(
                        'button.gradio-button'
                    );
                    // Assuming the order is Worst, Worse, Better, Best
                    const buttonMap = {'1': 0, '2': 1, '3': 2, '4': 3};
                    if (e.key in buttonMap && buttons.length > buttonMap[e.key]) {
                        buttons[buttonMap[e.key]].click();
                    }
                }
            }
            // Add event listener
            document.addEventListener('keydown', label_keydown);
            // Cleanup listener when Gradio block is removed (optional but good practice)
            return () => {
                document.removeEventListener('keydown', label_keydown);
            }
        """)

    return app


def migrate_json_to_csv(json_path, csv_path):
    """Migrates labels from a JSON file to a CSV file."""
    if not os.path.exists(json_path):
        print(f"JSON file not found at {json_path}. No migration needed.")
        return False

    print(f"Migrating labels from {json_path} to {csv_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as fj:
            try:
                labels_dict = json.load(fj)
            except json.JSONDecodeError as e:
                print(f"Error reading JSON file: {e}. Migration aborted.")
                return False

        with open(csv_path, 'w', newline='', encoding='utf-8') as fc:
            writer = csv.writer(fc)
            writer.writerow(['image_path', 'label']) # Write header
            count = 0
            for image_path, label in labels_dict.items():
                writer.writerow([image_path, label])
                count += 1
        print(f"Successfully migrated {count} labels to {csv_path}.")
        return True
    except IOError as e:
        print(f"An error occurred during migration: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during migration: {e}")
        return False

# ADD function to export CSV to JSON
def export_csv_to_json(csv_path, json_path):
    """Exports labels from a CSV file to a JSON file."""
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Cannot export.")
        return False

    print(f"Exporting labels from {csv_path} to {json_path}...")
    labels_dict = {}
    try:
        with open(csv_path, 'r', newline='', encoding='utf-8') as fc:
            reader = csv.reader(fc)
            header = next(reader) # Skip header
            if header != ['image_path', 'label']:
                 print(f"Warning: Unexpected CSV header: {header} during export.")
                 # Continue assuming correct column order

            count = 0
            for row in reader:
                if len(row) == 2:
                    image_path, label_str = row
                    try:
                        labels_dict[image_path] = int(label_str)
                        count += 1
                    except ValueError:
                         print(f"Warning: Invalid label '{label_str}' for "
                               f"{image_path} found during export. Skipping.")
                else:
                    print(f"Warning: Skipping malformed row during export: {row}")

        with open(json_path, 'w', encoding='utf-8') as fj:
            json.dump(labels_dict, fj, indent=2)

        print(f"Successfully exported {count} labels to {json_path}.")
        return True
    except FileNotFoundError:
        # This case is handled by the initial check, but included for robustness
        print(f"CSV file not found at {csv_path}. Cannot export.")
        return False
    except StopIteration: # Handles empty file or only header
        print("CSV file is empty or contains only header. Exporting empty JSON.")
        with open(json_path, 'w', encoding='utf-8') as fj:
            json.dump({}, fj, indent=2)
        return True
    except IOError as e:
        print(f"An error occurred during export: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during export: {e}")
        return False


if __name__ == "__main__":
    # Configure these parameters according to your setup
    IMAGES_FOLDER = "./data/data-0000-cleaned"  # Path to your images folder
    # Define both potential input/output filenames
    OLD_JSON_FILE = None #"aesthetic_labels_train.json"
    OUTPUT_CSV_FILE = "aesthetic_labels_data_0000.csv"
    
    # --- Migration Step ---
    # Check if the old JSON exists and the new CSV doesn't, then migrate.
    if OLD_JSON_FILE:
        if os.path.exists(OLD_JSON_FILE) and not os.path.exists(OUTPUT_CSV_FILE):
            migrated = migrate_json_to_csv(OLD_JSON_FILE, OUTPUT_CSV_FILE)
            if migrated:
                print(f"Optional: You may want to rename or delete the old "
                    f"JSON file: {OLD_JSON_FILE}")
            else:
                print(f"Migration failed. Please check the files and errors.")
                exit(1)

    app = create_app(IMAGES_FOLDER, OUTPUT_CSV_FILE)
    if app: # Check if app creation was successful (images found)
        print("\n--- Starting Gradio App ---")
        print(f"Images Folder: {IMAGES_FOLDER}")
        print(f"Labels CSV File: {OUTPUT_CSV_FILE}")
        print("Use keys 1 (Worst) to 4 (Best) for faster labeling.")
        print("Close the terminal or press Ctrl+C to stop the app.")
        
        try:
            app.launch(
                share=False,
                server_port=1234,
                inbrowser=True,
                prevent_thread_lock=True,
                debug=True,
            )
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt received. Shutting down the app...")
        # ADD finally block for guaranteed execution
        finally:
            print("\nApp closed.")
            # --- Export to JSON after app closes ---
            final_json_path = Path(OUTPUT_CSV_FILE).with_suffix('.final.json')
            
            # Check if the CSV file actually exists before attempting export
            if not os.path.exists(OUTPUT_CSV_FILE):
                 print(f"CSV file {OUTPUT_CSV_FILE} not found. "
                       "Skipping final JSON export.")
            else:
                # You might want to call this manually or based on a condition
                # after labeling is complete.
                final_json_path = Path(OUTPUT_CSV_FILE).with_suffix('.final.json')
                print(f"\nLabeling finished or app closed.")
                export_csv_to_json(OUTPUT_CSV_FILE, str(final_json_path))

    else:
        print("Could not create Gradio app. Check image folder and paths.")
