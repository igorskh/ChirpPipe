from chirps.ml.lar_iqa_assess import LarIqaAssess
import gradio as gr
import os

images_folder = "demo"


def get_images_from_folder(folder_path):
    if not os.path.exists(folder_path):
        return []
    supported_exts = {"jpeg", "jpg", "png", "bmp", "gif"}
    images = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
        and os.path.splitext(f)[1].lower().lstrip(".") in supported_exts
    ]
    return sorted(images)


def load_images(images_folder):
    images = get_images_from_folder(images_folder)
    if not images:
        return [], [], "No images found in the folder."

    return images, [], "Images loaded."


def assess_images(images_folder):
    images = get_images_from_folder(images_folder)
    if not images:
        return [], [], "No images found in the folder."

    assessor = LarIqaAssess()
    results = assessor.process(input=images_folder)

    sorted_results = sorted(
        results["results"], key=lambda x: x["score"], reverse=True)

    gallery_items = []
    scores_data = []

    for r in sorted_results:
        image_path = r["image_path"]
        score = r["score"]
        gallery_items.append((image_path, f"Score: {score:.4f}"))
        scores_data.append([os.path.basename(image_path), f"{score:.4f}"])

    return gallery_items, scores_data, "Assessment complete."


def create_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# Image Quality Assessment")

        with gr.Row():
            with gr.Column():
                folder_input = gr.Textbox(
                    label="Images Folder",
                    value=images_folder,
                    placeholder="Enter folder path"
                )
                assess_btn = gr.Button("Assess Images", variant="primary")

            with gr.Column():
                status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            with gr.Column():
                images_output = gr.Gallery(
                    label="Images Grid",
                    columns=4,
                    rows=2,
                    object_fit="contain"
                )

            with gr.Column():
                scores_output = gr.DataFrame(
                    label="IQA Scores",
                    headers=["Image", "Score"],
                    wrap=True
                )

        def update_with_scores(gallery_items):
            if not gallery_items:
                return None

            scores_data = []
            for item in gallery_items:
                image_path = None
                score = None

                if isinstance(item, (tuple, list)) and len(item) >= 1:
                    image_path = item[0]
                    if len(item) >= 2 and isinstance(item[1], str) and "Score:" in item[1]:
                        try:
                            score = float(item[1].split(
                                "Score:", 1)[1].strip())
                        except (TypeError, ValueError):
                            score = None
                elif isinstance(item, dict):
                    image_path = item.get("path")
                    if "score" in item:
                        score = item.get("score")
                else:
                    image_path = item

                if score is None:
                    score = 0.0

                scores_data.append(
                    [os.path.basename(image_path), f"{score:.4f}"])

            return scores_data

        demo.load(
            fn=load_images,
            inputs=[folder_input],
            outputs=[images_output, scores_output, status]
        )

        assess_btn.click(
            fn=assess_images,
            inputs=[folder_input],
            outputs=[images_output, scores_output, status]
        )

        images_output.change(
            fn=update_with_scores,
            inputs=[images_output],
            outputs=[scores_output]
        )

    return demo


if __name__ == "__main__":
    demo = create_ui()
    demo.launch()
