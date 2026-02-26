"""Web interface for the Shadow Generator application using Gradio.
Allows users to upload images, adjust shadow parameters, and view processed results.
Supports both local and production environments with optional SSL configuration."""

import os
import sys

import gradio as gr

from contracts.contracts import ShadowParams
from log.logging_setup import bind_context, get_logger, log_timing, new_request_id, setup_logging
from WEB.job_api import JobClient

setup_logging()
log = get_logger("UI")

SERVER_URL = "http://127.0.0.1:9001/"

CERTIFICATE_PATH = "/opt/shadowgen/https-cert/certificate.pem"
PRIVATE_KEY_PATH = "/opt/shadowgen/https-cert/private_key.pem"

ENV = os.getenv("ENV", "local")

JOB_API_URL = os.getenv("JOB_API_URL", "https://style-app.solofarm.ru/api")
JOB_API_KEY = os.getenv("API_KEY")

job_client = JobClient(
    base_url=JOB_API_URL,
    api_key=JOB_API_KEY,
    poll_interval=0.35,
    timeout_sec=180,
)

client = job_client


def process_image_server(image, rot, max_size=1024, request_id=None):
    """Send image and rotation parameter to the API and return processed images."""
    if image.size[0] > max_size or image.size[1] > max_size:
        image.thumbnail((max_size, max_size))

    params = ShadowParams(rot=rot, max_objects=4, return_debug=False).to_dict()
    return client.process(image, params, request_id=request_id)


def select_image(index, images):
    """Select the image at the specified index from the list."""
    index = int(index)
    return images[index][0]


with gr.Blocks() as app:
    gr.Markdown("# Shadow Generator")
    gr.Markdown("Загрузите изображение для обработки. \nКнопки +/- 20 вращают тень.")

    with gr.Row():
        image_input = gr.Image(type="pil", label="Загрузите изображение")

    with gr.Row():
        process_button = gr.Button("Обработать")
        angle_input = gr.Number(value=0, label="Угол тени (0-359)", visible=False)
        decrease_button = gr.Button("-20")
        increase_button = gr.Button("+20")

    with gr.Row():
        thumbnails = gr.Gallery(label="Миниатюры", columns=2, rows=2)

    reload_button = gr.Button("Перезагрузить приложение")

    def update_angle(current_angle, delta, image):
        """Update angle and reprocess image."""
        new_angle = (current_angle + delta) % 360
        angle_input.value = new_angle
        processed_images = process_image(image, new_angle)
        return new_angle, processed_images

    def process_image(image, rot):
        """Process image through the job API and return result images."""
        rid = new_request_id()
        bind_context(request_id=rid, job_id=None)
        log.info("ui_submit", has_image=image is not None, shadow_angle=rot)

        try:
            if image is None:
                log.warning("ui_no_image")
                return None

            with log_timing(log, "ui_process_total"):
                processed_images, job_id = process_image_server(image, rot, request_id=rid)

            bind_context(job_id=job_id)
            log.info("ui_done", job_id=job_id, images_count=len(processed_images))
            return processed_images
        except Exception:
            log.error("ui_failed", exc_info=True)
            return None

    def reload_app():
        """Reload the app by restarting the current Python process."""
        python = sys.executable
        os.execl(python, python, *sys.argv)

    decrease_button.click(
        fn=update_angle,
        inputs=[angle_input, gr.State(-20), image_input],
        outputs=[angle_input, thumbnails],
    )

    increase_button.click(
        fn=update_angle,
        inputs=[angle_input, gr.State(20), image_input],
        outputs=[angle_input, thumbnails],
    )

    process_button.click(
        fn=process_image,
        inputs=[image_input, angle_input],
        outputs=thumbnails,
    )

    reload_button.click(
        fn=reload_app,
        inputs=[],
        outputs=[],
    )


if __name__ == "__main__":
    if ENV == "production":
        log.info("ui_startup", mode="production", ssl=True)
        app.launch(
            server_name="127.0.0.1",
            server_port=7860,
        )
    else:
        log.info("ui_startup", mode="local", ssl=False)
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
        )
