# job_api.py

import io
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from PIL import Image

from log.logging_setup import bind_context, get_logger, log_timing, new_request_id

log = get_logger("UI.job_api")


@dataclass
class JobClient:
    base_url: str  # например "https://style-app.solofarm.ru/api"
    api_key: Optional[str] = None  # если включен X-API-Key
    poll_interval: float = 0.35  # для интерактива
    timeout_sec: float = 120.0  # на всякий

    def _headers(self, request_id: Optional[str] = None) -> dict[str, str]:
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        if request_id:
            h["X-Request-ID"] = request_id
        return h

    def create_job(
        self, image_pil: Image.Image, params: dict[str, Any], request_id: str = None
    ) -> str:
        # JPEG нормально для скорости; если надо без потерь — меняем на PNG
        buf = io.BytesIO()
        image_pil.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        resp = requests.post(
            f"{self.base_url.rstrip('/')}/v1/jobs",
            headers=self._headers(request_id=request_id),
            files={"image": ("image.jpg", buf.getvalue(), "image/jpeg")},
            data={"params": json.dumps(params)},  # JSON string
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["job_id"]

    def get_job(self, job_id: str, request_id: str = None) -> dict[str, Any]:
        resp = requests.get(
            f"{self.base_url.rstrip('/')}/v1/jobs/{job_id}",
            headers=self._headers(request_id=request_id),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_done(self, job_id: str, request_id: str = None) -> dict[str, Any]:
        t0 = time.time()

        while True:
            j = self.get_job(job_id, request_id=request_id)
            status = (j.get("job") or {}).get("status") or j.get("message")
            if status == "done":
                return j
            if status == "error":
                # если API кладёт error_message в job — покажем
                err = (j.get("job") or {}).get("error_message") or "job error"
                raise RuntimeError(err)
            if time.time() - t0 > self.timeout_sec:
                raise TimeoutError(f"job timeout: {job_id}")
            time.sleep(self.poll_interval)

    def fetch_images(self, job_payload: dict[str, Any]) -> list[Image.Image]:
        images = []
        for item in job_payload.get("images", []) or []:
            url = item.get("url")
            if not url:
                continue
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            images.append(Image.open(io.BytesIO(r.content)).convert("RGBA"))
        return images

    def submit_and_wait(
        self, image_pil: Image.Image, params: dict[str, Any], request_id: str = None
    ) -> tuple[list[Image.Image], str]:
        rid = request_id or new_request_id()
        bind_context(request_id=rid)
        try:
            with log_timing(log, "job_submit"):
                job_id = self.create_job(image_pil, params, request_id=rid)

            bind_context(job_id=job_id)
            log.info("job_created", job_id=job_id)

            with log_timing(log, "job_wait", job_id=job_id):
                payload = self.wait_done(job_id, request_id=rid)

            with log_timing(log, "job_fetch_images", job_id=job_id):
                images = self.fetch_images(payload)
            return images, job_id
        except Exception:
            log.error("job_api_failed", exc_info=True)
            raise

    def process(
        self, image_pil: Image.Image, params: dict[str, Any], request_id: str = None
    ) -> tuple[list[Image.Image], str]:
        return self.submit_and_wait(image_pil, params, request_id=request_id)
