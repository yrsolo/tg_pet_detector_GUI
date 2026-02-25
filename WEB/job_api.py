# job_api.py

import io
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
from PIL import Image


@dataclass
class JobClient:
    base_url: str  # например "https://style-app.solofarm.ru/api"
    api_key: Optional[str] = None  # если включен X-API-Key
    poll_interval: float = 0.35  # для интерактива
    timeout_sec: float = 120.0  # на всякий

    def _headers(self) -> dict[str, str]:
        h = {}
        if self.api_key:
            h["X-API-Key"] = self.api_key
        return h

    def create_job(self, image_pil: Image.Image, params: dict[str, Any]) -> str:
        # JPEG нормально для скорости; если надо без потерь — меняем на PNG
        buf = io.BytesIO()
        image_pil.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        resp = requests.post(
            f"{self.base_url.rstrip('/')}/v1/jobs",
            headers=self._headers(),
            files={"image": ("image.jpg", buf.getvalue(), "image/jpeg")},
            data={"params": requests.utils.json.dumps(params)},  # JSON string
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["job_id"]

    def get_job(self, job_id: str) -> dict[str, Any]:
        resp = requests.get(
            f"{self.base_url.rstrip('/')}/v1/jobs/{job_id}",
            headers=self._headers(),
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def wait_done(self, job_id: str) -> dict[str, Any]:
        t0 = time.time()
        while True:
            j = self.get_job(job_id)
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

    def submit_and_wait(self, image_pil: Image.Image, params: dict[str, Any]) -> list[Image.Image]:
        job_id = self.create_job(image_pil, params)
        payload = self.wait_done(job_id)
        return self.fetch_images(payload)
