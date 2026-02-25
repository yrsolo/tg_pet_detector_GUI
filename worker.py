"""
ShadowGEN worker (single-thread, Redis queue, S3 storage, ML via HTTP)

What it does:
- Pops job_id from Redis queue with BRPOPLPUSH (safe: queue -> processing)
- Reads job:{id} in Redis: input_key, input_sha256, params_json
- Uses S3 as the source of truth
- Outputs are stored next to cached inputs:
    cache/{sha}/rot_{+/-NNN}/output_{i}.{ext}
  plus a per-rot marker:
    cache/{sha}/rot_{+/-NNN}/done.json
- Output count/mime/ext depends ONLY on the image (sha), not on rot.
  So we store global meta once per sha:
    cache/{sha}/meta.json
  and re-use it to construct output keys for any rot once done.json exists.

If cache hit (done.json exists), we skip ML completely and just write output_keys_json to Redis.
"""

import base64
import json
import os
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import boto3
import redis
import requests
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()
# -------------------- config --------------------
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
QUEUE = os.getenv("QUEUE_NAME", "queue:jobs")
PROCESSING = os.getenv("PROCESSING_NAME", "queue:processing")

ML_URL = os.getenv("ML_URL", "http://127.0.0.1:9001")
ML_TIMEOUT_SEC = int(os.getenv("ML_TIMEOUT_SEC", "300"))

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://storage.yandexcloud.net")
S3_BUCKET = os.getenv("S3_BUCKET", "shadowgen")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")
S3_REGION = os.getenv("S3_REGION", "ru-central1")


META_NAME = "meta.json"
DONE_NAME = "done.json"


# -------------------- utils --------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_redis_key(job_id: str) -> str:
    return f"job:{job_id}"


def rot_tag_from_params(params: dict[str, Any]) -> str:
    # stable formatting: rot_+020 rot_-005 rot_+000
    try:
        rot = int(params.get("rot", 0))
    except Exception:
        rot = 0
    return f"rot_{rot:+04d}"


def extract_sha_from_input_key(input_key: str) -> Optional[str]:
    # expects: cache/<sha>/input.*
    parts = (input_key or "").split("/")
    if len(parts) >= 2 and parts[0] == "cache":
        return parts[1]
    return None


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def s3_get_bytes(s3, key: str) -> bytes:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


def s3_put_bytes(s3, key: str, data: bytes, content_type: str = "application/octet-stream"):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)


def s3_exists(s3, key: str) -> bool:
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def set_status(r: redis.Redis, redis_key: str, status: str, **extra: str) -> None:
    mapping = {"status": status, "updated_at": now_iso(), **extra}
    r.hset(redis_key, mapping=mapping)


# -------------------- job model --------------------
@dataclass
class JobData:
    job_id: str
    redis_key: str
    input_key: str
    sha: str
    params: dict[str, Any]


def load_job(r: redis.Redis, job_id: str) -> JobData:
    key = job_redis_key(job_id)

    input_key = r.hget(key, "input_key")
    if not input_key:
        raise RuntimeError("job has no input_key in redis")

    params_json = r.hget(key, "params_json") or "{}"
    try:
        params = json.loads(params_json)
        if not isinstance(params, dict):
            params = {}
    except Exception:
        params = {}

    sha = r.hget(key, "input_sha256") or extract_sha_from_input_key(input_key) or job_id

    return JobData(job_id=job_id, redis_key=key, input_key=input_key, sha=sha, params=params)


# -------------------- S3 layout helpers --------------------
def meta_key(sha: str) -> str:
    return f"cache/{sha}/{META_NAME}"


def rot_prefix(sha: str, rot_tag: str) -> str:
    return f"cache/{sha}/{rot_tag}"


def done_key(prefix: str) -> str:
    return f"{prefix}/{DONE_NAME}"


def read_meta(s3, sha: str) -> Optional[dict[str, Any]]:
    key = meta_key(sha)
    if not s3_exists(s3, key):
        return None
    raw = s3_get_bytes(s3, key)
    try:
        data = json.loads(raw.decode("utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_meta(s3, sha: str, outputs_meta: list[dict[str, Any]]) -> None:
    # outputs_meta: [{"idx":1,"ext":"png","mime":"image/png"}, ...]
    body = json.dumps(
        {"outputs": outputs_meta, "created_at": now_iso()}, ensure_ascii=False
    ).encode("utf-8")
    s3_put_bytes(s3, meta_key(sha), body, content_type="application/json")


def write_done_marker(s3, prefix: str, outputs_count: int) -> None:
    body = json.dumps(
        {"ok": True, "outputs_count": outputs_count, "at": now_iso()}, ensure_ascii=False
    ).encode("utf-8")
    s3_put_bytes(s3, done_key(prefix), body, content_type="application/json")


def build_output_keys_from_meta(prefix: str, meta: dict[str, Any]) -> list[str]:
    outputs = meta.get("outputs")
    if not isinstance(outputs, list):
        return []
    keys: list[str] = []
    for item in outputs:
        if not isinstance(item, dict):
            continue
        idx = item.get("idx")
        ext = item.get("ext")
        if isinstance(idx, int) and isinstance(ext, str) and ext:
            keys.append(f"{prefix}/output_{idx}.{ext}")
    return keys


# -------------------- ML call & store --------------------
def call_ml_process(image_bytes: bytes, params: dict[str, Any]) -> dict[str, Any]:
    form = {k: (str(v).lower() if isinstance(v, bool) else str(v)) for k, v in params.items()}
    resp = requests.post(
        f"{ML_URL.rstrip('/')}/v1/process",
        files={"image": ("input.jpg", image_bytes, "image/jpeg")},
        data=form,
        timeout=ML_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    return resp.json()


def store_outputs_for_rot(
    s3, prefix: str, payload: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Returns:
      output_keys: ["cache/<sha>/<rot>/output_1.png", ...]
      outputs_meta: [{"idx":1,"ext":"png","mime":"image/png"}, ...]
    """
    output_keys: list[str] = []
    outputs_meta: list[dict[str, Any]] = []

    images = payload.get("images") or []
    if not isinstance(images, list):
        images = []

    for idx, img in enumerate(images, start=1):
        if not isinstance(img, dict) or "b64" not in img:
            continue

        b = base64.b64decode(img["b64"])
        mime = img.get("mime", "image/png")
        ext = "png" if "png" in mime else "jpg"

        key = f"{prefix}/output_{idx}.{ext}"
        s3_put_bytes(s3, key, b, content_type=mime)

        output_keys.append(key)
        outputs_meta.append({"idx": idx, "ext": ext, "mime": mime})

    # done marker for this rot (even if zero outputs, still mark done)
    write_done_marker(s3, prefix, outputs_count=len(output_keys))
    return output_keys, outputs_meta


# -------------------- core logic --------------------
def compute_or_restore_outputs(s3, job: JobData) -> list[str]:
    rot_tag = rot_tag_from_params(job.params)
    prefix = rot_prefix(job.sha, rot_tag)

    # Cache hit: per-rot done marker exists -> build keys from global meta and return
    if s3_exists(s3, done_key(prefix)):
        meta = read_meta(s3, job.sha)
        if meta:
            keys = build_output_keys_from_meta(prefix, meta)
            if keys:
                return keys
        # If meta missing/corrupt, fall back to compute (rare, but safe)

    # Cache miss (or meta missing) -> compute with ML
    image_bytes = s3_get_bytes(s3, job.input_key)
    payload = call_ml_process(image_bytes, job.params)

    output_keys, outputs_meta = store_outputs_for_rot(s3, prefix, payload)

    # Global meta should be stable per sha (count/ext/mime depend only on image)
    # If meta doesn't exist yet, or differs, overwrite (safe + simple).
    current_meta = read_meta(s3, job.sha)
    current_outputs = current_meta.get("outputs") if isinstance(current_meta, dict) else None
    if current_outputs != outputs_meta:
        write_meta(s3, job.sha, outputs_meta)

    return output_keys


# -------------------- main loop --------------------
def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    print("worker: connected to redis:", r.ping())
    s3 = s3_client()

    while True:
        job_id = r.brpoplpush(QUEUE, PROCESSING, timeout=5)
        if not job_id:
            continue

        ok = False
        redis_key = job_redis_key(job_id)

        try:
            job = load_job(r, job_id)
            print("worker: got job", job_id)

            set_status(
                r,
                redis_key,
                "running",
                started_at=now_iso(),
                worker=platform.node(),
            )

            output_keys = compute_or_restore_outputs(s3, job)

            set_status(
                r,
                redis_key,
                "done",
                finished_at=now_iso(),
                output_keys_json=json.dumps(output_keys, ensure_ascii=False),
                error_message="",
            )
            ok = True
            print("worker: done job", job_id, "->", output_keys)

        except Exception as e:
            print("worker: error processing job", job_id, ":", str(e))
            set_status(
                r,
                redis_key,
                "error",
                error_message=str(e),
            )

        finally:
            # remove from processing
            r.lrem(PROCESSING, 1, job_id)

            # if failed, push back to queue
            if not ok:
                r.rpush(QUEUE, job_id)


if __name__ == "__main__":
    main()
