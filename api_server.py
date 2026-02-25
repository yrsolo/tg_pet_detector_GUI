import hashlib
import json
import os
import time
import uuid
from typing import Any, Optional

import boto3
import redis
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile

# ---- config from env ----
load_dotenv()
REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
S3_ENDPOINT = os.getenv("S3_ENDPOINT", "https://storage.yandexcloud.net")
S3_BUCKET = os.getenv("S3_BUCKET")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

API_KEY = os.getenv("API_KEY")  # optional

if not S3_BUCKET:
    raise RuntimeError("S3_BUCKET is not set")

r = redis.Redis.from_url(REDIS_URL, decode_responses=True)

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    config=Config(signature_version="s3v4"),
)

app = FastAPI(title="ShadowGEN VPS Job API", version="1.0")

QUEUE_KEY = "queue:jobs"


def now_ms() -> int:
    return int(time.time() * 1000)


def require_api_key(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


def parse_params(params_raw: Optional[str]) -> dict[str, Any]:
    if not params_raw:
        return {}
    try:
        data = json.loads(params_raw)
        if not isinstance(data, dict):
            raise ValueError("params must be a JSON object")
        return data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad params JSON: {e}")


def job_key(job_id: str) -> str:
    return f"job:{job_id}"


def s3_put_bytes(key: str, data: bytes, content_type: str):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)


@app.get("/health")
def health():
    try:
        r.ping()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Redis error: {e}")
    return {"ok": True}


@app.post("/v1/jobs")
async def create_job(
    image: UploadFile = File(...),
    params: Optional[str] = Form(None),  # JSON string
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)

    job_id = str(uuid.uuid4())
    p = parse_params(params)

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty image")

    # --- dedup input by sha256(file bytes) ---
    sha = hashlib.sha256(content).hexdigest()

    # расширение — чисто для удобства (можно оставить .bin)
    _, ext = os.path.splitext(image.filename or "")
    ext = (ext or ".bin").lower()

    input_key = f"cache/{sha}/input{ext}"

    # проверить есть ли уже в S3
    exists = False
    try:
        s3.head_object(Bucket=S3_BUCKET, Key=input_key)
        exists = True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in ("404", "NoSuchKey", "NotFound"):
            raise

    # залить только если нет
    if not exists:
        s3_put_bytes(
            input_key,
            content,
            image.content_type or "application/octet-stream",
        )
    job = {
        "job_id": job_id,
        "status": "queued",
        "created_ms": str(now_ms()),
        "updated_ms": str(now_ms()),
        "input_sha256": sha,
        "input_key": input_key,
        "params_json": json.dumps(p, ensure_ascii=False),
        "output_keys_json": json.dumps([], ensure_ascii=False),
        "message": "",
        "error": "",
    }
    r.hset(job_key(job_id), mapping=job)
    r.rpush(QUEUE_KEY, job_id)

    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/jobs/{job_id}")
def get_job(
    job_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    require_api_key(x_api_key)
    key = job_key(job_id)
    if not r.exists(key):
        raise HTTPException(status_code=404, detail="Job not found")
    job = r.hgetall(key)

    images = []
    if job.get("status") == "done":
        try:
            output_keys = json.loads(job.get("output_keys_json") or "[]")
        except Exception:
            output_keys = []

        for obj_key in output_keys:
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_BUCKET, "Key": obj_key},
                ExpiresIn=3600,
            )
            images.append({"mime": "image/png", "url": url})

    return {
        "api_version": "1",
        "request_id": job_id,
        "message": job.get("status", "unknown"),
        "images": images,
        "meta": {"timings_ms": {"total": 0, "processing": 0}},
        "warnings": [],
        "job": {
            "id": job_id,
            "status": job.get("status"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "updated_at": job.get("updated_at"),
        },
    }
