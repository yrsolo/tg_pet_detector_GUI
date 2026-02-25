import base64
import json
import os
import platform
from datetime import datetime, timezone

import boto3
import redis
import requests
from botocore.config import Config
from dotenv import load_dotenv

load_dotenv()
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
QUEUE = os.getenv("QUEUE_NAME", "queue:jobs")
PROCESSING = os.getenv("PROCESSING_NAME", "queue:processing")

S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT")  # например https://storage.yandexcloud.net
S3_BUCKET = os.getenv("S3_BUCKET", "shadowgen")
S3_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
S3_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")

ML_URL = os.getenv("ML_URL", "http://127.0.0.1:9001")


def s3_client():
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
    )


def s3_get_bytes(s3, key: str) -> bytes:
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return obj["Body"].read()


def s3_put_bytes(s3, key: str, data: bytes, content_type: str = "image/png"):
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=data, ContentType=content_type)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def main():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    print("worker: connected to redis:", r.ping())

    while True:
        job_id = r.brpoplpush(QUEUE, PROCESSING, timeout=5)
        if not job_id:
            continue

        job_key = f"job:{job_id}"
        ok = False
        try:
            print("worker: got job", job_id)

            r.hset(
                job_key,
                mapping={
                    "status": "running",
                    "started_at": now_iso(),
                    "updated_at": now_iso(),
                    "worker": platform.node(),
                },
            )

            s3 = s3_client()

            input_key = r.hget(job_key, "input_key")
            if not input_key:
                raise RuntimeError("job has no input_key in redis")

            # 1) скачать input из S3
            data = s3_get_bytes(s3, input_key)

            # 2) вызвать ML_server как раньше UI
            params_json = r.hget(job_key, "params_json") or "{}"
            params = json.loads(params_json)

            resp = requests.post(
                f"{ML_URL.rstrip('/')}/v1/process",
                files={"image": ("input.jpg", data, "image/jpeg")},
                data={
                    k: str(v).lower() if isinstance(v, bool) else str(v) for k, v in params.items()
                },
                timeout=300,
            )
            resp.raise_for_status()
            payload = resp.json()

            # 3) сохранить все картинки из payload["images"] в S3
            output_keys = []
            for idx, img in enumerate(payload.get("images", []), start=1):
                b = base64.b64decode(img["b64"])
                mime = img.get("mime", "image/png")
                ext = "png" if "png" in mime else "jpg"
                out_key = f"jobs/{job_id}/output_{idx}.{ext}"
                s3_put_bytes(s3, out_key, b, content_type=mime)
                output_keys.append(out_key)

            r.hset(
                job_key,
                mapping={
                    "status": "done",
                    "finished_at": now_iso(),
                    "updated_at": now_iso(),
                    "output_keys_json": json.dumps(output_keys),
                },
            )
            ok = True
            print("worker: done job", job_id, "->", output_keys)
        except Exception as e:
            print("worker: error processing job", job_id, ":", str(e))
            r.hset(
                job_key,
                mapping={
                    "status": "error",
                    "error_message": str(e),
                    "updated_at": now_iso(),
                },
            )
        finally:
            # снять из processing
            r.lrem(PROCESSING, 1, job_id)

            # если не ok — вернуть обратно в очередь
            if not ok:
                r.rpush(QUEUE, job_id)


if __name__ == "__main__":
    main()
