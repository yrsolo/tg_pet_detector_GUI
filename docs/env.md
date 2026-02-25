# docs/env.md

## Переменные окружения и секреты

Проект использует `.env` (через `python-dotenv`) для конфигурации.  
**Файл `.env` содержит секреты и не должен попадать в git.**  
В репозитории держим только шаблон: `.env.example`.

---

## Правило хранения секретов

- ✅ `.env` — только локально/на сервере (вне git)
- ✅ `.env.example` — в git, без секретов
- ✅ systemd EnvironmentFile=/opt/shadowgen/.env — для сервисов на VPS
- ❌ токены, пароли, ключи доступа — не коммитить никогда

---

## Группы переменных

### 1) UI (Gradio)

Используется для связи UI → JobAPI.

- `JOB_API_URL` — базовый URL JobAPI через nginx  
  Пример: `https://style-app.solofarm.ru/api`
- `API_KEY` — если в JobAPI включена защита `X-API-Key` (один общий ключ)

Опционально:
- любые UI-настройки (порты, режимы) — по мере необходимости

---

### 2) JobAPI (FastAPI)

- `REDIS_URL` — строка подключения Redis  
  Пример: `redis://127.0.0.1:6379/0`

#### S3 (Yandex Object Storage)
- `S3_ENDPOINT` — endpoint S3  
  Обычно: `https://storage.yandexcloud.net`
- `S3_BUCKET` — имя бакета  
  Пример: `shadowgen`

Ключи доступа (секреты):
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

#### Безопасность
- `API_KEY` — если задан, JobAPI требует `X-API-Key: <API_KEY>` для всех запросов.

---

### 3) Worker

#### Redis
- `REDIS_HOST` (по умолчанию `127.0.0.1`)
- `REDIS_PORT` (по умолчанию `6379`)
- `QUEUE_NAME` (по умолчанию `queue:jobs`)
- `PROCESSING_NAME` (по умолчанию `queue:processing`)

#### ML
- `ML_URL` — адрес ML сервера, доступный воркеру  
  На VPS при reverse tunnel обычно: `http://127.0.0.1:9001`
- `ML_TIMEOUT_SEC` — таймаут инференса

#### S3
- `S3_ENDPOINT_URL` — endpoint, обычно `https://storage.yandexcloud.net`
- `S3_BUCKET`
- `S3_REGION` — обычно `ru-central1`
- `S3_ACCESS_KEY_ID`
- `S3_SECRET_ACCESS_KEY`

---

### 4) ML Server (Home GPU)

- `HOST`/`PORT` — если конфигурируется запуск
- модельные пути/кэши — по мере необходимости

Важно: ML Server в интернет не выставляется, доступен на VPS через reverse SSH tunnel.

---

## Пример файла `.env.example`

```dotenv
# ----------------
# Common / security
# ----------------
API_KEY=change-me-please

# ----------------
# JobAPI
# ----------------
REDIS_URL=redis://127.0.0.1:6379/0

S3_ENDPOINT=https://storage.yandexcloud.net
S3_BUCKET=shadowgen
AWS_ACCESS_KEY_ID=YOUR_KEY
AWS_SECRET_ACCESS_KEY=YOUR_SECRET

# ----------------
# Worker
# ----------------
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
QUEUE_NAME=queue:jobs
PROCESSING_NAME=queue:processing

ML_URL=http://127.0.0.1:9001
ML_TIMEOUT_SEC=300

S3_ENDPOINT_URL=https://storage.yandexcloud.net
S3_REGION=ru-central1
S3_ACCESS_KEY_ID=YOUR_KEY
S3_SECRET_ACCESS_KEY=YOUR_SECRET

# ----------------
# UI
# ----------------
JOB_API_URL=https://style-app.solofarm.ru/api