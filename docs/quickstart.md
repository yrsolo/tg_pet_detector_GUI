# Quickstart (VPS + Home GPU)

Цель: запустить полный контур **UI → JobAPI → Redis → Worker (VPS) → ML (Home GPU) → S3**, и проверить его через curl и UI.

---

## 0) Предварительно

Нужно:
- VPS с nginx + python3 + redis
- Домашний ПК с GPU, где запускается `ML_server.py`
- Yandex Object Storage бакет (`shadowgen`) и ключи доступа

**Важно:** `.env` содержит секреты и не должен попадать в git.

---

## 1) Конфигурация `.env` на VPS

Создай файл:

- `/opt/shadowgen/.env`

Пример (заполни своими значениями):

```dotenv
# --- security ---
API_KEY=change-me

# --- redis ---
REDIS_URL=redis://127.0.0.1:6379/0

# --- yandex s3 (for JobAPI) ---
S3_ENDPOINT=https://storage.yandexcloud.net
S3_BUCKET=shadowgen
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# --- worker settings ---
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
QUEUE_NAME=queue:jobs
PROCESSING_NAME=queue:processing

# ML on VPS is доступен через reverse tunnel:
ML_URL=http://127.0.0.1:9001
ML_TIMEOUT_SEC=300

# --- yandex s3 (for Worker) ---
S3_ENDPOINT_URL=https://storage.yandexcloud.net
S3_REGION=ru-central1
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...

# --- UI ---
JOB_API_URL=https://<your-domain>/api


Права:

```bash
chmod 600 /opt/shadowgen/.env
```

---

## 2) Проверка Redis на VPS

```bash
redis-cli ping
redis-cli LLEN queue:jobs
redis-cli LLEN queue:processing
```

---

## 3) Запуск ML на домашнем GPU

На домашнем ПК:

```bash
python3 ML_server.py
```

Порт по умолчанию: `127.0.0.1:9001`.

---

## 4) Reverse SSH tunnel (Home GPU → VPS)

Туннель инициируется с домашнего ПК так, чтобы **на VPS** появился локальный порт `127.0.0.1:9001`, ведущий на домашний `127.0.0.1:9001`.

Пример команды (подставь пользователя и хост VPS):

```bash
ssh -N -R 9001:127.0.0.1:9001 user@<vps-host>
```

Проверка на VPS:

```bash
ss -lntp | grep 9001
```

---

## 5) Сервисы на VPS

Обычно работают systemd-сервисы:

* `shadowgen` (UI)
* `shadowgen-api` (JobAPI)
* `shadowgen-worker` (Worker)

Проверка:

```bash
sudo systemctl status shadowgen --no-pager
sudo systemctl status shadowgen-api --no-pager
sudo systemctl status shadowgen-worker --no-pager
```

Логи:

```bash
journalctl -u shadowgen -f
journalctl -u shadowgen-api -f
journalctl -u shadowgen-worker -f
```

---

## 6) Smoke-test JobAPI

Health:

```bash
curl -sS https://<your-domain>/api/health && echo
```

Создание job (на любой машине, где есть `image.jpg`):

```bash
curl -sS -X POST "https://<your-domain>/api/v1/jobs" \
  -H "X-API-Key: <API_KEY>" \
  -F "image=@image.jpg" \
  -F 'params={"rot":0,"max_objects":2,"return_debug":false}'
```

В ответе будет `job_id`.

---

## 7) Проверка очереди в Redis

На VPS:

```bash
redis-cli LLEN queue:jobs
redis-cli LRANGE queue:jobs 0 10
redis-cli LLEN queue:processing
redis-cli LRANGE queue:processing 0 10
```

---

## 8) Проверка статуса job

На VPS:

```bash
JOB_ID=<id_from_create_job>
redis-cli HGETALL "job:${JOB_ID}"
```

Статусы:

* `queued` → job в очереди
* `running` → воркер взял job
* `done` → outputs готовы
* `error` → ошибка (смотри `error_message`)

---

## 9) Получение результатов (presigned urls)

```bash
curl -sS "https://<your-domain>/api/v1/jobs/${JOB_ID}" \
  -H "X-API-Key: <API_KEY>"
```

В ответе:

* `images[]` содержит `{mime, url}`

Открой любую `url` в браузере — должна открыться картинка.

---

## 10) Проверка UI

Открой:

* `https://<your-domain>/`

Загрузи картинку → нажми обработку → попробуй `+20/-20`.
UI создаёт job, ждёт `done`, показывает миниатюры.

---

## 11) Типовые проблемы

### 11.1) Job залип в `queue:processing`
Сценарий: воркер забрал job (BRPOPLPUSH), но упал до завершения.

Проверка:
```bash
redis-cli LLEN queue:processing
redis-cli LRANGE queue:processing 0 10
````

Быстрый фикс (вернуть 1 job обратно):

```bash
redis-cli RPOPLPUSH queue:processing queue:jobs
```

### 11.2) Worker не видит ML (reverse tunnel не поднят)

Сценарий: воркер на VPS обращается к `ML_URL=http://127.0.0.1:9001`, но порт не слушается.

Проверка на VPS:

```bash
ss -lntp | grep 9001
```

Если пусто:

* ML server на домашней машине не запущен
* reverse SSH tunnel упал или не поднимался

Проверка, что tunnel поднят с home:

```bash
# на домашнем ПК должен висеть ssh -R 9001:127.0.0.1:9001 ...
ps aux | grep "ssh -N -R 9001"
```

### 11.3) JobAPI или Worker ходят в Amazon вместо Yandex S3

Сценарий: в ошибке фигурирует `amazonaws.com` (например `...s3.ru-central1.amazonaws.com...`).

Причина: `endpoint_url` не подхватился.

Проверка:

* JobAPI: `S3_ENDPOINT=https://storage.yandexcloud.net`
* Worker: `S3_ENDPOINT_URL=https://storage.yandexcloud.net`

Решение:

* убедиться, что `.env` действительно подхватывается сервисом (systemd)
* перезапустить сервисы

### 11.4) `POST /v1/jobs` работает, но результатов нет

Сценарий: job создаётся, в Redis есть запись, но статус не становится `done`.

Проверка:

```bash
JOB_ID=<id>
redis-cli HGETALL "job:${JOB_ID}"
```

* если `status=queued` — воркер не забирает очередь (воркер не запущен или не видит Redis)
* если `status=running` и не меняется — воркер завис на ML/S3
* если `status=error` — смотреть `error_message` и логи воркера

Логи воркера:

```bash
journalctl -u shadowgen-worker -f
```

### 11.5) `GET /v1/jobs/{id}` не отдаёт `images[].url`

Сценарий: статус `done`, но `images` пустой.

Причины:

* `output_keys_json` пустой или некорректный JSON
* воркер записал outputs не туда, где JobAPI ожидает
* в JobAPI не включена генерация presigned urls

Проверка:

```bash
JOB_ID=<id>
redis-cli HGET "job:${JOB_ID}" output_keys_json
```

### 11.6) 502/504 от nginx

Сценарий: сайт открывается с ошибкой, или API недоступно.

Проверка локальных сервисов на VPS:

```bash
sudo systemctl status shadowgen --no-pager
sudo systemctl status shadowgen-api --no-pager
sudo systemctl status shadowgen-worker --no-pager
```

Проверка локальных портов:

* UI (gradio): `127.0.0.1:7860`
* API (uvicorn): `127.0.0.1:8000`

```bash
ss -lntp | egrep "7860|8000"
```

### 11.7) Ошибка авторизации (`401 Unauthorized`)

Сценарий: JobAPI требует `X-API-Key`, но UI/curl не передают.

Проверка:

* в `.env` задан `API_KEY`
* запросы содержат заголовок `X-API-Key: <API_KEY>`
* UI использует `API_KEY` и передаёт его в JobClient

### 11.8) Повторная загрузка “того же файла” не дедуплицируется

Сценарий: ожидаешь, что input не перезаливается, но в S3 появляются новые копии.

Важно:

* dedup считается по **байтам файла** (`sha256(content)`).
* если файл пересохранён/пережат/добавлены метаданные — байты другие, sha другой → это нормально.

Проверка:

```bash
JOB_ID=<id>
redis-cli HGET "job:${JOB_ID}" input_sha256
redis-cli HGET "job:${JOB_ID}" input_key
```

### 11.9) Кэш по углу не срабатывает

Сценарий: повторный запрос с тем же `rot` всё равно вызывает ML.

Проверка в S3:

* должен существовать `cache/{sha}/rot_{tag}/done.json`
* должен существовать `cache/{sha}/meta.json`

Если `done.json` нет — значит воркер не пишет маркер, либо пишет в другой префикс.
Если `meta.json` нет — воркер не сохранил структуру outputs.

### 11.10) Секреты случайно попали в git

Сценарий: `.env` или токены закоммичены.

Действия:

1. Считать секреты скомпрометированными
2. Перевыпустить токены/ключи
3. Удалить из истории git (BFG/ filter-repo)
4. Добавить `.env` в `.gitignore` и использовать `.env.example`

```
```
