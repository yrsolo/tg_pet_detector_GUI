# ShadowGEN — описание проекта и архитектура

ShadowGEN — сервис обработки фотографий: вырезает объект из фона, помещает на светлый фон и добавляет лёгкую реалистичную тень.  
Проект состоит из веб-интерфейса, API очереди задач и ML-инференса на GPU.

## Цели архитектуры

1. **Надёжность**: очередь и статусы живут на VPS, домашний GPU может перезагружаться/падать без потери очереди.
2. **Масштабируемость**: UI не зависит от конкретного ML-хоста, можно менять воркеров/инференс без изменения UI.
3. **Хранение артефактов**: входы/выходы лежат в S3-совместимом Object Storage, в Redis только метаданные.
4. **Кеширование**: повторная обработка идентичного файла не требует повторной загрузки в Object Storage; результаты кешируются по углу.

---

## Компоненты

### 1) UI (Gradio) — VPS
- Показывает форму загрузки и “крутилку” угла тени.
- Не ходит напрямую в ML: общается только с JobAPI.
- Для каждого изменения угла создаёт новую задачу (job) и ждёт готовность результата.

### 2) JobAPI (FastAPI) — VPS
Единственная точка входа для UI.

Функции:
- `POST /v1/jobs`:
  - принимает `image` (multipart) и `params` (JSON строка)
  - вычисляет `sha256` по байтам файла
  - складывает input в Object Storage по ключу `cache/{sha}/input.<ext>` (если уже существует — не перезаливает)
  - создаёт запись `job:{job_id}` в Redis со статусом `queued`
  - кладёт `job_id` в Redis очередь `queue:jobs`
- `GET /v1/jobs/{job_id}`:
  - читает job из Redis
  - когда `status=done`, генерирует presigned URL для каждого output и отдаёт их в поле `images`

Контракт `GET /v1/jobs/{job_id}` приближен к контракту ML-ответа:
- `api_version`, `request_id`, `message`, `images[]`, `meta`, `warnings`
- `images[]` содержит `{mime, url}` (а не base64)

Безопасность:
- API может быть защищён `X-API-Key` (конфиг через env).

### 3) Worker — VPS
Однопоточный воркер, который исполняет задачи из очереди.

Функции:
- Забирает `job_id` из `queue:jobs` через `BRPOPLPUSH` в `queue:processing` (защита от потери задач).
- Скачивает input из S3 по `input_key`.
- Вызывает ML-инференс (пока через HTTP) по адресу `http://127.0.0.1:9001/v1/process`.
  - На VPS этот порт доступен благодаря reverse SSH-туннелю от домашней GPU-машины.
- Сохраняет outputs в Object Storage:
  - `cache/{sha}/meta.json` — описание структуры outputs (количество, mime/ext), **зависит только от картинки**
  - `cache/{sha}/rot_{+/-NNN}/output_1.png` … — результаты для конкретного угла
  - `cache/{sha}/rot_{+/-NNN}/done.json` — маркер готовности результата для конкретного угла
- Обновляет Redis `job:{job_id}`:
  - `status=done`
  - `output_keys_json=[...]`
  - timestamps (`started_at`, `finished_at`, `updated_at`)
- При ошибке:
  - ставит `status=error`, пишет `error_message`
  - возвращает job обратно в `queue:jobs`

### 4) ML Server (Flask) — Home GPU
ML-инференс работает на домашнем ПК с GPU:
- endpoint: `POST /v1/process` (multipart image + form params)
- ответ: JSON с `images[]` в base64 + `meta/warnings`

ML Server не доступен в интернет напрямую. Доступ на VPS обеспечивается reverse SSH-туннелем:
- на VPS локально открыт `127.0.0.1:9001`, который проброшен на домашний `127.0.0.1:9001`.

### 5) Redis — VPS
- Хранит очередь и статусы:
  - `queue:jobs` — очередь задач
  - `queue:processing` — задачи “в работе” (для надёжности)
  - `job:{id}` — hash с метаданными задачи
- Redis наружу не открывается.

### 6) Object Storage (Yandex S3)
- Хранит все входы и выходы.
- Ключи:
  - `cache/{sha}/input.<ext>` — input, dedup по sha256 файла
  - `cache/{sha}/meta.json` — структура outputs (не зависит от угла)
  - `cache/{sha}/rot_{tag}/output_{i}.{ext}` — outputs для угла
  - `cache/{sha}/rot_{tag}/done.json` — маркер готовности output для угла

---

## Поток данных

### A) Обработка в UI (включая “крутилку”)
1) UI отправляет `POST /v1/jobs` (image + params)
2) JobAPI сохраняет input в S3 (dedup) и ставит job в Redis очередь
3) Worker берёт job из Redis, вызывает ML, сохраняет outputs в S3, отмечает `done`
4) UI polling’ом вызывает `GET /v1/jobs/{id}`:
   - пока `queued/running` — ждёт
   - когда `done` — получает `images[].url`, скачивает и показывает в Gallery

### B) Кеширование по углу
- Если `cache/{sha}/rot_{tag}/done.json` уже существует, воркер **не вызывает ML**:
  - берёт структуру outputs из `cache/{sha}/meta.json`
  - формирует список output_keys и сразу ставит job `done`

---

## Переменные окружения

### JobAPI (VPS)
- `REDIS_URL` — строка подключения к Redis
- `S3_ENDPOINT` — `https://storage.yandexcloud.net`
- `S3_BUCKET` — имя бакета
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` — ключи S3
- `API_KEY` — (опционально) проверка заголовка `X-API-Key`

### Worker (VPS)
- `REDIS_HOST`, `REDIS_PORT`
- `QUEUE_NAME`, `PROCESSING_NAME`
- `ML_URL` — `http://127.0.0.1:9001`
- `S3_ENDPOINT_URL`, `S3_BUCKET`, `S3_REGION`
- `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY`

### UI (VPS)
- `JOB_API_URL` — базовый URL JobAPI через nginx, например `https://<domain>/api`
- `API_KEY` — если включён `X-API-Key`

---

## Запуск и сервисы (кратко)

На VPS обычно работают systemd-сервисы:
- UI (`shadowgen`)
- JobAPI (`shadowgen-api`)
- Worker (`shadowgen-worker`)

ML Server работает на домашней GPU машине как отдельный процесс/сервис.

---

## Что считается “истиной”

- **Файлы**: только Object Storage (S3).  
- **Статусы/очередь**: Redis на VPS.  
- **UI** — только клиент JobAPI, не содержит состояния задач.

---

## Следующие шаги (высокоуровнево)

1. Логирование всех компонентов в едином стиле (UI/JobAPI/Worker/ML).
2. Health-check: мониторинг доступности ML через туннель, мониторинг очереди.
3. Ограничения по нагрузке: rate-limit для UI, лимиты на размер файлов.
4. (Позже) Объединение Worker+ML на одной машине (прямой импорт вместо HTTP), если потребуется.