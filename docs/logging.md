# Логирование (каркас)

Этот документ описывает стандарты логирования для компонентов ShadowGEN.

---

## 1) Цели логирования

- Наблюдаемость по всему потоку обработки.
- Быстрый поиск причин ошибок.
- Сопоставление событий между UI / API / Worker / ML.

---

## 2) Формат логов

- Формат: JSON (structured logs).
- Базовые поля:
  - `service`
  - `env`
  - `request_id`
  - `job_id`
  - `event`
  - `level`
  - `timestamp`

---

## 3) Обязательные события по компонентам

### UI
- `ui_request_received`
- `request_call_start`
- `request_call_done`
- `ui_process_failed`

### JobAPI
- `job_created`
- `job_status_changed`
- `job_result_ready`
- `job_api_error`

### Worker
- `worker_job_picked`
- `worker_ml_call_start`
- `worker_ml_call_done`
- `worker_job_done`
- `worker_job_error`

### ML
- `ml_request_received`
- `ml_processing_start`
- `ml_processing_done`
- `ml_processing_failed`

---

## 4) Корреляция запросов

- `request_id` создаётся на входе в UI/API и передаётся по всей цепочке.
- `job_id` назначается в JobAPI и используется в Worker.
- При ошибках оба идентификатора должны попадать в лог.

---

## 5) Уровни логов

- `INFO`: штатные этапы обработки.
- `WARNING`: восстановимые/ожидаемые отклонения.
- `ERROR`: ошибки, влияющие на результат для пользователя.

---

## 6) Антипаттерны

- Не использовать `print` в production-потоке.
- Не логировать секреты (`API_KEY`, токены, ключи доступа).
- Не логировать полные бинарные payload изображения.

---

## 7) TODO для заполнения документа

- [ ] Добавить реальные примеры лог-сообщений из каждого сервиса.
- [ ] Зафиксировать naming-схему событий.
- [ ] Описать интеграцию с сборщиком логов (если используется).
- [ ] Добавить рекомендации по алертам.
