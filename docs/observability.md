# Observability: Promtail -> Loki -> Grafana

## What is configured

- `ops/observability/docker-compose.yml`:
  - Loki (log storage)
  - Promtail (journal collector)
  - Grafana (UI)
- `ops/promtail/promtail.yml`:
  - Reads `journald`
  - Keeps only `shadowgen*.service` units
  - Extracts JSON fields: `service`, `env`, `level`, `request_id`, `job_id`
- `ops/grafana/provisioning/*`:
  - Auto-provisions Loki datasource
  - Auto-loads dashboard `ShadowGEN Logs`

## Why journald

Current app deployment is `systemd`, so journald already has all WEB/API/Worker logs.
Promtail scrapes journal directly; no file shippers needed.

## Labels and cardinality

Labels used in Loki:
- `unit` (systemd unit)
- `service`
- `env`
- `level`
- `host`

High-cardinality fields (`request_id`, `job_id`) are parsed from JSON but **not** promoted to labels.
Use them in LogQL with `| json` filtering.

## Useful LogQL

All app services:

```logql
{unit=~"shadowgen(-api|-worker)?\\.service"} | json
```

Errors:

```logql
{unit=~"shadowgen(-api|-worker)?\\.service"} | json | level="error"
```

By request id:

```logql
{unit=~"shadowgen(-api|-worker)?\\.service"} | json | request_id="YOUR_REQUEST_ID"
```

By job id:

```logql
{unit=~"shadowgen(-api|-worker)?\\.service"} | json | job_id="YOUR_JOB_ID"
```
