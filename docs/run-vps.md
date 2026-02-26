# Run On VPS (No Docker For App)

This runbook keeps the app stack as `systemd` services and uses Docker only for observability:
- `shadowgen.service` (WEB)
- `shadowgen-api.service` (API)
- `shadowgen-worker.service` (Worker)
- `loki + promtail + grafana` (logs)

## 1. Install/update ShadowGEN services

Copy unit files from `ops/systemd/` into `/etc/systemd/system/`, then reload:

```bash
sudo cp /opt/shadowgen/ops/systemd/shadowgen*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable shadowgen shadowgen-api shadowgen-worker
sudo systemctl restart shadowgen shadowgen-api shadowgen-worker
```

Check status:

```bash
sudo systemctl status shadowgen shadowgen-api shadowgen-worker --no-pager
```

## 2. Install observability stack (Loki/Promtail/Grafana)

Run installer:

```bash
sudo bash /opt/shadowgen/ops/scripts/install_promtail.sh
```

This starts:
- Grafana on `127.0.0.1:3000`
- Loki on `127.0.0.1:3100`

Default Grafana login:
- user: `admin`
- password: `admin`

## 3. Publish Grafana via nginx (recommended)

Example nginx location:

```nginx
location /grafana/ {
    proxy_pass http://127.0.0.1:3000/;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

## 4. Verify logs in Grafana

Open dashboard: `ShadowGEN / ShadowGEN Logs` (auto-provisioned).

Quick checks in Explore:

```logql
{unit="shadowgen.service"} | json
```

```logql
{unit=~"shadowgen(-api|-worker)?\\.service"} | json | level="error"
```

```logql
{unit=~"shadowgen(-api|-worker)?\\.service"} | json | request_id="PUT_REQUEST_ID_HERE"
```

## 5. Local fallback without Grafana

```bash
journalctl -u shadowgen -f
journalctl -u shadowgen-api -f
journalctl -u shadowgen-worker -f
```
