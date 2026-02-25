#!/bin/bash
set -euo pipefail

cd /opt/shadowgen
git fetch --all --prune
git reset --hard origin/master
git pull --ff-only

# если зависимости веба могут меняться — можно раскомментировать:
/opt/shadowgen/.venv/bin/pip install -r requirements.web.txt

sudo systemctl restart shadowgen
sudo systemctl is-active shadowgen >/dev/null

sudo systemctl restart shadowgen-api
sudo systemctl is-active shadowgen-api >/dev/null

sudo systemctl restart shadowgen-worker
sudo systemctl is-active shadowgen-worker >/dev/null

echo "OK: deployed and restarted"