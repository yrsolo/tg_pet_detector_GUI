#!/bin/bash
set -euo pipefail

cd /opt/shadowgen
git fetch --all --prune
git pull --ff-only

# если зависимости веба могут меняться — можно раскомментировать:
# /opt/shadowgen/.venv/bin/pip install -r requirements.web.txt

sudo systemctl restart shadowgen
sudo systemctl is-active shadowgen >/dev/null
echo "OK: deployed and restarted"