#!/usr/bin/env bash
set -euo pipefail

# Install and run Loki+Promtail+Grafana with Docker Compose.
# Run as root or via sudo.

REPO_DIR="${REPO_DIR:-/opt/shadowgen}"
STACK_DIR="$REPO_DIR/ops/observability"

if ! command -v docker >/dev/null 2>&1; then
  apt-get update
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  . /etc/os-release
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $VERSION_CODENAME stable" > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

systemctl enable docker
systemctl start docker

cd "$STACK_DIR"
docker compose pull
docker compose up -d

echo "Observability stack started."
echo "Grafana: http://127.0.0.1:3000 (admin/admin)"
echo "Loki:    http://127.0.0.1:3100/ready"
