#!/bin/bash

# Путь к ключу сервисного аккаунта
SERVICE_ACCOUNT_KEY_PATH=/opt/shadowgen/https-cert/cert-updater-key.json

# Папка, где хранятся сертификаты
CERT_DIR=/opt/shadowgen/https-cert/

# ID сертификата в Yandex Cloud
CERTIFICATE_ID=fpqca9ot3om9d5a68rj1

# Авторизация с использованием сервисного аккаунта
yc --profile cert-updater config list
yc --profile cert-updater iam create-token

# Обновление сертификатов
echo "Обновляю сертификат..."

yc --profile cert-updater certificate-manager certificate content \
  --id $CERTIFICATE_ID \
  --chain $CERT_DIR/certificate.pem \
  --key $CERT_DIR/private_key.pem \
  --key-format pkcs8

# Рестарт приложения (если требуется)
echo "Перезапускаю приложение..."

sudo systemctl restart shadowgen

echo "Обновление сертификата завершено."