#!/bin/bash

# Путь к ключу сервисного аккаунта
SERVICE_ACCOUNT_KEY_PATH=~/tg-det/https-cert/cert-updater-key.json

# Папка, где хранятся сертификаты
CERT_DIR=~/tg-det/https-cert

# ID сертификата в Yandex Cloud
CERTIFICATE_ID=fpqca9ot3om9d5a68rj1

# Авторизация с использованием сервисного аккаунта
yc auth create-key --key-file $SERVICE_ACCOUNT_KEY_PATH

# Обновление сертификатов
echo "Обновляю сертификат..."
yc certificate-manager certificate content \
  --id $CERTIFICATE_ID \
  --chain $CERT_DIR/certificate.pem \
  --key $CERT_DIR/private_key.pem \
  --key-format pkcs8

# Рестарт приложения (если требуется)
echo "Перезапускаю приложение..."
pkill -f "app.py"
bash ~/run_app.sh

echo "Обновление сертификата завершено."
