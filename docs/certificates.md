# SSL Сертификаты для сервера

Лежат в папке ***https-cert***

Обновляются через ***/ops/scripts/update_certificates.sh***

### Добалвено регулятное обновление 

``` bash 
crontab -e
```

```crontab
0 3 * * * /bin/bash /opt/shadowgen/opt/scripts/update_certificates.sh >> /opt/shadowgen/log/cert_update.log 2>&1
```