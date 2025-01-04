#!/bin/bash

# Переменные
APP_DIR=~/tg-det                 # Директория с приложением
VENV_DIR=$APP_DIR/.venv          # Директория виртуального окружения
APP_SCRIPT=$APP_DIR/app.py       # Путь к вашему Python-скрипту
PORT=7860                        # Порт, на котором запускается приложение
LOG_FILE=$APP_DIR/app.log        # Файл для логов
ENV="production"

# Функция завершения процессов на порту
kill_process_on_port() {
    local port=$1
    local pid=$(lsof -t -i:$port)
    if [ -n "$pid" ]; then
        echo "Приложение уже запущено на порту $port. Завершаю процесс с PID $pid..."
        kill -9 $pid
        echo "Процесс $pid завершён."
    else
        echo "Порт $port свободен."
    fi
}

# Проверка наличия директории с приложением
if [ ! -d "$APP_DIR" ]; then
    echo "Ошибка: Директория $APP_DIR не найдена."
    exit 1
fi

# Проверка наличия виртуального окружения
if [ ! -d "$VENV_DIR" ]; then
    echo "Ошибка: Виртуальное окружение $VENV_DIR не найдено."
    exit 1
fi

# Проверка наличия Python-скрипта
if [ ! -f "$APP_SCRIPT" ]; then
    echo "Ошибка: Скрипт $APP_SCRIPT не найден."
    exit 1
fi

# Завершаем процессы на указанном порту
kill_process_on_port $PORT

# Запуск приложения с изолированным окружением
(
    echo "Активирую виртуальное окружение..."
    source $VENV_DIR/bin/activate
    export ENV=$ENV

    echo "Запускаю приложение..."
    nohup python $APP_SCRIPT > $LOG_FILE 2>&1 &

    NEW_PID=$!
    echo "Приложение запущено с PID $NEW_PID."
    echo "Логи записываются в $LOG_FILE."

    deactivate
) > /dev/null 2>&1 &

echo "Скрипт завершён. Процесс запущен в фоне."
