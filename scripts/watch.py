from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os
import signal
import time
import subprocess

APP_PORT=7860

def kill_port(port):
    try:
        result = subprocess.check_output(f"netstat -ano | findstr :{port}", shell=True).decode()
        for line in result.splitlines():
            if "LISTENING" in line or "ESTABLISHED" in line:
                pid = line.strip().split()[-1]
                os.system(f"taskkill /F /PID {pid}")
                print(f"Процесс {pid}, использующий порт {port}, завершён.")
    except Exception as e:
        print(f"Ошибка при завершении процесса на порту {port}: {e}")

class RestartHandler(FileSystemEventHandler):
    def __init__(self, process, port=APP_PORT, watch_dir="."):
        self.process = process
        self.port = port
        self.watch_dir = os.path.abspath(watch_dir)
        self.last_modified = None

    def on_modified(self, event):
        if not event.is_directory and os.path.dirname(event.src_path) == self.watch_dir and event.src_path.endswith(".py"):  # Отслеживаем только изменения в Python-файлах
            current_time = time.time()
            if self.last_modified is None or current_time - self.last_modified > 1:
                print(f"Изменения в {event.src_path}. Перезапуск...")
                os.kill(self.process.pid, signal.SIGTERM)  # Убиваем текущий процесс
                kill_port(self.port)
                time.sleep(1)  # Даем немного времени на перезапуск
                self.process = subprocess.Popen(["run_app.cmd"])  # Перезапуск приложения


if __name__ == "__main__":
    process = subprocess.Popen(["run_app.cmd"])  # Запуск приложения
    watch_dir = os.getcwd()
    event_handler = RestartHandler(process, APP_PORT, watch_dir)
    observer = Observer()
    observer.schedule(event_handler, path=watch_dir, recursive=False)  # Отслеживание текущей директории
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        process.terminate()
    observer.join()
