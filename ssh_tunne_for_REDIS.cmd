@echo off
set SERVER=89.169.132.198
set USER=yrsolo

rem Redis на сервере (куда прокидываем)
set RHOST=127.0.0.1
set RPORT=6379

rem Локальный порт на домашнем ПК (куда будет коннектиться твой код)
set LHOST=127.0.0.1
set LPORT=6379

:loop
echo.
echo ==== START LOCAL SSH TUNNEL %LHOST%:%LPORT% -> %RHOST%:%RPORT% on %SERVER% ====

ssh -v ^
  -N ^
  -L %LHOST%:%LPORT%:%RHOST%:%RPORT% ^
  -o ExitOnForwardFailure=yes ^
  -o ServerAliveInterval=60 ^
  -o ServerAliveCountMax=3 ^
  %USER%@%SERVER%

echo ERROR: SSH exited, restart in 5 sec...
timeout /t 5 >nul
goto loop