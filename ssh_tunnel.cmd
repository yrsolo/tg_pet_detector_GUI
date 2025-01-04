@echo off
title SSH Tunnel to VDS (Port 9001)
echo runing SSH-tunnel...

:start
ssh -v -N -R 9001:127.0.0.1:9001 yrsolo@89.169.132.198 -o ServerAliveInterval=30
if %ERRORLEVEL% NEQ 0 (
    echo ERROR. Reboot after 5 sec...
    timeout /t 5 /nobreak >nul
    goto start
)

echo End of tunnel.
pause