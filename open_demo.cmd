@echo off
setlocal
cd /d "%~dp0"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\open_demo.ps1" %*
if errorlevel 1 (
  echo.
  echo Demo could not be opened. Check the message above.
  pause
  exit /b 1
)
endlocal
