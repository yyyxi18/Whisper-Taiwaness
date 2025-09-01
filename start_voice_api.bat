@echo off
echo 🎤 啟動語音輸入API...
echo.

REM 檢查Python是否安裝
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python未安裝或不在PATH中
    pause
    exit /b 1
)

echo ✅ Python已安裝
echo.

REM 啟動API
python start_voice_api.py

pause
