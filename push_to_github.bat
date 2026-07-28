@echo off
title Kush Tracker Lite - Auto GitHub Push
echo ========================================================
echo 🚀 Kush Tracker Lite - Automatic GitHub Push Tool
echo ========================================================
echo.

cd /d "C:\projects\Kush Tracker Lite"

if not exist ".git" (
    echo [Git] Initializing Git repository...
    git init
    git branch -M main
)

echo.
set /p msg="Enter Commit Message (Press Enter for default timestamp): "
if "%msg%"=="" set msg=Update Kush Tracker Lite - %date% %time%

echo.
echo [1/3] Staging modified files...
git add .

echo.
echo [2/3] Creating commit: "%msg%"
git commit -m "%msg%"

echo.
echo [3/3] Pushing changes to GitHub (origin main)...
git push origin main

if %errorlevel% neq 0 (
    echo.
    echo ⚠️ Push failed or remote origin not connected yet.
    echo If this is your first push, connect your GitHub repo using:
    echo   git remote add origin https://github.com/YOUR_USERNAME/kush-tracker-lite.git
    echo   git push -u origin main
) else (
    echo.
    echo ✅ SUCCESS! Changes pushed to GitHub. Streamlit Cloud will auto-deploy in ~1 minute.
)

echo.
pause
