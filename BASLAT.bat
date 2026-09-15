@echo off
REM Saytı lokal işə salır (Windows). Node.js 20+ lazımdır.
cd /d "%~dp0"
if not exist node_modules call npm install
call npm run dev -- --open
pause
