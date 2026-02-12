@echo off
REM Start PEST++ agents on a Windows machine (Docker Desktop required)
REM Run this script to contribute workers to calibration

setlocal enabledelayedexpansion

REM Configuration - modify these for your setup
if "%MANAGER_IP%"=="" set MANAGER_IP=192.168.1.100
if "%MANAGER_PORT%"=="" set MANAGER_PORT=4004
if "%MINIO_HOST%"=="" set MINIO_HOST=%MANAGER_IP%
if "%MINIO_PORT%"=="" set MINIO_PORT=9000
if "%MINIO_ACCESS_KEY%"=="" set MINIO_ACCESS_KEY=minioadmin
if "%MINIO_SECRET_KEY%"=="" set MINIO_SECRET_KEY=minioadmin
if "%NUM_AGENTS%"=="" set NUM_AGENTS=4
if "%IMAGE%"=="" set IMAGE=modflow-pest-agent:latest

echo ==============================================
echo PEST++ Agent Launcher (Windows)
echo ==============================================
echo Manager: %MANAGER_IP%:%MANAGER_PORT%
echo MinIO: %MINIO_HOST%:%MINIO_PORT%
echo Agents to start: %NUM_AGENTS%
echo ==============================================

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop for Windows
    pause
    exit /b 1
)

REM Stop any existing agents
echo Stopping existing agents...
for /f "tokens=*" %%i in ('docker ps -q --filter "name=pest-agent-"') do (
    docker stop %%i >nul 2>&1
)
for /f "tokens=*" %%i in ('docker ps -aq --filter "name=pest-agent-"') do (
    docker rm %%i >nul 2>&1
)

REM Create work directory
set WORK_DIR=%TEMP%\pest-agents-work
if not exist "%WORK_DIR%" mkdir "%WORK_DIR%"

REM Start agents
echo Starting %NUM_AGENTS% agents...
for /l %%i in (1,1,%NUM_AGENTS%) do (
    set CONTAINER_NAME=pest-agent-%%i
    set AGENT_WORK=%WORK_DIR%\agent-%%i
    if not exist "!AGENT_WORK!" mkdir "!AGENT_WORK!"

    echo   Starting !CONTAINER_NAME!...
    docker run -d ^
        --name !CONTAINER_NAME! ^
        --restart on-failure:5 ^
        -v "!AGENT_WORK!:/work" ^
        -e MANAGER_HOST=%MANAGER_IP% ^
        -e MANAGER_PORT=%MANAGER_PORT% ^
        -e MINIO_HOST=%MINIO_HOST% ^
        -e MINIO_PORT=%MINIO_PORT% ^
        -e MINIO_ACCESS_KEY=%MINIO_ACCESS_KEY% ^
        -e MINIO_SECRET_KEY=%MINIO_SECRET_KEY% ^
        -e PROJECT_ID=%PROJECT_ID% ^
        -e RUN_ID=%RUN_ID% ^
        %IMAGE%

    REM Small delay
    timeout /t 1 /nobreak >nul
)

echo.
echo ==============================================
echo Started %NUM_AGENTS% PEST++ agents
echo ==============================================
echo.
echo To view agent logs:
echo   docker logs -f pest-agent-1
echo.
echo To stop all agents:
echo   for /f "tokens=*" %%i in ('docker ps -q --filter "name=pest-agent-"') do docker stop %%i
echo.
echo To check agent status:
echo   docker ps --filter "name=pest-agent-"
echo.
pause
