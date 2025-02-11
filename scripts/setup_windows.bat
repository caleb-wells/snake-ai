@echo off
SETLOCAL EnableDelayedExpansion

echo === Snake AI Environment Setup ===

:: Check for Python installation
python --version > NUL 2>&1
if errorlevel 1 (
    echo Python is not installed! Please install Python 3.8 or higher.
    exit /b 1
)

:: Set up virtual environment
echo Setting up virtual environment...
if not exist venv (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment and install dependencies
echo Installing dependencies...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Install development tools
pip install pre-commit
pre-commit install

echo.
echo Setup completed successfully!
echo.
echo Next steps:
echo 1. Activate the virtual environment: venv\Scripts\activate
echo 2. Run the training script: python src\main.py
echo 3. To deactivate the environment: deactivate
echo.

ENDLOCAL
