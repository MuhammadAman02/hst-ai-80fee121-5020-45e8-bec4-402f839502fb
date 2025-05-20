@echo off
echo Running pre-deployment setup...

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH. Please install Python and try again.
    exit /b 1
)

:: Run the setup_deployment.py script
python setup_deployment.py

if %ERRORLEVEL% neq 0 (
    echo Failed to run setup_deployment.py. Please check the error message above.
    exit /b 1
)

echo Pre-deployment setup completed successfully.
echo You can now deploy your application.