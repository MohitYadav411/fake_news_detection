@echo off
setlocal
echo ----------------------------------------
echo GitHub Easy Push Script
echo ----------------------------------------

set /p message="Enter your commit message (e.g., 'Completed Phase 7 and 8 backend'): "

if "%message%"=="" (
    echo Commit message cannot be empty.
    exit /b 1
)

echo Adding files to git...
git add .

echo Committing...
git commit -m "%message%"

echo Pushing to GitHub...
git push origin main

echo ----------------------------------------
echo Done! Your code is now on GitHub.
echo ----------------------------------------
pause
