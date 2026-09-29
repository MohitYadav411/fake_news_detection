@echo off

echo Running features extraction...
python src\run_features.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo Training basic and intermediate models...
python src\train.py --tier all
if %errorlevel% neq 0 exit /b %errorlevel%

echo Training hardcore models...
python src\train_hardcore.py
if %errorlevel% neq 0 exit /b %errorlevel%

echo All training completed successfully!
