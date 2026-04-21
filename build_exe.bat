@echo off
REM ─────────────────────────────────────────────────────────────────────────────
REM  Build EDF Scalp Inspector  (Windows)
REM  Output: dist\ScalpInspector-Windows.zip  (ready to share)
REM ─────────────────────────────────────────────────────────────────────────────
setlocal

echo [1/4] Installing / upgrading PyInstaller...
python -m pip install --quiet --upgrade pyinstaller
if errorlevel 1 ( echo ERROR: pip failed. Make sure Python is on PATH. & pause & exit /b 1 )

echo [2/4] Installing project dependencies...
python -m pip install --quiet -r requirements.txt
if errorlevel 1 ( echo ERROR: dependency install failed. & pause & exit /b 1 )

echo [3/4] Building executable...
python -m PyInstaller --clean --noconfirm scalp_inspector.spec
if errorlevel 1 ( echo ERROR: PyInstaller build failed. See output above. & pause & exit /b 1 )

echo [4/4] Creating zip archive...
powershell -NoProfile -Command "Compress-Archive -Force -Path dist\ScalpInspector -DestinationPath dist\ScalpInspector-Windows.zip"

echo.
echo ============================================================
echo  BUILD COMPLETE
echo  Launcher : dist\ScalpInspector\ScalpInspector.exe
echo  Archive  : dist\ScalpInspector-Windows.zip  (share this)
echo ============================================================
pause
