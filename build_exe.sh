#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Build EDF Scalp Inspector  (macOS / Linux)
#
#  macOS  → dist/ScalpInspector-macOS.zip    (double-click ScalpInspector.app)
#  Linux  → dist/ScalpInspector-Linux.zip    (run ./ScalpInspector)
#
#  Usage:  bash build_exe.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

PLATFORM="$(uname)"

echo "[1/4] Installing / upgrading PyInstaller..."
python3 -m pip install --quiet --upgrade pyinstaller

echo "[2/4] Installing project dependencies..."
python3 -m pip install --quiet -r requirements.txt

# Linux: ensure tkinter system package is present
if [[ "$PLATFORM" == "Linux" ]]; then
    if ! python3 -c "import tkinter" 2>/dev/null; then
        echo "  tkinter not found — trying: sudo apt-get install python3-tk"
        sudo apt-get install -y python3-tk
    fi
fi

echo "[3/4] Building executable..."
python3 -m PyInstaller --clean --noconfirm scalp_inspector.spec

echo "[4/4] Creating zip archive..."
if [[ "$PLATFORM" == "Darwin" ]]; then
    ZIP="dist/ScalpInspector-macOS.zip"
    (cd dist && zip -qr ScalpInspector-macOS.zip ScalpInspector.app)
    echo ""
    echo "============================================================"
    echo " BUILD COMPLETE"
    echo " App bundle : dist/ScalpInspector.app"
    echo " Archive    : $ZIP  (share this)"
    echo "============================================================"
else
    ZIP="dist/ScalpInspector-Linux.zip"
    (cd dist && zip -qr ScalpInspector-Linux.zip ScalpInspector)
    echo ""
    echo "============================================================"
    echo " BUILD COMPLETE"
    echo " Binary  : dist/ScalpInspector/ScalpInspector"
    echo " Archive : $ZIP  (share this)"
    echo "============================================================"
fi
