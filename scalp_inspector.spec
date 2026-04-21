# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for EDF Scalp Inspector GUI
# Cross-platform: Windows (.exe), macOS (.app), Linux (binary)
#
# Build:
#   Windows : build_exe.bat
#   macOS   : bash build_exe.sh
#   Linux   : bash build_exe.sh

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# ── Collect all non-Python files that live inside mne and matplotlib ──────────
datas = []
datas += collect_data_files('mne')           # channels, report/js_and_css, html_templates, icons, data …
datas += collect_data_files('matplotlib')    # mpl-data (fonts, style sheets, …)

# ── Hidden imports ─────────────────────────────────────────────────────────────
hiddenimports = (
    collect_submodules('mne')           # mne.report, mne.io.edf, mne.channels …
    + collect_submodules('scipy')       # scipy.signal, scipy.linalg, scipy.sparse …
    + [
        # numpy
        'numpy.core._multiarray_umath',
        'numpy.core._multiarray_tests',
        # matplotlib backend for tkinter
        'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_agg',
        'matplotlib.figure',
        # pandas / openpyxl
        'pandas',
        'openpyxl',
        # stdlib GUI
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'pathlib',
        'threading',
        'datetime',
    ]
)

block_cipher = None

a = Analysis(
    ['edf_scalp_inspector.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'IPython', 'jupyter', 'notebook',
        'pytest', 'wx',
        'PyQt5', 'PyQt6', 'PySide2', 'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── Platform-specific packaging ───────────────────────────────────────────────
if sys.platform == 'darwin':
    # macOS: produce a .app bundle
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name='ScalpInspector',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,          # UPX unreliable on macOS arm64
        console=False,
        icon=None,
    )
    coll = COLLECT(
        exe, a.binaries, a.zipfiles, a.datas,
        strip=False, upx=False, name='ScalpInspector',
    )
    app = BUNDLE(
        coll,
        name='ScalpInspector.app',
        icon=None,
        bundle_identifier='com.scalp.inspector',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': '1.0.0',
        },
    )
else:
    # Windows + Linux: folder distribution with launcher exe
    exe = EXE(
        pyz, a.scripts, [],
        exclude_binaries=True,
        name='ScalpInspector',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,      # no black console window
        icon=None,
    )
    coll = COLLECT(
        exe, a.binaries, a.zipfiles, a.datas,
        strip=False, upx=True, upx_exclude=[],
        name='ScalpInspector',
    )
