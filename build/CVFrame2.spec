# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for cvframe_v3
#
# Build (run from the build/ folder):
#   Windows : pyinstaller cvframe_v3.spec
#   macOS   : pyinstaller cvframe_v3.spec
#
# Output:
#   Windows : dist/cvframe_v3.exe
#   macOS   : dist/cvframe_v3.app

import sys
import os
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE

IS_MAC = sys.platform == "darwin"

# Project root is one level up from this spec file (which lives in build/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(SPEC)))

a = Analysis(
    [os.path.join(PROJECT_ROOT, 'main.py')],
    pathex=[PROJECT_ROOT, os.path.join(PROJECT_ROOT, 'tools')],
    binaries=[],
    datas=[
        # Bundle the archive/ camera parameter JSONs so the app works out-of-the-box
        (os.path.join(PROJECT_ROOT, 'archive'), 'archive'),
    ],
    hiddenimports=[
        'cv2',
        'numpy',
        'pandas',
        'PyQt5',
        'PyQt5.QtWidgets',
        'PyQt5.QtGui',
        'PyQt5.QtCore',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='cvframe_v3',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# macOS: wrap into a .app bundle
if IS_MAC:
    app = BUNDLE(
        exe,
        name='cvframe_v3.app',
        icon=None,              # Replace with path to a .icns file if you have one
        bundle_identifier='com.cvgroup.cvframe_v3cvframe_v3',
        info_plist={
            'NSHighResolutionCapable': True,
            'CFBundleShortVersionString': '2.0',
        },
    )
