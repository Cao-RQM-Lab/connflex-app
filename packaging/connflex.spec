# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build of ConnFlex: one-folder bundle, .app on macOS.

Run from the repository root:  pyinstaller packaging/connflex.spec --noconfirm --clean
"""

import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = os.path.abspath(os.path.join(os.path.dirname(SPEC), ".."))
sys.path.insert(0, ROOT)

from connflex_app import __version__  # noqa: E402

NAME = "ConnFlex"
IS_MAC = sys.platform == "darwin"
IS_WIN = sys.platform.startswith("win")
icon = os.path.join(ROOT, "packaging", "icon.icns" if IS_MAC else "icon.ico")

datas = collect_data_files("connflex_app", includes=["data/*"])
# connflex_core reads its own version from package metadata at import time.
datas += copy_metadata("connflex-core")
datas += copy_metadata("connflex-app")

hiddenimports = (
    collect_submodules("OpenGL.platform")
    + collect_submodules("OpenGL.arrays")
    + collect_submodules("connflex_core")
    + [
        "OpenGL.GL",
        "OpenGL.converters",
        "scipy.linalg",
        "scipy.sparse",
        "scipy.sparse.linalg",
        "scipy.special",
        "shapely",
        "shapely.geometry",
        "gdstk",
        "networkx",
        "tomllib",
    ]
)

excludes = [
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.QtQml",
    "PySide6.QtQuick",
    "PySide6.QtQuick3D",
    "PySide6.QtQuickWidgets",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtCharts",
    "PySide6.QtDataVisualization",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtBluetooth",
    "PySide6.QtNfc",
    "PySide6.QtPositioning",
    "PySide6.QtLocation",
    "PySide6.QtSensors",
    "PySide6.QtSerialPort",
    "PySide6.QtRemoteObjects",
    "PySide6.QtTextToSpeech",
    "PySide6.QtDesigner",
    "PySide6.QtHelp",
    "PySide6.QtSql",
    "PySide6.QtTest",
    "PySide6.QtNetworkAuth",
    "PySide6.QtWebSockets",
    "PySide6.QtWebChannel",
    "PySide6.QtHttpServer",
    "PySide6.QtGraphs",
    "PySide6.QtSpatialAudio",
    "matplotlib",
    "tkinter",
    "IPython",
    "pytest",
]

a = Analysis(
    [os.path.join(ROOT, "packaging", "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon if os.path.exists(icon) else None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name=NAME,
)

if IS_MAC:
    app = BUNDLE(
        coll,
        name=f"{NAME}.app",
        icon=icon if os.path.exists(icon) else None,
        bundle_identifier="org.connflex.app",
        version=__version__,
        info_plist={
            "CFBundleName": NAME,
            "CFBundleDisplayName": NAME,
            "CFBundleShortVersionString": __version__,
            "CFBundleVersion": __version__,
            "NSHighResolutionCapable": True,
            "NSPrincipalClass": "NSApplication",
            "LSMinimumSystemVersion": "12.0",
            "LSApplicationCategoryType": "public.app-category.developer-tools",
            "CFBundleDocumentTypes": [
                {
                    "CFBundleTypeName": "GDSII layout",
                    "CFBundleTypeExtensions": ["gds", "gds2"],
                    "CFBundleTypeRole": "Viewer",
                }
            ],
        },
    )
