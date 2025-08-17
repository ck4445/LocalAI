# -*- mode: python ; coding: utf-8 -*-

# This is the final PyInstaller spec file for Local AI Chat.
# It is configured to create a production-ready, single-folder application.

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # Frontend files are placed in the root of the distribution folder.
        ('index.html', '.'),
        ('animations.js', '.'),

        # The bundled Ollama executable for user convenience.
        # This creates a 'bin' folder inside your application directory.
        ('bin/ollama.exe', 'bin'),

        # The entire 'data' folder (with modelnames.json, etc.) is included.
        # This is crucial for providing default settings and friendly model names.
        ('data', 'data')
    ],
    hiddenimports=[],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False
)
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LocalAIChat',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # 'console=False' is the key for a production release.
    # It prevents the black command prompt window from appearing.
    console=False,
    # 'icon' sets the icon for the .exe file.
    # This requires an 'icon.ico' file in your project's root directory.
    icon='icon.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    # 'name' sets the name of the final output folder in the 'dist' directory.
    name='LocalAIChat'
)