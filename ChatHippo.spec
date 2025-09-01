# -*- mode: python ; coding: utf-8 -*-

# This is the final PyInstaller spec file for ChatHippo.
# It is configured to create a production-ready, single-folder application
# that includes all necessary web assets (HTML, CSS, JS) and data files.

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        # --- Web Frontend Assets ---
        # HTML files
        ('index.html', '.'),
        ('model-store.html', '.'),
        ('list.html', '.'),

        # Styles and scripts
        ('css', 'css'),
        ('js', 'js'),
        ('loadinganimations', 'loadinganimations'),
        ('animations.js', '.'),

        # Core app data (defaults); app recreates if missing
        ('data', 'data'),
        ('list.txt', '.'),

        # Optional docs / extras
        ('llamacppdocs.txt', '.'),

        # Icon (optional, if present)
        ('icon.ico', '.'),

        # Ship entire bin directory if present (e.g., helpers)
        ('bin', 'bin'),
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
    name='ChatHippo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # 'console=False' is the key for a production release.
    # It prevents the black command prompt window from appearing.
    console=False,
    # 'icon' sets the icon for the .exe file.
    # The 'icon.ico' file must be present in the project's root directory.
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
    name='ChatHippo'
)
