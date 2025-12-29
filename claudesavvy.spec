# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['src/claudesavvy/__main__.py'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('src/claudesavvy/web/templates', 'claudesavvy/web/templates'),
        ('src/claudesavvy/web/static', 'claudesavvy/web/static'),
    ],
    hiddenimports=[
        'flask',
        'jinja2',
        'click',
        'rich',
        'dateutil',
        'claudesavvy',
        'claudesavvy.utils',
        'claudesavvy.utils.paths',
        'claudesavvy.web',
        'claudesavvy.web.app',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludedimports=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='claudesavvy',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
