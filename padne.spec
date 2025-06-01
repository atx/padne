
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

# Collect all padne submodules including the C++ extension
hiddenimports = collect_submodules('padne')
hiddenimports += collect_submodules('OpenGL.arrays')
hiddenimports += [
    'scipy.special._ufuncs_cxx',
    'scipy.linalg._fblas',
    'scipy.sparse.linalg._isolve',
    'OpenGL.platform.glx',
    'OpenGL.platform.egl',
    'OpenGL.raw.GLX',
    'OpenGL.raw.GLX._types',
    'OpenGL.GLX',
    'PySide6.QtOpenGL',
    'shapely._geos',
    'OpenGL.arrays',
    'OpenGL.arrays.ctypesarrays',
    'OpenGL.arrays.numpymodule',
    'OpenGL.arrays.lists',
    'OpenGL.arrays.numbers',
    'OpenGL.arrays.strings',
    'OpenGL.arrays.nones',
    '_pcbnew',
]


FILTERED_DATA_FILES = [
    'webengine', 'webkit', 'pdf', 'bluetooth', 'multimedia',
    'location', '3d', 'positioning', 'nfc', 'llvm', 'clang',
    'd3dcompiler',
    # Qt development tools:
    'lupdate', 'lrelease', 'linguist',
    'designer', 'assistant',
    'uic', 'rcc', 'moc',
    # Qt icons and translations consume like 100MB+
    'icons', 'share/icons', 'iconengines',
    'translations', 'share/translations',
    # Remove other unnecessary Qt data
    'fonts', 'share/fonts',
    'qml', 'share/qml',
]

# Collect data files
datas = []
datas += collect_data_files('pygerber')
datas += collect_data_files('PySide6')

with open("/tmp/pyside.txt", "w") as f:
    for module in collect_data_files('PySide6'):
        f.write(f"{module}\n")

datas = [
    f for f in datas if not any(
        pattern in f[0].lower() for pattern in FILTERED_DATA_FILES
    )
]

# Collect excludes
excludes = [
    'tkinter',
    # Block PyQt5 and PySide2 to avoid conflicts with PySide6
    # (PyInstaller can't do more than one and we do not use the matplotlib bindings)
    # TODO: The proper solution here is to just not use matplotlib since we
    # only use it for the viridis color map.
    'PyQt5',
    'PySide2',
    'matplotlib',
    'matplotlib.backends.backend_qt5agg',
    'matplotlib.backends.backend_qt5',
    'matplotlib.backends.backend_qt4agg',
    'matplotlib.backends.backend_qt4',
    'matplotlib.backends.backend_gtk3agg',
    'matplotlib.backends.backend_gtk4agg',
    'matplotlib.backends.backend_gtk3cairo',
    'matplotlib.backends.backend_gtk4cairo',
    # Exlude random Qt shit that is wayyyy too large
    # QtWebEngine is like 160MB...
    'PySide6.QtWebEngine',
    'PySide6.QtWebEngineCore', 
    'PySide6.QtWebEngineWidgets',
    'PySide6.QtWebChannel',
    'PySide6.QtWebSockets',
    'PySide6.QtPdf',
    'PySide6.QtPdfWidgets',
    # More shit
    'PySide6.QtBluetooth',
    'PySide6.QtNfc',
    'PySide6.QtMultimedia',
    'PySide6.QtMultimediaWidgets',
    'PySide6.QtLocation',
    'PySide6.QtPositioning',
    'PySide6.Qt3DCore',
    'PySide6.Qt3DRender',
    'PySide6.Qt3DInput',
    'PySide6.Qt3DLogic',
    'PySide6.Qt3DAnimation',
    'PySide6.Qt3DExtras',
    # This causes libLLVM to be included which drags in like 100MB of dependencies...
    'PySide6.lupdate',
    'PySide6.QtQuick',
    'PySide6.QtDBus',
    # Block Gtk (getting pulled in by matplotlib). This loads in like 100MB in share/icons...
    'gi',
    'gi.repository',
    'gi.repository.Gtk',
    'gi.repository.Gdk',
    'gi.repository.GLib',
    'gi.repository.Gio',
    'gi.repository.GObject',
    # Some other random stuff that is not needed
    'IPython',
    'jedi',
    'parso',
    'pygments',
    'pytest',
]

binaries_exclude = [
    '*WebEngine*',
    '*libQt6WebEngine*',
    '*libQt6Pdf*',
    '*libQt6Bluetooth*',
    '*libQt6Multimedia*',
    '*libQt6Location*',
    '*libQt63D*',
    '*libLLVM*',
    'libclang-cpp*'
]

# TODO: This is not ideal, I don't think we should be importing padne in
# the spec file.
import padne._cgal
cgal_path = padne._cgal.__file__

import PyInstaller.utils.hooks

original_collect_data_files = PyInstaller.utils.hooks.collect_data_files

a = Analysis(
    ['padne/cli.py'],
    pathex=[],
    binaries=[
        # Add the compiled CGAL extension
        (cgal_path, 'padne'),
    ],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    binaries_exclude=binaries_exclude,
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
    name='padne',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=False,  # UPX can corrupt Qt libraries
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
