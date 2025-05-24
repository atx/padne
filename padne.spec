
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
    'matplotlib.backends.backend_qtagg',
    '_pcbnew',
]

# Collect data files
datas = []
datas += collect_data_files('matplotlib')
datas += collect_data_files('pygerber')
datas += collect_data_files('PySide6')

# TODO: This is not ideal, I don't think we should be importing padne in
# the spec file.
import padne._cgal
cgal_path = padne._cgal.__file__

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
    excludes=[
        'tkinter',
        # Block PyQt5 and PySide2 to avoid conflicts with PySide6
        # (PyInstaller can't do more than one and we do not use the matplotlib bindings)
        'PyQt5',
        'PySide2',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.backend_qt5',
        'matplotlib.backends.backend_qt4agg',
        'matplotlib.backends.backend_qt4',
    ],
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
    strip=False,
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
