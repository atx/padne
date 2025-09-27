"""
Custom ASV plugin that creates virtualenv environments with --system-site-packages.
This allows access to system-installed packages like pcbnew (KiCad Python API).
"""

import os
import sys
from packaging.version import Version

from asv.plugins import virtualenv
from asv import util
from asv.console import log


class SystemVirtualenv(virtualenv.Virtualenv):
    """
    Virtualenv environment that includes system site packages.
    """

    tool_name = "system_virtualenv"

    def _setup(self):
        """
        Setup the environment on disk using virtualenv with --system-site-packages.
        Then, all of the requirements are installed into it using `pip install`.
        """
        env = dict(os.environ)
        env.update(self.build_env_vars)

        # NOTE: Omit `--wheel=bundle` for virtualenv v20.31 and later.
        import virtualenv
        use_wheel = Version(virtualenv.__version__) < Version('20.31')

        log.info(f"Creating virtualenv with system packages for {self.name}")
        util.check_call(
            [
                sys.executable,
                "-m",
                "virtualenv",
                "--system-site-packages",  # This is the key addition
                *(["--wheel=bundle"] if use_wheel else []),
                "--setuptools=bundle",
                "-p",
                self._executable,
                self._path,
            ],
            env=env,
        )
        log.info(f"Installing requirements for {self.name}")
        self._install_requirements()
