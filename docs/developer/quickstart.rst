Quickstart
===========

This is a quick guide that explains how to get an editable version of this project
up and running.

We are asuming **Ubuntu 24.04**, but Debian Sid also works.

First, to install the dependencies:

.. code-block:: bash

   sudo apt update
   sudo apt install python3-full python3-dev python3-pip git libegl1 \
        libegl1-mesa-dev libmpfr-dev libgmp-dev libboost-dev kicad


Next, clone the repository and install it using pip:


.. caution::

   If you want to set up a virtual environment, you need to pass `--system-site-packages`,
   as otherwise the KiCad Python API won't be available.


.. code-block:: bash

   pip3 install --user --editable .[test,bench] -v


This will install ``padne`` executable in your ``~/.local/bin`` directory.

The test suite
--------------

The test suite uses ``pytest`` and can be run with the following command:

.. code-block:: bash

   python3 -m pytest


The documentation
-------------------

The documentation is built using ``Sphinx`` and can be built with the following command:

.. code-block:: bash

   make -C docs html  # or use `make -C docs devserver` if you want a live rebuild server

Github Actions
----------------

The repository contains a variety of Github Actions that serve to run the test
suite, build the documentation, PyInstaller binary and validate the benchmark suite.

You can run these actions locally using the ``act`` tool. For example, to run the test action:

.. code-block:: bash

   act -W .github/workflows/run-tests.yaml \
      --matrix kicad-version:10.0 -j build_and_test \
      -P ubuntu-24.04=catthehacker/ubuntu:act-24.04 \
      --bind
