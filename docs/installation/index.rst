Installation
============

Three installation methods are supported.

All-in-one binary
-----------------

Simply download the binary from
`here <https://atx.github.io/padne/padne-linux-x64>`_. It ships with
all dependencies bundled (including KiCad). The downside is the
relatively large size (around 200 MB) and some startup time penalty.

For a quick start:

.. code-block:: shell

   wget https://atx.github.io/padne/padne-linux-x64
   chmod +x padne-linux-x64

   # Clone the repository to get the example KiCad projects
   git clone git@github.com:atx/padne.git
   ./padne-linux-x64 gui padne/tests/kicad/via_tht_4layer/via_tht_4layer.kicad_pro

This launches a simple GUI that looks like the animation below:

.. image:: /_static/images/gui-demo.webp
   :align: center
   :alt: padne GUI demo


pipx
----

.. code-block:: shell

   sudo apt install python3-full python3-dev python3-pip git libegl1 \
           libegl1-mesa-dev libmpfr-dev libgmp-dev libboost-dev pipx kicad
   pipx run --spec git+https://github.com/atx/padne.git padne

Note that in this variant (and the one below) padne uses the system
KiCad Python bindings.


Build it yourself
-----------------

This is ideal if you want to hack on padne itself or are looking for a
more lightweight setup. See the :doc:`/developer/quickstart` in the
developer guide for the full walk-through.
