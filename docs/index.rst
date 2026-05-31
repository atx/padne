padne
=====

.. image:: /_static/images/banner.png
   :align: center
   :alt: padne — power delivery network analyser for KiCad

*Turn your KiCad designs into interactive voltage drop and current
density visualizations.*

padne is a KiCad-native power delivery network analysis tool. It uses
the finite element method to simulate the voltage drop induced by DC
currents on printed circuit boards. This allows easy identification of
resistive bottlenecks, design of high-current distribution networks, or
implementing complex heating elements.

Features
--------

* **KiCad native** — Loads KiCad projects directly.
* **2.5D FEM solver** — Uses the finite element method to quickly solve
  the Laplace equation.
* **Easy to integrate** — Control via text directives in your schematic
  files.
* **Interactive GUI** — Contains an interactive Qt GUI for exploring
  the computed solution (ParaView export is also available).


.. grid:: 2
    :gutter: 3

    .. grid-item-card:: Installation

        Get padne up and running on your system.

        +++

        .. button-ref:: installation/index
            :expand:
            :color: secondary
            :click-parent:

            To the installation guide

    .. grid-item-card:: User Guide

        Learn how to analyze power delivery networks with padne.

        +++

        .. button-ref:: user_guide/index
            :expand:
            :color: secondary
            :click-parent:

            To the user guide

    .. grid-item-card:: API Reference

        Detailed description of padne's Python API,
        including modules, classes, and functions.

        +++

        .. button-ref:: api/index
            :expand:
            :color: secondary
            :click-parent:

            To the API reference

    .. grid-item-card:: Developer Guide

        Want to contribute? The developer guide
        walks through the internals of padne.

        +++

        .. button-ref:: developer/index
            :expand:
            :color: secondary
            :click-parent:

            To the developer guide


.. toctree::
   :hidden:

   installation/index
   user_guide/index
   developer/index
   api/index
