Directive reference
===================

padne is controlled by placing special text directives inside your KiCad
schematic file. Each directive wires a lumped element (voltage source,
current source, resistor, regulator) to one or more pads in your PCB
geometry, or injects a global parameter into the simulation.

This page is the complete reference of every directive padne understands.


Endpoint and value syntax
-------------------------

Every directive that connects to PCB geometry takes one or more
**endpoints**. Endpoints follow these conventions:

* The format is ``DESIGNATOR.PAD``, such as ``R1.1`` or ``U14.A2``.
* Multiple pads can be specified as a comma-separated list, e.g.
  ``p=R1.1,R2.1``.

Numeric **values** accept SI prefixes and units, e.g. ``1k`` or ``500mA``.


Specifying multiple pads
------------------------

The directives support specifying multiple physical pads connected to a
single lumped element node. Internally, this is implemented by connecting
the pads with small resistors in a "star" topology. For a single pad,
these resistors are omitted. For voltage sources, they are always omitted
and the coupling is implemented with 0 V voltage sources instead.

This is particularly useful for multi-pad consumers (where you do not
care or even know which pads will be ingesting the specified current) or
for switching elements with intrinsic internal resistance such as
multi-pad transistors. For example:

.. code-block:: text

   !padne CURRENT i=1.5A f=U1.12,U1.3,U1.21 t=U1.15,U1.5

results in the following lumped elements being wired into the mesh:

.. image:: /_static/images/multipad-topology.svg
   :align: center

The resistance of the coupling resistors can be adjusted with the
optional ``coupling`` parameter (defaults to 1 mΩ).


Lumped element directives
-------------------------

These directives specify a discrete lumped element connected somewhere
in the geometry.

.. _directive-voltage:

VOLTAGE
^^^^^^^

Creates an ideal voltage source between the specified terminals.

**Parameters:**

* ``p=ENDPOINTS`` — Positive terminal(s)
* ``n=ENDPOINTS`` — Negative terminal(s)
* ``v=VALUE`` — Voltage

Example:

.. code-block:: text

   !padne VOLTAGE v=3.3V p=U1.VCC n=U1.GND


.. _directive-current:

CURRENT
^^^^^^^

Creates an ideal current source flowing from one terminal to another.

**Parameters:**

* ``f=ENDPOINTS`` — From terminal(s) (current source)
* ``t=ENDPOINTS`` — To terminal(s) (current sink)
* ``i=VALUE`` — Current magnitude
* ``coupling=VALUE`` — *(Optional)* Coupling resistance for multi-pad
  terminals

Examples:

.. code-block:: text

   !padne CURRENT i=1.0A f=R2.1 t=R2.2
   !padne CURRENT i=3A f=TP2.1,TP3.1,TP4.1 t=TP1.1 coupling=1


.. _directive-resistance:

RESISTANCE
^^^^^^^^^^

Creates a resistor between two terminals.

**Parameters:**

* ``a=ENDPOINTS`` — Terminal A
* ``b=ENDPOINTS`` — Terminal B
* ``r=VALUE`` — Resistance value
* ``coupling=VALUE`` — *(Optional)* Coupling resistance for multi-pad
  terminals

Examples:

.. code-block:: text

   !padne RESISTANCE r=0.1 a=R2.1 b=R2.2
   !padne RESISTANCE r=10000 a=R3.1,R2.1 b=R3.2 coupling=0.1


.. _directive-regulator:

REGULATOR
^^^^^^^^^

A current-controlled current source that can also set a voltage between
its "sense" terminals. This is useful for modeling dependent sources
such as LDOs or DC-DC converters.

**Parameters:**

* ``p=ENDPOINTS`` — Positive voltage sense
* ``n=ENDPOINTS`` — Negative voltage sense
* ``f=ENDPOINTS`` — Current source terminal (from)
* ``t=ENDPOINTS`` — Current sink terminal (to)
* ``v=VALUE`` — Target voltage (``V_p - V_n = v``)
* ``gain=VALUE`` — Current gain factor
* ``coupling=VALUE`` — *(Optional)* Coupling resistance for multi-pad
  terminals

The idea is that you pre-compute the gain factor based on the nominal
values for the regulator input voltage, output voltage and efficiency.
For an LDO the gain is always 1. For a DC-DC converter stepping 12 V
down to 5 V with 80 % efficiency, the gain factor is about 0.52. This
is illustrated by the schematic below:

.. image:: /_static/images/regulator-diagram.svg
   :align: center
   :alt: REGULATOR schematic


Other directives
----------------

These do not create new lumped elements; they inject simulation
parameters instead.

.. _directive-copper:

COPPER
^^^^^^

Specifies a custom copper conductivity for all copper layers in the
project, overriding the default.

**Parameters:**

* ``conductivity=VALUE`` — Copper conductivity in S/m

Example:

.. code-block:: text

   !padne COPPER conductivity=5.97e7

.. note::

   The surface conductivity is computed inside padne by extracting the
   stackup (copper layer thickness in particular) info from your PCB
   file. See *File → Board Setup → Physical Stackup* inside pcbnew.
