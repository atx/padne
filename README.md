# padne
*Turn your KiCad designs into an interactive voltage drop and current density visualizations*

<img width="1875" height="600" alt="image" src="https://github.com/user-attachments/assets/5c27ca8f-dac5-44c1-97ec-30a2e30950d4" />


padne is a KiCad-native power delivery network analysis tool. It uses the finite element method in order to simulate the voltage drop induced by DC currents on printed circuit boards. This allows easy identification of resistive bottlenecks, design of high current distribution networks or implementing complex heating elements.

![image](https://github.com/atx/padne/actions/workflows/run-tests.yaml/badge.svg) ![image](https://github.com/atx/padne/actions/workflows/build-binary.yaml/badge.svg)

## Features
 - **KiCad Native** - Loads KiCad projects directly
 - **2.5D FEM Solver** - Uses the finite element method to quickly solve the Laplace equation
 - **Easy to integrate** - Control via text directives in your schematic files
 - **Interactive GUI** - Contains an interactive Qt GUI for exploring the computed solution (ParaView export is also available)

## Installation

Two methods are supported at the moment:

### All-in-one binary

Simply download the binary from [here](https://atx.github.io/padne/padne-linux-x64). This comes with all dependencies already bundled. The downside is the relatively large size (around 200MB binary) and some startup time penalty (a few seconds or so).

For a quick start, you can run:
```sh
wget https://atx.github.io/padne/padne-linux-x64
chmod +x padne-linux-x64

# Clone the repository to get the example KiCad projects
git clone git@github.com:atx/padne.git
./padne-linux-x64 gui padne/tests/kicad/via_tht_4layer/via_tht_4layer.kicad_pro
```

This launches a simple GUI that looks like the GIF below:

<p align="center">
<img src="https://github.com/user-attachments/assets/2f2f0753-b140-4d7a-9dfd-50b160771076">
</p>


### Build-it-yourself

This is ideal if you want to hack on padne itself or are looking for a more lightweight setup. On Ubuntu, we proceed as follows:

```
sudo apt update
sudo apt upgrade
sudo apt install python3-full python3-dev python3-pip git libegl1 libegl1-mesa-dev libmpfr-dev libgmp-dev libboost-dev kicad
pip3 install --user -e .[test] -v
```

## Running the solver

To run the solver and display the solution in one step, do:

```
padne gui my_project.kicad_pro
```

There is also a capability for saving a solution and displaying it later:

```
padne solve my_project.kicad_pro pdn.padne
padne show pdn.padne
```

For advanced visualization, you can export solutions to ParaView/VTK format:

```
padne paraview pdn.padne output_directory/
```

This creates separate `.vtu` files for each layer that can be opened in ParaView for visualization and analysis.

> [!TIP]
> Run `padne gui --help` to see the exposed mesher parameters.

## Usage



padne is controlled by placing special text directives inside your KiCad schematic file. The idea is that you use these directives to wire up lumped elements (voltage sources, current sources etc.) to your PCB geometry.

![image](https://github.com/user-attachments/assets/0171c06f-a332-4b24-bcce-07ca67d9f857)


**Index of directives:**
- [VOLTAGE](#voltage) - Create an ideal voltage source between the specified terminals.
- [CURRENT](#current) - Create an ideal current source flowing from one terminal to another.
- [RESISTANCE](#resistance) - Creates a resistor between two terminals.
- [REGULATOR](#regulator) - Creates a current-controlled current source
- [COPPER](#copper) - Serves to specify custom copper conductivity


### Lumped element directives

These directives specify a discrete lumped element connected to somewhere in the geometry.

For endpoints
 * The format is `DESIGNATOR.PAD`, such as `R1.1` or `U14.A2`
 * It is possible to specify multiple pads as a comma-separated list, such as `p=R1.1,R2.1`

For values
 * SI prefixes and units are supported, such as `1k` or `500mA`

#### VOLTAGE

Creates an ideal voltage source between the specified terminals

**Parameters:**
- `p=ENDPOINTS` - Positive terminal(s)
- `n=ENDPOINTS` - Negative terminal(s)
- `v=VALUE` - Voltage

For example:
```
!padne VOLTAGE v=3.3V p=U1.VCC n=U1.GND
```

#### CURRENT

Creates an ideal current source flowing from one terminal to another.

**Parameters:**
- `f=ENDPOINTS` - From terminal(s) (current source)
- `t=ENDPOINTS` - To terminal(s) (current sink)
- `i=VALUE` - Current magnitude
- `coupling=VALUE` - (Optional) Coupling resistance, for multi-pad terminals

For example:
```
!padne CURRENT i=1.0A f=R2.1 t=R2.2
!padne CURRENT i=3A f=TP2.1,TP3.1,TP4.1 t=TP1.1 coupling=1
```

#### RESISTANCE

Creates a resistor between two terminals.

**Parameters:**
- `a=ENDPOINTS` - Terminal A
- `b=ENDPOINTS` - Terminal B
- `r=VALUE` - Resistance value
- `coupling=VALUE` - (Optional) Coupling resistance, for multi-pad terminals

For example:
```
!padne RESISTANCE r=0.1 a=R2.1 b=R2.2
!padne RESISTANCE r=10000 a=R3.1,R2.1 b=R3.2 coupling=0.1
```

#### REGULATOR

This is effectively a current-controlled current source, that can also set a voltage between its "sense" terminals. This is useful for modeling dependent sources, such as LDOs or DCDC converters.


**Parameters:**
- `p=ENDPOINTS` - Positive voltage sense
- `n=ENDPOINTS` - Negative voltage sense
- `f=ENDPOINTS` - Current source terminal (from)
- `t=ENDPOINTS` - Current sink terminal (to)
- `v=VALUE` - Target voltage (V_p - V_n = v)
- `gain=VALUE` - Current gain factor
- `coupling=VALUE` - (Optional) Coupling resistance, for multi-pad terminals

The idea is that you pre-compute the gain factor based on the nominal values for the regulator input voltage, output voltage and efficiency. For example, for an LDO, the gain is always equal to 1. For a DCDC converter switching from 12V to 5V, with efficiency of 80%, the gain factor would be about 0.52). This is illustrated by the schematic below:

![regulator](https://github.com/user-attachments/assets/734e884d-0e93-4917-9c5f-519de0301a60)

### Specifying multiple pads

The directives support specifying multiple physical pads connected to a single lumped element node. Internally, this is implemented by  connecting the pads by small resistors in a "star" topology. Note that for a single pad, these resistors are omitted. For voltage sources, they are always omitted and the coupling is implemented by 0V voltage sources.

This is particularly useful for specifying multi-pad consumers (where you do not care or even know which pads are going to be ingesting the specified current) or switching elements with intrinsic internal resistance, such as multi-pad transistors. For example, this directive:
```
!padne CURRENT i=1.5A f=U1.12,U1.3,U1.21 t=U1.15,U1.5
```
results in the following lumped elements being wired into the mesh:

<img src="https://github.com/user-attachments/assets/4dd49da1-7702-42b3-8763-d371c681aada"/>

Note that the resistance of the coupling resistors can be adjusted by the optional `coupling` parameter (defaults to 1mΩ).

### Other directives

These do not create new lumped elements, but serve instead as a way to
inject various parameters into the simulation process.

#### COPPER

Specifies custom copper conductivity for all copper layers in the project, overriding the default value.

**Parameters:**
- `conductivity=VALUE` - Copper conductivity in S/m

For example:
```
!padne COPPER conductivity=5.97e7
```

> [!NOTE]
> The surface conductivity is computed inside padne by extracting the stackup (copper layer thickness in particular) info from your PCB file. See `File -> Board Setup -> Physical Stackup` inside pcbnew.
