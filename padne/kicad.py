
import warnings

import collections
import enum
import math
import logging
import pathlib
import pcbnew
import pygerber.gerber.api
import pygerber.vm
import shapely
import shapely.affinity
import tempfile

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Iterator, List, ClassVar

from . import problem, units


log = logging.getLogger(__name__)

# This file is responsible for loading KiCad files and converting them to our
# internal representation.


def nm_to_mm(f: float) -> float:
    return f / 1000000


@dataclass
class StackupItem:

    class Type(enum.Enum):
        DIELECTRIC = "DIELECTRIC"
        COPPER = "COPPER"

    name: str
    thickness: float
    # Beware that this field is in S/mm, not S/m (!!!)
    conductivity: Optional[float] = None

    @property
    def conductance(self):
        return self.thickness * self.conductivity


@dataclass
class Stackup:
    items: list[StackupItem]

    def index_by_name(self, name: str) -> int:
        return next(
            (i for i, item in enumerate(self.items) if item.name == name)
        )


DEFAULT_STACKUP = Stackup(
    items=[
        StackupItem(name="F.Cu", thickness=0.035, conductivity=5.95e4),
        StackupItem(name="dielectric 1", thickness=1.51),
        StackupItem(name="B.Cu", thickness=0.035, conductivity=5.95e4),
    ]
)


def copper_layers(board: pcbnew.BOARD) -> Iterator[int]:
    """
    Iterate over layer IDs of copper layers in the given KiCad board.
    """
    for layer_id in range(pcbnew.PCB_LAYER_ID_COUNT):
        if not board.IsLayerEnabled(layer_id) or not pcbnew.IsCopperLayer(layer_id):
            continue
        yield layer_id


def extract_stackup_from_kicad_pcb(board: pcbnew.BOARD) -> Stackup:
    """
    Extract the stackup from a KiCad PCB file.
    """
    # Unfortunately, the Python pcbnew API does not support reading the stackup
    # directly. We need to parse the file manually...
    with open(board.GetFileName(), "r") as f:
        import sexpdata
        sexpr = sexpdata.load(f)

    stackup_items = []

    if sexpr[0] != sexpdata.Symbol("kicad_pcb"):
        raise ValueError("Unknown initial key in the PCB file")

    setup = next((item for item in sexpr if isinstance(item, list) and
                 item and item[0] == sexpdata.Symbol('setup')), None)

    if not setup:
        raise ValueError("Could not find setup section in PCB file")

    stackup = next((item for item in setup if isinstance(item, list) and
                   item and item[0] == sexpdata.Symbol('stackup')), None)

    if not stackup:
        # TODO: Return verify that the board only has two layers
        # I am not sure if it is possible to have no stackup section and
        # more than two layers. It seems KiCad generates the section
        # on every change in the stackup window...
        return DEFAULT_STACKUP

    # Process each layer in the stackup
    for item in stackup:
        if not item[0] == sexpdata.Symbol("layer"):
            continue

        layer_name = item[1]
        layer_type = None
        thickness = None
        conductivity = None

        # Find properties in the layer definition
        for prop in item:
            if not isinstance(prop, list) or len(prop) < 2:
                continue

            match str(prop[0]):
                case "type":
                    layer_type_str = prop[1].lower()
                    if "copper" in layer_type_str:
                        layer_type = StackupItem.Type.COPPER
                        conductivity = 5.95e4  # S/mm (!!! not S/m)
                    elif any(x in layer_type_str for x in ['core', 'prepreg']):
                        layer_type = StackupItem.Type.DIELECTRIC
                    else:
                        # We do not care just yet. Those are usually silkscreen
                        # and mask layers
                        pass
                case "thickness":
                    thickness = float(prop[1])
                case _:
                    pass

        if not layer_type or thickness is None:
            # Shrug
            continue

        stackup_items.append(StackupItem(
            name=layer_name,
            thickness=thickness,
            conductivity=conductivity
        ))

    return Stackup(items=stackup_items)


@dataclass(frozen=True)
class Directive:
    """
    Represents a single directive in the schematic.
    """
    name: str
    params: dict[str, str]

    @classmethod
    def parse(cls, directive: str) -> 'Directive':
        """
        Parse a directive string with key-value pairs into a Directive object.

        Format: !padne DIRECTIVE_NAME key1=value1 key2=value2 ...

        Args:
            directive: The directive string to parse

        Returns:
            A Directive object with the parsed name and parameters

        Raises:
            ValueError: If the directive format is invalid
        """
        tokens = directive.split()

        # Check prefix
        if not tokens or tokens[0] != "!padne":
            raise ValueError("Directive must start with '!padne'")

        # Check directive name
        if len(tokens) < 2:
            raise ValueError("Directive must have a name")

        name = tokens[1]
        params = {}

        # Parse key-value pairs
        for param in tokens[2:]:
            if '=' not in param:
                raise ValueError(f"Invalid parameter format: {param}")

            key, value = param.split('=', 1)

            if not key:
                raise ValueError("Empty parameter key")

            # Handle quoted values
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]

            params[key] = value

        return cls(name=name, params=params)


@dataclass(frozen=True)
class Endpoint:
    designator: str
    pad: str


@dataclass(frozen=True)
class LayerPoint:
    layer: str
    point: shapely.geometry.Point


@dataclass
class PadIndex:
    """
    This class effectively serves as a mapper from Endpoint to a bunch of LayerPoints
    that can then be used to construct a problem.Connection object.
    """
    mapping: dict[Endpoint, list[LayerPoint]] = field(default_factory=dict)

    def find_by_endpoint(self, ep: Endpoint) -> list[LayerPoint]:
        return self.mapping[ep]

    def load_smd_pads(self, board: pcbnew.BOARD) -> None:
        """
        Load all SMD pads from the given PCB board into the mapping.
        """
        for footprint in board.GetFootprints():
            designator = footprint.GetReference()

            for pad_obj in footprint.Pads():
                # Only process SMD pads
                if pad_obj.GetAttribute() != pcbnew.PAD_ATTRIB_SMD:
                    continue

                pad_name = pad_obj.GetName()
                endpoint = Endpoint(designator=designator, pad=pad_name)

                # Get pad position and convert from nm to mm
                position = pad_obj.GetPosition()
                x_mm = nm_to_mm(position.x)
                y_mm = nm_to_mm(position.y)

                # Get the layer for SMD pads
                layer_id = pad_obj.GetLayer()

                # Handle flipped footprints
                if footprint.IsFlipped():
                    match layer_id:
                        case pcbnew.F_Cu:
                            layer_id = pcbnew.B_Cu
                        case pcbnew.B_Cu:
                            layer_id = pcbnew.F_Cu
                        case _:
                            raise NotImplementedError("Flipped footprints with SMD pads on internal layers are not supported yet")
                    footprint_pos_y = nm_to_mm(footprint.GetPosition().y)
                    y_mm = 2 * footprint_pos_y - y_mm

                layer_name = board.GetLayerName(layer_id)
                point = shapely.geometry.Point(x_mm, y_mm)
                layer_point = LayerPoint(layer=layer_name, point=point)

                # Add to mapping (initialize list if endpoint doesn't exist)
                if endpoint not in self.mapping:
                    self.mapping[endpoint] = []
                self.mapping[endpoint].append(layer_point)

    def insert_via_specs(self, via_specs: list["ViaSpec"]) -> None:
        """
        Insert via specifications into the mapping.
        Uses all boundary points of the via shape for all layers it connects.
        """
        for via_spec in via_specs:
            # Only process vias that have an endpoint (THT pads)
            if via_spec.endpoint is None:
                continue

            if not via_spec.layer_names:
                continue

            # Get all boundary points from the via shape
            boundary_coords = list(via_spec.shape.exterior.coords)

            # For each layer the via connects to
            for layer_name in via_spec.layer_names:
                # Do not forget that boundary_cords[0] == boundary_coords[-1]
                for x, y in boundary_coords[:-1]:
                    point = shapely.geometry.Point(x, y)
                    layer_point = LayerPoint(layer=layer_name, point=point)

                    # Add to mapping (initialize list if endpoint doesn't exist)
                    if via_spec.endpoint not in self.mapping:
                        self.mapping[via_spec.endpoint] = []
                    self.mapping[via_spec.endpoint].append(layer_point)


def _parse_endpoints_param(param_str: Optional[str]) -> list[Endpoint]:
    """Helper to parse a comma-separated string of endpoints."""
    if not param_str:
        return []
    return [
        parse_endpoint(ep_str.strip())
        for ep_str in param_str.split(',')
        if ep_str.strip()
    ]


@dataclass
class BaseLumpedSpec:
    """
    Represents a base class that specifies a _single_ lumped element
    being wired to a bunch of pads on the PCB.
    """

    endpoints: dict[str, list[Endpoint]] = \
        field(default_factory=lambda: collections.defaultdict(list))

    values: dict[str, float] = field(default_factory=dict)

    coupling: float = 0.001

    # To be overridden by subclasses
    endpoint_names: ClassVar[dict[str, str]] = {}
    value_names: ClassVar[dict[str, str]] = {}
    lumped_type: ClassVar[type] = None

    @classmethod
    def from_directive(cls, directive: Directive) -> 'BaseLumpedSpec':
        """
        Parse a directive into a BaseLumpedSpec object.

        Args:
            directive: The directive to parse

        Returns:
            A BaseLumpedSpec object with parsed endpoints and values
        """
        spec = cls()

        for name in cls.endpoint_names.keys():
            if name not in directive.params:
                raise ValueError(f"Missing endpoint parameter: {name} for {directive.name}")
            spec.endpoints[name].extend(
                _parse_endpoints_param(directive.params[name])
            )

        for name in cls.value_names.keys():
            if name not in directive.params:
                raise ValueError(f"Missing value parameter: {name} for {directive.name}")
            spec.values[name] = units.Value.parse(directive.params[name]).value

        # Parse optional coupling parameter
        if "coupling" in directive.params:
            spec.coupling = units.Value.parse(directive.params["coupling"]).value

        return spec

    def construct(self,
                  pad_index: PadIndex,
                  layer_dict: dict[str, problem.Layer]
                  ) -> problem.BaseLumped:
        """
        Constructs a problem.BaseLumped element from the current specification.
        This method should be implemented in subclasses to create the specific
        type of lumped element.
        """
        # First, we construct the NodeID object that are connected to the endpoints
        # of the internal lumped element we are going to create
        internal_nodes = {
            internal_arg_name: problem.NodeID()
            for internal_arg_name in self.endpoint_names.values()
        }

        # Next, we walk through the endpoint lists for every endpoint name
        connections = []
        elements = []
        for directive_param_name, endpoints_list in self.endpoints.items():
            if not endpoints_list:
                raise ValueError(f"No endpoints specified for {directive_param_name} in {self.__class__.__name__}")

            internal_arg_name = self.endpoint_names[directive_param_name]

            layerpoints = [
                lp
                for ep in endpoints_list
                for lp in pad_index.find_by_endpoint(ep)
            ]

            if len(layerpoints) == 1:
                lp = layerpoints[0]
                layer = layer_dict[lp.layer]
                conn = problem.Connection(
                    layer=layer,
                    point=lp.point,
                    node_id=internal_nodes[internal_arg_name]
                )
                connections.append(conn)
            else:
                # If there are multiple endpoints, we create a "star"
                # shaped resistor network leading from the endpoints to the
                # internal node
                for lp in layerpoints:
                    layer = layer_dict[lp.layer]
                    resistor = problem.Resistor(
                        a=problem.NodeID(),
                        b=internal_nodes[internal_arg_name],
                        resistance=self.coupling,
                    )
                    conn = problem.Connection(
                        layer=layer,
                        point=lp.point,
                        node_id=resistor.a,
                    )
                    elements.append(resistor)
                    connections.append(conn)

        # Now we can create the internal lumped element
        if not self.lumped_type:
            raise NotImplementedError("lumped_type must be defined in subclasses")
        kwargs = internal_nodes.copy()
        kwargs.update({
            arg_name: self.values[name]
            for name, arg_name in self.value_names.items()
        })
        internal_lumped = self.lumped_type(**kwargs)

        elements.append(internal_lumped)

        # Finally, we create the Network object
        return problem.Network(
            connections=connections,
            elements=elements,
        )


class ResistorSpec(BaseLumpedSpec):

    endpoint_names = {"a": "a", "b": "b"}
    value_names = {"r": "resistance"}
    lumped_type = problem.Resistor


class VoltageSourceSpec(BaseLumpedSpec):

    endpoint_names = {"p": "p", "n": "n"}
    value_names = {"v": "voltage"}
    lumped_type = problem.VoltageSource

    def construct(self,
                  pad_index: PadIndex,
                  layer_dict: dict[str, problem.Layer]
                  ) -> problem.Network:
        """
        Custom construct method for voltage sources that properly handles
        multiple endpoints without introducing coupling resistance.

        Strategy:
        1. Create main voltage source between first positive and first negative endpoints
        2. Create 0V voltage sources to connect additional endpoints to the first ones
        """
        elements = []

        # Get endpoint lists
        p_endpoints = self.endpoints["p"]
        n_endpoints = self.endpoints["n"]

        if not p_endpoints:
            raise ValueError("No positive endpoints specified for voltage source")
        if not n_endpoints:
            raise ValueError("No negative endpoints specified for voltage source")

        # Create connections for all endpoints
        p_connections = []
        n_connections = []

        for endpoints, connections in zip([p_endpoints, n_endpoints],
                                          [p_connections, n_connections]):
            layerpoints = [
                lp
                for ep in endpoints
                for lp in pad_index.find_by_endpoint(ep)
            ]

            for lp in layerpoints:
                layer = layer_dict[lp.layer]
                conn = problem.Connection(layer=layer, point=lp.point)
                connections.append(conn)

        # Create main voltage source between first positive and first negative
        main_voltage_source = problem.VoltageSource(
            p=p_connections[0].node_id,
            n=n_connections[0].node_id,
            voltage=self.values["v"]
        )
        elements.append(main_voltage_source)

        # Create 0V voltage sources to connect additional positive terminals to first positive
        for i in range(1, len(p_connections)):
            zero_v_source = problem.VoltageSource(
                p=p_connections[i].node_id,
                n=p_connections[0].node_id,
                voltage=0.0
            )
            elements.append(zero_v_source)

        # Create 0V voltage sources to connect additional negative terminals to first negative
        for i in range(1, len(n_connections)):
            zero_v_source = problem.VoltageSource(
                p=n_connections[i].node_id,
                n=n_connections[0].node_id,
                voltage=0.0
            )
            elements.append(zero_v_source)

        return problem.Network(
            connections=(p_connections + n_connections),
            elements=elements
        )


class CurrentSourceSpec(BaseLumpedSpec):
    endpoint_names = {"f": "f", "t": "t"}
    value_names = {"i": "current"}
    lumped_type = problem.CurrentSource


@dataclass
class RegulatorSpec(BaseLumpedSpec):
    endpoint_names: ClassVar[dict[str, str]] = {
        "p": "v_p",
        "n": "v_n",
        "f": "s_f",
        "t": "s_t",
    }
    value_names: ClassVar[dict[str, str]] = {
        "v": "voltage",
        "gain": "gain",
    }
    lumped_type: ClassVar[type] = problem.VoltageRegulator


@dataclass(frozen=True)
class ViaSpec:
    """
    Specifies a via in the PCB.
    """
    point: shapely.geometry.Point
    drill_diameter: float
    layer_names: list[str]
    endpoint: Optional[Endpoint] = None
    shape: shapely.geometry.Polygon = field(init=False)

    def __post_init__(self):
        radius = self.drill_diameter / 2
        shape = shapely.geometry.Point(self.point).buffer(radius, quad_segs=4)
        object.__setattr__(self, 'shape', shape)

    def compute_resistance(self, length: float, plating_thickness: float, conductivity: float) -> float:
        """
        Compute via resistance using hollow cylinder model.

        Args:
            length: Via length in mm
            plating_thickness: Copper plating thickness in mm
            conductivity: Copper conductivity in S/mm

        Returns:
            Resistance in ohms
        """
        # Cross-sectional area of hollow cylinder (plated wall thickness)
        outer_radius = self.drill_diameter / 2 + plating_thickness
        inner_radius = self.drill_diameter / 2
        cross_sectional_area = math.pi * (outer_radius**2 - inner_radius**2)

        # Resistance = length / (conductivity * area)
        return length / (conductivity * cross_sectional_area)


@dataclass(frozen=True)
class KiCadProject:
    """
    Represents a KiCad project with paths to its component files.
    """
    pro_path: Path
    pcb_path: Path
    sch_path: Path

    @property
    def name(self) -> str:
        """Get the project name (stem of the project file)."""
        return self.pro_path.stem

    @classmethod
    def from_pro_file(cls, pro_file_path: Path) -> 'KiCadProject':
        """
        Create a KiCadProject from a .kicad_pro file path.

        Args:
            pro_file_path: Path to the KiCad project file (*.kicad_pro)

        Returns:
            A KiCadProject instance with validated file paths

        Raises:
            FileNotFoundError: If any required files are missing
        """
        pro_file_path = Path(pro_file_path)

        if not pro_file_path.exists():
            raise FileNotFoundError(f"Project file not found: {pro_file_path}")

        base_name = pro_file_path.stem

        pcb_file_path = pro_file_path.parent / f"{base_name}.kicad_pcb"
        if not pcb_file_path.exists():
            raise FileNotFoundError(f"PCB file not found: {pcb_file_path}")

        sch_file_path = pro_file_path.parent / f"{base_name}.kicad_sch"
        if not sch_file_path.exists():
            raise FileNotFoundError(f"Schematic file not found: {sch_file_path}")

        return cls(
            pro_path=pro_file_path,
            pcb_path=pcb_file_path,
            sch_path=sch_file_path
        )


def extract_via_specs_from_pcb(board: pcbnew.BOARD) -> list[ViaSpec]:
    """
    Extract via specifications from a KiCad PCB.

    Args:
        board: The KiCad board object

    Returns:
        A list of ViaSpec objects containing information about vias in the PCB
    """
    via_specs = []

    # Get the tracks (which include vias)
    for track in board.GetTracks():
        # Check if the track is a via
        if track.Type() != pcbnew.PCB_VIA_T:
            continue

        # Cast to a via object
        via = track.Cast()

        # Get the via drill diameter (convert from nm to mm)
        drill_diameter = nm_to_mm(via.GetDrillValue())

        # Get the layers this via connects
        layer_names = []
        layer_set = via.GetLayerSet()

        for layer_id in copper_layers(board):
            if not layer_set.Contains(layer_id):
                continue
            layer_names.append(board.GetLayerName(layer_id))

        # Get the via's position (convert from KiCad internal units - nanometers to mm)
        pos_x = nm_to_mm(via.GetPosition().x)
        pos_y = nm_to_mm(via.GetPosition().y)
        via_point = shapely.geometry.Point(pos_x, pos_y)

        # Create a ViaSpec object
        via_spec = ViaSpec(
            point=via_point,
            drill_diameter=drill_diameter,
            layer_names=layer_names
        )

        via_specs.append(via_spec)

    return via_specs


def extract_tht_pad_specs_from_pcb(board: pcbnew.BOARD) -> list[ViaSpec]:
    """
    Extract through-hole pad specifications from a KiCad PCB.

    Args:
        board: The KiCad board object

    Returns:
        A list of ViaSpec objects representing through-hole pads in the PCB
    """
    tht_specs = []

    # Walk through all footprints on the PCB
    for footprint in board.GetFootprints():
        # For each footprint, examine all pads
        for pad in footprint.Pads():
            # Check if the pad is through-hole type
            if pad.GetAttribute() != pcbnew.PAD_ATTRIB_PTH:
                continue
            # Get the pad position and convert from nm to mm
            pos_x = nm_to_mm(pad.GetPosition().x)
            pos_y = nm_to_mm(pad.GetPosition().y)
            pad_point = shapely.geometry.Point(pos_x, pos_y)

            # Get the drill diameter
            # For oval/slot drills, use average of width and height as an approximation
            drill_diameter = nm_to_mm((pad.GetDrillSize().x + pad.GetDrillSize().y) / 2)

            # Determine which layers this pad connects
            layer_names = []
            layer_set = pad.GetLayerSet()

            for layer_id in copper_layers(board):
                if not layer_set.Contains(layer_id):
                    continue
                layer_names.append(board.GetLayerName(layer_id))

            endpoint = Endpoint(
                designator=footprint.GetReference(),
                pad=pad.GetName()
            )

            # Create a ViaSpec object for this through-hole pad
            tht_spec = ViaSpec(
                point=pad_point,
                drill_diameter=drill_diameter,
                layer_names=layer_names,
                endpoint=endpoint
            )

            tht_specs.append(tht_spec)

    return tht_specs


@dataclass(frozen=True)
class Directives:
    """
    Accumulates different directive types that can be present in the schematic.
    """
    lumped_specs: list[BaseLumpedSpec]


@dataclass
class SchemaInstance:
    """
    Represents a schematic instance in the hierarchy.
    """
    file_path: pathlib.Path
    sheet_name: str
    parsed_sexp: Any
    child_instances: list['SchemaInstance'] = field(default_factory=list)


def parse_endpoint(token: str) -> Endpoint:
    """
    Parse an endpoint in the format DESIGNATOR.PAD.
    For example, "R1.1" will become Endpoint(designator="R1", pad="1").
    """
    parts = token.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid endpoint format: {token}")
    return Endpoint(designator=parts[0], pad=parts[1])


def process_directives(directives: list[Directive]) -> Directives:
    directive_name_to_spec_type = {
        "VOLTAGE": VoltageSourceSpec,
        "CURRENT": CurrentSourceSpec,
        "RESISTANCE": ResistorSpec,
        "REGULATOR": RegulatorSpec
    }
    lumped_specs = []

    for directive in directives:
        if directive.name not in directive_name_to_spec_type:
            warnings.warn(f"Unknown directive: {directive.name}")
            continue
        lumped_spec = directive_name_to_spec_type[directive.name].from_directive(directive)
        lumped_specs.append(lumped_spec)

    return Directives(lumped_specs=lumped_specs)


def build_schema_hierarchy(sch_file_path: pathlib.Path,
                           sheet_name: str = "Root") -> SchemaInstance:
    """
    Build the complete schematic hierarchy starting from a root schematic.
    """
    # Resolve the path to handle any relative references consistently
    sch_file_path = sch_file_path.resolve()

    # Parse the schematic file once
    with open(sch_file_path, "r") as f:
        import sexpdata
        parsed_sexp = sexpdata.load(f)

    # Create the schema instance
    schema_instance = SchemaInstance(
        file_path=sch_file_path,
        sheet_name=sheet_name,
        parsed_sexp=parsed_sexp,
        child_instances=[]
    )

    # Find sheet elements in the parsed data
    def find_sheet_elements(sexp_data):
        """Recursively find all (sheet ...) elements in the sexp tree."""
        if not isinstance(sexp_data, list):
            return []

        ret = []

        if len(sexp_data) > 0 and sexp_data[0] == sexpdata.Symbol("sheet"):
            ret.append(sexp_data)

        for item in sexp_data:
            ret.extend(find_sheet_elements(item))

        return ret

    def extract_sheet_properties(sheet_element):
        """Extract Sheetname and Sheetfile properties from a sheet element."""
        sheetname = None
        sheetfile = None

        for item in sheet_element:
            if (isinstance(item, list) and len(item) >= 3 and
                    item[0] == sexpdata.Symbol("property")):
                if item[1] == "Sheetname":
                    sheetname = item[2]
                elif item[1] == "Sheetfile":
                    sheetfile = item[2]

        if sheetname is None:
            log.warning(f"Sheetname not found in {sheet_element}")

        return sheetname, sheetfile

    # Process all sheet elements
    sheet_elements = find_sheet_elements(parsed_sexp)

    for sheet_element in sheet_elements:
        sheetname, sheetfile = extract_sheet_properties(sheet_element)

        if not sheetfile:
            log.warning(f"Sheetfile not found in {sheet_element}, skipping this child")
            continue

        # Resolve relative path relative to the parent schematic directory
        nested_sch_path = sch_file_path.parent / sheetfile

        if not nested_sch_path.exists():
            log.warning(f"Referenced schematic file not at {nested_sch_path}, skipping this child")
            continue

        child_instance = build_schema_hierarchy(
            nested_sch_path,
            sheetname or "Unnamed"
        )
        schema_instance.child_instances.append(child_instance)

    return schema_instance


def flatten_schema_hierarchy(schema_instance: SchemaInstance) -> list[SchemaInstance]:
    """
    Flatten a schema hierarchy into a list of all instances.

    Args:
        schema_instance: Root schema instance

    Returns:
        List of all schema instances in the hierarchy (root first, then children)
    """
    result = [schema_instance]

    for child_instance in schema_instance.child_instances:
        result.extend(flatten_schema_hierarchy(child_instance))

    return result


def extract_directives_from_schema(instance: SchemaInstance) -> list[Directive]:
    import sexpdata

    def find_text_elements(sexp_data):
        """Recursively find all (text ...) elements in the sexp tree."""
        if not isinstance(sexp_data, list):
            return []

        ret = []

        if len(sexp_data) > 0 and sexp_data[0] == sexpdata.Symbol("text"):
            ret.append(sexp_data)

        for item in sexp_data:
            ret.extend(find_text_elements(item))

        return ret

    def extract_content_from_text_element(text_element):
        """Extract text content from a text element."""
        assert isinstance(text_element, list)
        assert text_element[0] == sexpdata.Symbol("text")
        # This should probably always be the second element
        return text_element[1]

    all_texts = [
        extract_content_from_text_element(text_element)
        for text_element in find_text_elements(instance.parsed_sexp)
    ]

    return [
        Directive.parse(text)
        for text in all_texts
        if text.startswith("!padne")
    ]


def extract_directives_from_hierarchy(schema_instance: SchemaInstance) -> list[Directive]:
    """
    Extract all directives from a schematic hierarchy.

    Args:
        schema_instance: Root of the schematic hierarchy

    Returns:
        List of all directives found in the hierarchy (deduplicated by file path)
    """
    # Flatten the hierarchy to get all instances
    all_instances = flatten_schema_hierarchy(schema_instance)

    # Track processed file paths to avoid duplicates
    processed_files: set[pathlib.Path] = set()
    directives = []

    # Iterate through all instances and extract directives from unique files
    for instance in all_instances:
        # Skip if this file has already been processed
        if instance.file_path in processed_files:
            warnings.warn(f"Schematic files with multiple instances are not supported, loaded only one instance of {instance.file_path}, skipping the rest")
            continue

        # Mark this file as processed
        processed_files.add(instance.file_path)

        # Extract directives from this instance (if it has content)
        if instance.parsed_sexp is None:
            # This should not happen I think?
            log.error(f"Parsed S-expression is None for instance: {instance.file_path}")
            continue

        instance_directives = extract_directives_from_schema(instance)

        directives.extend(instance_directives)

    return directives


@dataclass(frozen=True)
class PlottedGerberLayer:
    name: str
    layer_id: int
    geometry: shapely.geometry.MultiPolygon


def render_gerbers_from_kicad(board: pcbnew.BOARD) -> list[PlottedGerberLayer]:
    """
    Generate Gerber files from a KiCad PCB file and convert them to PlottedGerberLayer objects.

    Args:
        pcb_file_path: Path to the KiCad PCB file

    Returns:
        List of PlottedGerberLayer objects containing layer geometries
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Plot gerbers and get paths to generated files
        gerber_layers = plot_board_to_gerbers(board, Path(tmpdir))

        # Extract geometry from gerber files
        return extract_layers_from_gerbers(board, gerber_layers)


def plot_board_to_gerbers(board, output_dir: Path) -> dict[int, Path]:
    """
    Plot copper layers of a KiCad board to Gerber files.

    Args:
        board: KiCad board object
        output_dir: Directory where Gerber files will be saved

    Returns:
        Dictionary mapping layer IDs to paths of generated Gerber files
    """
    # Create plot controller and options
    plot_controller = pcbnew.PLOT_CONTROLLER(board)
    plot_options = plot_controller.GetPlotOptions()

    # Configure plot options for Gerber output
    plot_options.SetOutputDirectory(str(output_dir))
    plot_options.SetFormat(pcbnew.PLOT_FORMAT_GERBER)
    plot_options.SetUseGerberAttributes(True)
    plot_options.SetCreateGerberJobFile(False)
    #plot_options.SetExcludeEdgeLayer(False) # TODO: Figure this out
    plot_options.SetUseAuxOrigin(True)
    # TODO: This is a rather important choice - for now, we make no drill
    # shapes and later after we get via sim online, we need to include the drill shape
    # and handle the edge of each hole correctly
    plot_options.SetDrillMarksType(pcbnew.DRILL_MARKS_FULL_DRILL_SHAPE)

    # Set up layer list to plot
    gerber_layers = {}

    # Plot each copper layer
    for layer_id in copper_layers(board):
        # Get layer name first
        layer_name = board.GetLayerName(layer_id)

        # Open plot file
        plot_controller.SetLayer(layer_id)
        plot_controller.OpenPlotfile(layer_name, pcbnew.PLOT_FORMAT_GERBER, "")

        # Plot the layer
        assert plot_controller.PlotLayer(), f"Failed to plot layer {layer_name}"

        gerber_path = Path(plot_controller.GetPlotFileName())

        assert gerber_path.exists(), f"Gerber file {gerber_path} does not exist"

        gerber_layers[layer_id] = gerber_path

    # Close the plot
    plot_controller.ClosePlot()

    return gerber_layers


def render_with_shapely(gerber_data: pygerber.gerber.api.GerberFile
                        ) -> shapely.geometry.MultiPolygon:
    # We have to call all of this manually, since we need to manually configure the
    # amount of segments in our arcs
    rvmc = gerber_data._get_rvmc()

    def angle_length_to_segment_count(angle_length: float) -> int:
        return int(abs(angle_length) * 0.4 + 10)

    result = pygerber.vm.render(
        rvmc,
        backend="shapely",
        angle_length_to_segment_count=angle_length_to_segment_count
    )
    return result.shape


def extract_layers_from_gerbers(board,
                                gerber_layers: dict[int, Path]
                                ) -> list[PlottedGerberLayer]:
    """
    Extract geometry from Gerber files and create PlottedGerberLayer objects.

    Args:
        board: KiCad board object (for layer names)
        gerber_layers: Dictionary mapping layer IDs to paths of Gerber files

    Returns:
        List of PlottedGerberLayer objects
    """
    plotted_layers = []

    for layer_id, gerber_path in gerber_layers.items():
        # Get layer name from the board
        layer_name = board.GetLayerName(layer_id)

        # Load gerber file and extract geometry
        gerber_data = pygerber.gerber.api.GerberFile.from_file(gerber_path)
        try:
            geometry = render_with_shapely(gerber_data)
        except AssertionError:
            # This is a bug in pygerber, which gets triggered if the
            # gerber file is empty. We should fix this in pygerber ideally
            # TODO: Figure out if there is at least a way to check if the
            # gerber file is empty before we try to render it
            continue

        # For reasons to be determined, the geometry generated like this has
        # a flipped y axis. Flip it back.
        geometry = shapely.affinity.scale(geometry, 1.0, -1.0, origin=(0, 0))

        # Simplify the geometry to remove almost-duplicate points
        # This is unfortunately a "bug" in pygerber, where drawing
        # a circle is implemented by drawing an arbitrary degree arc,
        # which sometimes results to the "starting" and "ending" points
        # not being exactly the same such as
        # (-1.0, 0.0) vs  (-1.0, 1.2246467991473532e-16)
        # Again, it would be nice to fix this in pygerber, but that
        # is a task for another day...
        geometry = geometry.simplify(tolerance=1e-4, preserve_topology=True)

        # If the layer has only a single connected component, convert it to a MultiPolygon
        if geometry.geom_type == "Polygon":
            geometry = shapely.geometry.MultiPolygon([geometry])

        # Create a PlottedGerberLayer object
        plotted_layer = PlottedGerberLayer(
            name=layer_name,
            layer_id=layer_id,
            geometry=geometry
        )

        plotted_layers.append(plotted_layer)

    return plotted_layers


def process_via_spec(via_spec: ViaSpec,
                     layer_dict: dict[str, problem.Layer],
                     stackup: Stackup) -> list[problem.Network]:
    # In theory, they should already be in physical order, but we reorder
    # them based on the Stackup just in case this ever changes

    via_layers_in_order = sorted(
        via_spec.layer_names,
        key=stackup.index_by_name
    )

    resistor_stack = []

    # Get boundary coordinates (excluding duplicate last point)
    boundary_coords = list(via_spec.shape.exterior.coords)[:-1]
    num_boundary_points = len(boundary_coords)

    # Find maximum plating thickness from all copper layers in the via spec
    involved_copper_layers = [
        stackup.items[stackup.index_by_name(layer_name)]
        for layer_name in via_spec.layer_names
    ]
    plating_thickness = max(
        layer.thickness for layer in involved_copper_layers
        if layer.conductivity is not None
    )

    # Use conductivity from copper layers (should be same for all copper)
    conductivity = next(
        layer.conductivity for layer in involved_copper_layers
        if layer.conductivity is not None
    )

    for i in range(len(via_layers_in_order) - 1):
        layer_a_name = via_layers_in_order[i]
        layer_b_name = via_layers_in_order[i + 1]
        layer_a = layer_dict[layer_a_name]
        layer_b = layer_dict[layer_b_name]

        j_a = stackup.index_by_name(layer_a_name)
        j_b = stackup.index_by_name(layer_b_name)

        assert j_a < j_b, f"Layer {layer_a_name} should be before {layer_b_name} in the stackup"

        segment_length = sum(
            stackup.items[j].thickness
            for j in range(j_a + 1, j_b + 1)
        )

        # Total via resistance for this segment
        total_resistance = via_spec.compute_resistance(segment_length, plating_thickness, conductivity)

        # Resistance per boundary resistor (parallel resistors)
        # Each resistor carries 1/N of the current, so needs N times the resistance
        distributed_resistance = total_resistance * num_boundary_points

        # Create connections and resistors for each boundary point
        connections = []
        elements = []

        for x, y in boundary_coords:
            point = shapely.geometry.Point(x, y)
            conn_a = problem.Connection(layer=layer_a, point=point)
            conn_b = problem.Connection(layer=layer_b, point=point)

            via_resistor = problem.Resistor(
                a=conn_a.node_id,
                b=conn_b.node_id,
                resistance=distributed_resistance
            )

            connections.extend([conn_a, conn_b])
            elements.append(via_resistor)

        network = problem.Network(
            connections=connections,
            elements=elements
        )

        resistor_stack.append(network)

    return resistor_stack


def punch_via_holes(plotted_layers: list[PlottedGerberLayer],
                    via_specs: list[ViaSpec]) -> list[PlottedGerberLayer]:

    # For efficiency, we furst make union of all holes
    # and then punch them into the layer in one go
    holes_by_layer = collections.defaultdict(list)
    for via_spec in via_specs:
        for layer_name in via_spec.layer_names:
            holes_by_layer[layer_name].append(via_spec.shape)

    union_holes_by_layer = {
        layer_name: shapely.union_all(holes)
        for layer_name, holes in holes_by_layer.items()
    }

    punched_layers = []
    for plotted_layer in plotted_layers:
        if plotted_layer.name in union_holes_by_layer:
            # Punch holes in the layer geometry
            punched_geometry = plotted_layer.geometry.difference(
                union_holes_by_layer[plotted_layer.name]
            )
            if punched_geometry.geom_type == "Polygon":
                punched_geometry = shapely.geometry.MultiPolygon([punched_geometry])
            # There are cases where the difference may result in
            # a GeometryCollection or empty geometry.
            # I think we are careful enough that it should not happen,
            # but check just in case
            assert punched_geometry.geom_type == "MultiPolygon", \
                f"Expected MultiPolygon after punching holes, got {punched_geometry.geom_type}"
        else:
            # No vias present, keep the original geometry
            punched_geometry = plotted_layer.geometry

        # Create a new PlottedGerberLayer with the punched geometry
        punched_layer = PlottedGerberLayer(
            name=plotted_layer.name,
            layer_id=plotted_layer.layer_id,
            geometry=punched_geometry
        )
        punched_layers.append(punched_layer)

    return punched_layers


def verify_stackup_contains_all_layers(stackup: Stackup,
                                       plotted_layers: list[PlottedGerberLayer]) -> bool:
    """
    Verify that all plotted layers are contained within the stackup.

    Args:
        stackup: Stackup object containing layers
        plotted_layers: List of PlottedGerberLayer objects

    Raises:
        ValueError: If any plotted layer is not found in the stackup
    """
    for pl in plotted_layers:
        if not any(pl.name == stackup_item.name for stackup_item in stackup.items):
            return False
    return True


def construct_layer_dict(plotted_layers: list[PlottedGerberLayer],
                         stackup: Stackup) -> dict[str, problem.Layer]:
    """
    Construct a dictionary mapping layer names to Layer objects.

    Args:
        plotted_layers: List of PlottedGerberLayer objects

    Returns:
        Dictionary mapping layer names to Layer objects
    """
    # TODO: Rename this function...
    layer_dict = {}
    for plotted_layer in plotted_layers:
        stackup_layer = next(
            (item for item in stackup.items if item.name == plotted_layer.name)
        )
        layer = problem.Layer(
            shape=plotted_layer.geometry,
            name=plotted_layer.name,
            conductance=stackup_layer.conductance
        )
        layer_dict[plotted_layer.name] = layer
    return layer_dict


def load_kicad_project(pro_file_path: pathlib.Path) -> problem.Problem:
    """
    Load a KiCad project and create a Problem object for PDN simulation.

    Args:
        project: Either a path to the KiCad project file (*.kicad_pro) or a KiCadProject instance

    Returns:
        A Problem object containing layers and lumped elements

    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If the project contains invalid data
    """
    project = KiCadProject.from_pro_file(pro_file_path)

    # Load metadata and geometry from the PCB file
    log.info("Plotting layers to gerbers")
    board = pcbnew.LoadBoard(str(project.pcb_path))
    stackup = extract_stackup_from_kicad_pcb(board)
    plotted_layers = render_gerbers_from_kicad(board)
    # Build schematic hierarchy first, then extract directives
    schema_hierarchy = build_schema_hierarchy(project.sch_path)
    directives = process_directives(extract_directives_from_hierarchy(schema_hierarchy))

    if not verify_stackup_contains_all_layers(stackup, plotted_layers):
        raise ValueError("Stackup does not contain all plotted layers")

    pad_index = PadIndex()
    pad_index.load_smd_pads(board)

    # Convert Spec objects to Network objects
    networks = []

    log.info("Processing vias and through hole pads")
    via_specs = extract_via_specs_from_pcb(board) + extract_tht_pad_specs_from_pcb(board)
    pad_index.insert_via_specs(via_specs)
    plotted_layers = punch_via_holes(plotted_layers, via_specs)
    # Note that we have to create the layer dict _after_ punching the holes,
    # since otherwise it would contain the original objects!
    layer_dict = construct_layer_dict(plotted_layers, stackup)
    for via_spec in via_specs:
        networks.extend(process_via_spec(via_spec, layer_dict, stackup))

    log.info("Creating networks from specifications")
    for lumped_spec in directives.lumped_specs:
        network = lumped_spec.construct(pad_index, layer_dict)
        networks.append(network)

    # Get all layers as a list
    # TODO: Sort them using the stackup
    layer_names_in_order = list(layer_dict.keys())
    layer_names_in_order.sort(key=lambda name: stackup.index_by_name(name))

    layers = [layer_dict[name] for name in layer_names_in_order]

    # Return the Problem object
    return problem.Problem(layers=layers, networks=networks)
