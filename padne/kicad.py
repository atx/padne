
import warnings
# This is to suppress pcbnew deprecation warning. Unfortunately the RPC API
# is not yet cooked enough for us
warnings.simplefilter("ignore", DeprecationWarning)

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

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Iterator, List

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
class LumpedSpec:
    """
    Specifies a single lumped element directly connected to a pad on the PCB.
    """

    class Type(enum.Enum):
        VOLTAGE = "VOLTAGE"
        CURRENT = "CURRENT"
        RESISTANCE = "RESISTANCE"

    endpoint_a: Endpoint
    endpoint_b: Endpoint

    # Use the unified lumped type from the problem module.
    type: "LumpedSpec.Type"
    value: float


@dataclass(frozen=True)
class ConsumerSpec:
    """
    Specifies a star-topology consumer, connected to multiple in/out pads.
    """
    endpoints_f: list[Endpoint]
    endpoints_t: list[Endpoint]
    current: float
    resistance: float


@dataclass(frozen=True)
class RegulatorSpec:
    """
    Specifies a voltage regulator, connected to multiple pins for its terminals.
    """
    voltage: float
    gain: float
    resistance: float
    endpoints_p: List[Endpoint]  # Positive voltage output/sense pins
    endpoints_n: List[Endpoint]  # Negative voltage output/sense pins
    endpoints_f: List[Endpoint]  # Input power "from" pins
    endpoints_t: List[Endpoint]  # Input power "to" pins


@dataclass(frozen=True)
class ViaSpec:
    """
    Specifies a via in the PCB.
    """
    point: shapely.geometry.Point
    drill_diameter: float
    layer_names: list[str]

    def compute_resistance(self, length: float) -> float:
        # TODO: This is very temporary solution. Will ultimately need to take
        # into account layer plating thickness etc
        # Resistance of a 1.6mm long via with 1mm diameter
        ref_resistance = 0.00027
        ref_area = math.pi * (0.5) ** 2
        ref_length = 1.6

        area = math.pi * (self.drill_diameter / 2) ** 2
        return ref_resistance * (area / ref_area) * length / ref_length


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
            
            # Create a ViaSpec object for this through-hole pad
            tht_spec = ViaSpec(
                point=pad_point,
                drill_diameter=drill_diameter,
                layer_names=layer_names
            )
            
            tht_specs.append(tht_spec)
    
    return tht_specs


@dataclass(frozen=True)
class Directives:
    """
    Accumulates different directive types that can be present in the schematic.
    """
    # Surface resistivity
    # TODO: Add default value for 1oz copper
    lumpeds: list[LumpedSpec]
    consumers: list[ConsumerSpec]
    regulators: list[RegulatorSpec]


def parse_endpoint(token: str) -> Endpoint:
    """
    Parse an endpoint in the format DESIGNATOR.PAD.
    For example, "R1.1" will become Endpoint(designator="R1", pad="1").
    """
    parts = token.split(".")
    if len(parts) != 2:
        raise ValueError(f"Invalid endpoint format: {token}")
    return Endpoint(designator=parts[0], pad=parts[1])


def parse_lumped_spec_directive(directive: Directive) -> LumpedSpec:
    try:
        type_enum = LumpedSpec.Type(directive.name)
    except ValueError:
        raise ValueError(f"Unknown directive type: {directive.name}")

    # TODO: I do not quite like this point here
    # realistically, we should have something that has less friction when
    # translating from an abstract directive into a lumped element
    # into a problem element
    # This is leaking abstraction since the parameter names here
    # are technically not the same as the ones in the problem elements
    # however, we are eventually going to want to support having, say,
    # parameter variations etc

    match type_enum:
        case LumpedSpec.Type.VOLTAGE:
            value_name = "v"
            a_name = "p"
            b_name = "n"
        case LumpedSpec.Type.CURRENT:
            value_name = "i"
            a_name = "f"
            b_name = "t"
        case LumpedSpec.Type.RESISTANCE:
            value_name = "r"
            a_name = "a"
            b_name = "b"
        case _:
            raise ValueError(f"Unknown directive type: {directive.name}")

    # TODO: Check that the unit string is valid
    value = units.Value.parse(directive.params[value_name])
    ep_a = parse_endpoint(directive.params[a_name])
    ep_b = parse_endpoint(directive.params[b_name])

    return LumpedSpec(
        endpoint_a=ep_a,
        endpoint_b=ep_b,
        type=type_enum,
        value=value.value
    )


def parse_consumer_spec_directive(directive: Directive) -> ConsumerSpec:
    resistance = units.Value.parse(directive.params.get("r", "0.01"))
    current = units.Value.parse(directive.params["i"])
    endpoints_f = [
        parse_endpoint(ep) for ep in directive.params.get("f", "").split(",")
    ]
    endpoints_t = [
        parse_endpoint(ep) for ep in directive.params.get("t", "").split(",")
    ]
    return ConsumerSpec(
        resistance=resistance.value,
        current=current.value,
        endpoints_f=endpoints_f,
        endpoints_t=endpoints_t
    )


def _parse_endpoints_param(param_str: Optional[str]) -> list[Endpoint]:
    """Helper to parse a comma-separated string of endpoints."""
    if not param_str:
        return []
    return [
        parse_endpoint(ep_str.strip())
        for ep_str in param_str.split(',')
        if ep_str.strip()
    ]


def parse_regulator_spec_directive(directive: Directive) -> RegulatorSpec:
    """
    Parse a REGULATOR directive into a RegulatorSpec object.
    Example: !padne REGULATOR v=5V gain=1.0 r=0.001 p=U1.1,U1.2 n=U1.3 f=U1.4 t=U1.5
    """
    voltage = units.Value.parse(directive.params["v"]).value
    
    gain = float(directive.params.get("gain", "1.0"))
    resistance = units.Value.parse(directive.params.get("r", "0.001")).value

    endpoints_p = _parse_endpoints_param(directive.params.get("p"))
    endpoints_n = _parse_endpoints_param(directive.params.get("n"))
    endpoints_f = _parse_endpoints_param(directive.params.get("f"))
    endpoints_t = _parse_endpoints_param(directive.params.get("t"))

    # Basic validation: at least one P and N pin should be defined for output.
    # And at least one F and T pin for input.
    # This is a soft check; the solver might handle disconnected regulators,
    # but it's unusual for a user to define them this way.
    if not endpoints_p:
        warnings.warn(f"REGULATOR directive has no 'p' (positive output) pins: {directive.params}")
    if not endpoints_n:
        warnings.warn(f"REGULATOR directive has no 'n' (negative output) pins: {directive.params}")
    if not endpoints_f:
        warnings.warn(f"REGULATOR directive has no 'f' (power input 'from') pins: {directive.params}")
    if not endpoints_t:
        warnings.warn(f"REGULATOR directive has no 't' (power input 'to') pins: {directive.params}")

    return RegulatorSpec(
        voltage=voltage,
        gain=gain,
        resistance=resistance,
        endpoints_p=endpoints_p,
        endpoints_n=endpoints_n,
        endpoints_f=endpoints_f,
        endpoints_t=endpoints_t
    )


def process_directives(directives: list[Directive]) -> Directives:
    lumpeds = []
    consumers = []
    regulators = []

    for directive in directives:
        match directive.name:
            case "VOLTAGE" | "CURRENT" | "RESISTANCE":
                lumped = parse_lumped_spec_directive(directive)
                lumpeds.append(lumped)
            case "CONSUMER":
                consumer = parse_consumer_spec_directive(directive)
                consumers.append(consumer)
            case "REGULATOR":
                regulator = parse_regulator_spec_directive(directive)
                regulators.append(regulator)
            case _:
                warnings.warn(f"Unknown directive: {directive.name}")

    return Directives(lumpeds=lumpeds, consumers=consumers, regulators=regulators)


def find_associated_files(pro_file_path: pathlib.Path) -> tuple[Path, Path]:
    """
    Given a KiCad project file, return the associated PCB and schematic file paths.
    
    Args:
        pro_file_path: The KiCad project file (*.kicad_pro)
        
    Returns:
        A tuple of (pcb_file_path, sch_file_path)
    """

    if not pro_file_path.exists():
        raise FileNotFoundError(f"Project file not found: {pro_file_path}")

    base_name = pro_file_path.stem

    pcb_file_path = pro_file_path.parent / f"{base_name}.kicad_pcb"
    if not pcb_file_path.exists():
        raise FileNotFoundError(f"PCB file not found: {pcb_file_path}")

    sch_file_path = pro_file_path.parent / f"{base_name}.kicad_sch"
    if not sch_file_path.exists():
        raise FileNotFoundError(f"Schematic file not found: {sch_file_path}")

    return pcb_file_path, sch_file_path


def extract_directives_from_eeschema(sch_file_path: pathlib.Path) -> list[Directive]:
    # First load the input schematic file
    with open(sch_file_path, "r") as f:
        import sexpdata
        sexpr = sexpdata.load(f)

    def find_text_elements(sexp_data):
        # Recurse to find all (text ...) elements in the sexp tree
        # This might be overkill, since I think they only live at the
        # top level
        if not isinstance(sexp_data, list):
            return []

        ret = []

        if len(sexp_data) > 0 and sexp_data[0] == sexpdata.Symbol("text"):
            ret.append(sexp_data)

        for item in sexp_data:
            ret.extend(find_text_elements(item))

        return ret

    def extract_content_from_text_element(text_element):
        assert isinstance(text_element, list)
        assert text_element[0] == sexpdata.Symbol("text")
        # This should probably always be the second element
        return text_element[1]

    all_texts = [
        extract_content_from_text_element(text_element)
        for text_element in find_text_elements(sexpr)
    ]

    directives = [
        Directive.parse(text)
        for text in all_texts
        if text.startswith("!padne")
    ]

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


def find_pad_and_footprint(board: pcbnew.BOARD,
                           designator: str,
                           pad: str
                           ) -> tuple[pcbnew.FOOTPRINT, pcbnew.PAD]:
    """
    Find a pad and its associated footprint on the PCB.
    """
    for footprint in board.GetFootprints():
        if footprint.GetReference() != designator:
            continue

        # Find the pad with the given number/name
        for pad_obj in footprint.Pads():
            if pad_obj.GetName() != pad:
                continue

            return footprint, pad_obj

        raise ValueError(f"Pad {pad} not found on component {designator}")
    raise ValueError(f"Component {designator} not found in the PCB")


def find_pad_location(board,
                      designator: str,
                      pad: str
                      ) -> tuple[str, shapely.geometry.Point]:
    """
    Find the physical location of a pad on the PCB.
    
    Args:
        board: KiCad board object
        designator: Component reference designator (e.g., "R1")
        pad: Pad number or name (e.g., "1")
        
    Returns:
        Tuple of (layer_name, point) where point is the pad's center
        
    Raises:
        ValueError: If the component or pad is not found
    """

    footprint, pad_obj = find_pad_and_footprint(board, designator, pad)

    # Get the pad's position (in KiCad internal units, nanometers)
    position = pad_obj.GetPosition()
    
    # Convert from KiCad internal units (nanometers) to mm
    x_mm = nm_to_mm(position.x)
    y_mm = nm_to_mm(position.y)
    point = shapely.geometry.Point(x_mm, y_mm)
    
    match pad_obj.GetAttribute():
        case pcbnew.PAD_ATTRIB_SMD:
            # For SMD pads, get the layer directly
            layer_id = pad_obj.GetLayer()

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
                point = shapely.geometry.Point(x_mm, y_mm)

            layer_name = board.GetLayerName(layer_id)
            return layer_name, point

        case pcbnew.PAD_ATTRIB_PTH:
            # For through-hole pads, we use the first layer in the layer
            # set as their location. In practice, this is not quite ideal,
            # but good enough for now...
            layer_set = pad_obj.GetLayerSet()
            # Find first copper layer in the layer set
            for layer_id in copper_layers(board):
                if not layer_set.Contains(layer_id):
                    continue
                return board.GetLayerName(layer_id), point
            
            raise ValueError(f"No copper layer found for through-hole pad {pad} on component {designator}")


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

        resistance = via_spec.compute_resistance(segment_length)

        conn_a = problem.Connection(layer=layer_a, point=via_spec.point)
        conn_b = problem.Connection(layer=layer_b, point=via_spec.point)
        via_resistor = problem.Resistor(
            a=conn_a.node_id,
            b=conn_b.node_id,
            resistance=resistance
        )
        # TODO: Eventually, we probably want to integrate the entire stack to
        # a single Network object
        network = problem.Network(
            connections=[conn_a, conn_b],
            elements=[via_resistor]
        )

        resistor_stack.append(network)

    return resistor_stack


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


def create_lumped_element_from_spec(board: pcbnew.BOARD,
                                    spec: LumpedSpec,
                                    layer_dict: dict[str, problem.Layer]
                                    ) -> problem.Network:
    a_layer_name, a_point = find_pad_location(
        board, spec.endpoint_a.designator, spec.endpoint_a.pad
    )
    b_layer_name, b_point = find_pad_location(
        board, spec.endpoint_b.designator, spec.endpoint_b.pad
    )

    a_layer = layer_dict[a_layer_name]
    b_layer = layer_dict[b_layer_name]

    conn_a = problem.Connection(layer=a_layer, point=a_point)
    conn_b = problem.Connection(layer=b_layer, point=b_point)

    # Create the specific lumped element subclass instance
    match spec.type:
        case LumpedSpec.Type.RESISTANCE:
            lumped_element = problem.Resistor(
                a=conn_a.node_id,
                b=conn_b.node_id,
                resistance=spec.value
            )
        case LumpedSpec.Type.VOLTAGE:
            # Assuming endpoint_a is positive (p) and endpoint_b is negative (n)
            lumped_element = problem.VoltageSource(
                p=conn_a.node_id,
                n=conn_b.node_id,
                voltage=spec.value
            )
        case LumpedSpec.Type.CURRENT:
            # Assuming current flows from endpoint_a (f) to endpoint_b (t)
            lumped_element = problem.CurrentSource(
                f=conn_a.node_id,
                t=conn_b.node_id,
                current=spec.value
            )
        case _:
            raise ValueError(f"Unhandled lumped element type: {spec.type}")

    return problem.Network(
        connections=[conn_a, conn_b],
        elements=[lumped_element]
    )


def create_consumer_from_spec(board: pcbnew.BOARD,
                              spec: ConsumerSpec,
                              layer_dict: dict[str, problem.Layer]) -> problem.Network:
    node_f = problem.NodeID()
    node_t = problem.NodeID()
    elements = []
    connections = []
    current_source = problem.CurrentSource(
        f=node_f,
        t=node_t,
        current=spec.current
    )
    elements.append(current_source)
    for ep in spec.endpoints_f:
        layer_name, point = find_pad_location(board, ep.designator, ep.pad)
        layer = layer_dict[layer_name]
        conn = problem.Connection(layer=layer, point=point)
        connections.append(conn)
        elements.append(problem.Resistor(
            a=node_f,
            b=conn.node_id,
            resistance=spec.resistance
        ))

    for ep in spec.endpoints_t:
        layer_name, point = find_pad_location(board, ep.designator, ep.pad)
        layer = layer_dict[layer_name]
        conn = problem.Connection(layer=layer, point=point)
        connections.append(conn)
        elements.append(problem.Resistor(
            a=node_t,
            b=conn.node_id,
            resistance=spec.resistance
        ))

    return problem.Network(
        elements=elements,
        connections=connections
    )


def create_regulator_from_spec(board: pcbnew.BOARD,
                               spec: RegulatorSpec,
                               layer_dict: dict[str, problem.Layer]) -> problem.Network:
    """
    Create a problem.Network object from a RegulatorSpec.
    This involves creating an ideal VoltageRegulator element and connecting its
    terminals to the specified PCB pads via small resistors.
    """
    elements: list[problem.BaseLumped] = []
    connections: list[problem.Connection] = []

    # Internal nodes for the ideal regulator
    node_vp_internal = problem.NodeID()  # Positive output voltage terminal
    node_vn_internal = problem.NodeID()  # Negative output voltage terminal (reference)
    node_f_internal = problem.NodeID()   # Power input "from" terminal
    node_t_internal = problem.NodeID()   # Power input "to" terminal (return)

    # Create the ideal voltage regulator element
    regulator_element = problem.VoltageRegulator(
        v_p=node_vp_internal,
        v_n=node_vn_internal,
        s_f=node_f_internal,
        s_t=node_t_internal,
        voltage=spec.voltage,
        gain=spec.gain
    )
    elements.append(regulator_element)

    # Connect positive output/sense pins (p)
    for ep in spec.endpoints_p:
        layer_name, point = find_pad_location(board, ep.designator, ep.pad)
        layer = layer_dict[layer_name]
        conn = problem.Connection(layer=layer, point=point)
        connections.append(conn)
        elements.append(problem.Resistor(
            a=node_vp_internal,
            b=conn.node_id,
            resistance=spec.resistance
        ))

    # Connect negative output/sense pins (n)
    for ep in spec.endpoints_n:
        layer_name, point = find_pad_location(board, ep.designator, ep.pad)
        layer = layer_dict[layer_name]
        conn = problem.Connection(layer=layer, point=point)
        connections.append(conn)
        elements.append(problem.Resistor(
            a=node_vn_internal,
            b=conn.node_id,
            resistance=spec.resistance
        ))

    # Connect power input "from" pins (f)
    for ep in spec.endpoints_f:
        layer_name, point = find_pad_location(board, ep.designator, ep.pad)
        layer = layer_dict[layer_name]
        conn = problem.Connection(layer=layer, point=point)
        connections.append(conn)
        elements.append(problem.Resistor(
            a=node_f_internal,
            b=conn.node_id,
            resistance=spec.resistance
        ))

    # Connect power input "to" pins (t)
    for ep in spec.endpoints_t:
        layer_name, point = find_pad_location(board, ep.designator, ep.pad)
        layer = layer_dict[layer_name]
        conn = problem.Connection(layer=layer, point=point)
        connections.append(conn)
        elements.append(problem.Resistor(
            a=node_t_internal,
            b=conn.node_id,
            resistance=spec.resistance
        ))
    
    return problem.Network(elements=elements, connections=connections)


def load_kicad_project(pro_file_path: pathlib.Path) -> problem.Problem:
    """
    Load a KiCad project and create a Problem object for PDN simulation.
    
    Args:
        pro_file_path: Path to the KiCad project file (*.kicad_pro)
        
    Returns:
        A Problem object containing layers and lumped elements
    
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If the project contains invalid data
    """
    # Find associated PCB and schematic files
    pcb_file_path, sch_file_path = find_associated_files(pro_file_path)
    
    # Load metadata and geometry from the PCB file
    log.info("Plotting layers to gerbers")
    board = pcbnew.LoadBoard(str(pcb_file_path))
    stackup = extract_stackup_from_kicad_pcb(board)
    plotted_layers = render_gerbers_from_kicad(board)
    directives = process_directives(extract_directives_from_eeschema(sch_file_path))
    
    if not verify_stackup_contains_all_layers(stackup, plotted_layers):
        raise ValueError("Stackup does not contain all plotted layers")
    
    layer_dict = construct_layer_dict(plotted_layers, stackup)
    
    # Convert Spec objects to Network objects
    log.info("Creating networks from specifications")
    networks = []
    for lumped_spec in directives.lumpeds:
        network = create_lumped_element_from_spec(board, lumped_spec, layer_dict)
        networks.append(network)

    for consumer_spec in directives.consumers:
        network = create_consumer_from_spec(board, consumer_spec, layer_dict)
        networks.append(network)

    for regulator_spec in directives.regulators:
        network = create_regulator_from_spec(board, regulator_spec, layer_dict)
        networks.append(network)

    log.info("Processing vias and through hole pads")
    via_specs = extract_via_specs_from_pcb(board) + extract_tht_pad_specs_from_pcb(board)
    for via_spec in via_specs:
        networks.extend(process_via_spec(via_spec, layer_dict, stackup))

    # Get all layers as a list
    # TODO: Sort them using the stackup
    layer_names_in_order = list(layer_dict.keys())
    layer_names_in_order.sort(key=lambda name: stackup.index_by_name(name))

    layers = [layer_dict[name] for name in layer_names_in_order]
    
    # Return the Problem object
    return problem.Problem(layers=layers, networks=networks)
