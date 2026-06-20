#!/usr/bin/env python3
"""
Validation of the solver against physically measured test boards.

A "test set" is a real PCB (under tests/kicad/) that has been built and probed
on the bench. We store the bench readings inline below and check that the solver
reproduces them. Two kinds of readings:

  * CalTrace  - a rung of the resistance ladder (200/300/400um traces). The
                fitted sheet conductance is applied to the copper before solving
                (calibrate), and the rung reading is itself confirmed like any
                other measurement.
  * Measurement - a plain point-to-point voltage reading between two pads.

The module is importable (it exposes solve_test_set / max_abs_error for
benchmarks/benchmarks.py), runnable under pytest, and executable for
investigation:

    python3 tests/test_sets.py calibrate test_set_1
    python3 tests/test_sets.py report    test_set_1
"""

import argparse
import functools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pcbnew
import pytest

from typing import Optional

from padne import kicad, mesh, problem, solver

KICAD_DIR = Path(__file__).parent / "kicad"


@dataclass(frozen=True)
class Measurement:
    """A single bench reading: V(p_ref) - V(n_ref) under the board's sources."""
    p_ref: str            # e.g. "TP3" or "J4.2" (designator or designator.pad)
    n_ref: str
    measured_v: float
    abs_tol: Optional[float] = None
    rel_tol: Optional[float] = 0.4  # Intentionally relaxed
    description: str = ""


@dataclass(frozen=True)
class CalTrace:
    """
    One rung of the resistance ladder. The embedded Measurement is the across-
    rung reading; nominal_width/length describe the drawn copper. injected_current
    converts the reading into a resistance (R = measured_v / injected_current).
    """
    measurement: Measurement
    nominal_width_mm: float
    length_mm: float
    injected_current: float = 1.0

    @property
    def measured_ohms(self) -> float:
        return self.measurement.measured_v / self.injected_current


@dataclass(frozen=True)
class TestSet:
    __test__ = False  # not a pytest test class despite the Test* name

    project: str                       # subdir name under tests/kicad/
    cal_traces: list[CalTrace] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    copper_thickness_mm: float = 0.035

    @property
    def pro_path(self) -> Path:
        return KICAD_DIR / self.project / f"{self.project}.kicad_pro"

    def all_measurements(self) -> list[Measurement]:
        """Every reading the solver must reproduce, cal rungs included."""
        return list(self.measurements) + [ct.measurement for ct in self.cal_traces]


@dataclass(frozen=True)
class CalibrationResult:
    sheet_conductance: float   # S, equals conductivity * thickness (slope of fit)
    overetch_delta_mm: float   # total width lost vs nominal (w_eff = w_nom - this)
    implied_thickness_mm: float  # sheet_conductance / pure-copper conductivity
    rung_residuals: list[tuple[CalTrace, float]]  # (rung, measured_R - predicted_R)


@dataclass(frozen=True)
class ResultRow:
    measurement: Measurement
    predicted_v: float

    @property
    def abs_err(self) -> float:
        return abs(self.predicted_v - self.measurement.measured_v)

    @property
    def rel_err(self) -> float:
        denom = abs(self.measurement.measured_v)
        return self.abs_err / denom if denom > 0 else math.inf

    @property
    def ok(self) -> bool:
        m = self.measurement
        abs_ok = m.abs_tol is not None and self.abs_err <= m.abs_tol
        rel_ok = m.rel_tol is not None and self.rel_err <= m.rel_tol
        return abs_ok or rel_ok


# Shorthands
CT = CalTrace
M = Measurement
TEST_SETS: dict[str, TestSet] = {
    "test_set_1_aisler": TestSet(
        project="test_set_1",
        cal_traces=[
            CT(M("TP61", "TP60", measured_v=210e-3), nominal_width_mm=0.2, length_mm=70),
            CT(M("TP63", "TP62", measured_v=120e-3), nominal_width_mm=0.3, length_mm=70),
            CT(M("TP65", "TP64", measured_v=82.7e-3), nominal_width_mm=0.4, length_mm=70),
            CT(M("TP67", "TP66", measured_v=63.7e-3), nominal_width_mm=0.5, length_mm=70),
        ],
        measurements=[
            M("TP34", "TP33", measured_v=49.6e-3),
            M("TP36", "TP35", measured_v=85.3e-3),
            M("TP30", "TP29", measured_v=39.1e-3),
            M("TP32", "TP31", measured_v=77.6e-3),
            M("TP26", "TP25", measured_v=32.4e-3),
            M("TP28", "TP27", measured_v=57.3e-3),
            M("TP22", "TP21", measured_v=31.1e-3),
            M("TP24", "TP23", measured_v=58.9e-3),
            M("TP18", "TP17", measured_v=18.8e-3),
            M("TP20", "TP19", measured_v=12.2e-3),
            M("TP14", "TP13", measured_v=40.9e-3),
            M("TP16", "TP15", measured_v=38.8e-3),
            M("TP10", "TP9", measured_v=30.1e-3),
            M("TP12", "TP11", measured_v=61.7e-3),
            # TODO: Maybe some these measurements should be done at higher current
            M("TP4", "TP1", measured_v=13.0e-3),
            M("TP3", "TP1", measured_v=4.93e-3),
            M("TP8", "TP5", measured_v=9.70e-3),
            M("TP6", "TP5", measured_v=5.98e-3),
            M("TP38", "TP37", measured_v=22.5e-3),
            M("TP41", "TP37", measured_v=13.5e-3),
            M("TP40", "TP39", measured_v=51.9e-3),
            M("TP42", "TP39", measured_v=19.3e-3),
            M("TP44", "TP43", measured_v=6.18e-3),
            M("TP45", "TP44", measured_v=5.09e-3),
            # Resistor chains
            M("TP47", "TP46", measured_v=96.0e-3),
            M("TP59", "TP46", measured_v=54.5e-3),
            M("TP48", "TP49", measured_v=64.7e-3),
            M("TP58", "TP49", measured_v=36.2e-3),
            M("TP51", "TP52", measured_v=25.8e-3),
            M("TP50", "TP53", measured_v=64.4e-3),
            M("TP57", "TP53", measured_v=29.7e-3),
            M("TP54", "TP55", measured_v=100e-3),
            M("TP56", "TP55", measured_v=69.0e-3),
        ],
    ),
}


def _parse_ref(ref: str) -> tuple[str, str | None]:
    """"TP3" -> ("TP3", None); "J4.2" -> ("J4", "2")."""
    if "." in ref:
        designator, pad = ref.split(".", 1)
        return designator, pad
    return ref, None


def pad_xy(board: pcbnew.BOARD, ref: str) -> tuple[float, float, str]:
    """Resolve a pad reference to (x_mm, y_mm, layer_name) from the board."""
    designator, pad_name = _parse_ref(ref)
    footprint = board.FindFootprintByReference(designator)
    if footprint is None:
        raise ValueError(f"No footprint with reference {designator!r}")

    pads = list(footprint.Pads())
    if pad_name is None:
        if len(pads) != 1:
            raise ValueError(
                f"{designator!r} has {len(pads)} pads; specify one as {designator}.<pad>")
        pad = pads[0]
    else:
        pad = next((p for p in pads if p.GetName() == pad_name), None)
        if pad is None:
            raise ValueError(f"{designator!r} has no pad {pad_name!r}")

    pos = pad.GetPosition()
    return kicad.nm_to_mm(pos.x), kicad.nm_to_mm(pos.y), board.GetLayerName(pad.GetLayer())


def probe_voltage(sol: solver.Solution, board: pcbnew.BOARD, ref: str) -> float:
    """Potential at the mesh vertex nearest to the given pad reference."""
    x, y, layer_name = pad_xy(board, ref)
    layer_i = next(i for i, layer in enumerate(sol.problem.layers)
                   if layer.name == layer_name)

    target = mesh.Point(x, y)
    layer_sol = sol.layer_solutions[layer_i]

    best_dist = math.inf
    best_value = None
    for msh, values in zip(layer_sol.meshes, layer_sol.potentials):
        for vertex in msh.vertices:
            dist = vertex.p.distance(target)
            if dist < best_dist:
                best_dist = dist
                best_value = values[vertex]

    if best_value is None or best_dist > 1e-3:
        raise ValueError(
            f"No mesh vertex near {ref} ({x:.3f}, {y:.3f}) on {layer_name} "
            f"(closest dist {best_dist:.3e}mm)")
    return best_value


def voltage_diff(sol: solver.Solution, board: pcbnew.BOARD, m: Measurement) -> float:
    return probe_voltage(sol, board, m.p_ref) - probe_voltage(sol, board, m.n_ref)


def _calibrated_problem(ts: TestSet) -> problem.Problem:
    """
    Load the KiCad project and, when the set has a ladder, override every copper
    layer's conductance with the sheet conductance fitted from the bench rungs.
    This closes the loop: the solver runs with the board's measured conductance
    instead of the nominal value baked into the project.
    """
    prob = kicad.load_kicad_project(ts.pro_path)
    if len(ts.cal_traces) < 2:
        return prob
    cal = extract_calibration(ts)
    # Patch conductance in place. Connections reference these same Layer objects
    # by identity (the solver does prob.layers.index(conn.layer)), so mutating
    # them keeps everything wired. object.__setattr__ bypasses frozen, as the
    # codebase itself does in Layer/Network __post_init__.
    for layer in prob.layers:
        object.__setattr__(layer, "conductance", cal.sheet_conductance)
    return prob


def solve_test_set(ts: TestSet, mesher_config=None
                   ) -> tuple[solver.Solution, pcbnew.BOARD]:
    """Load the (calibrated) project, solve it, and return solution plus board."""
    prob = _calibrated_problem(ts)
    board = pcbnew.LoadBoard(str(KICAD_DIR / ts.project / f"{ts.project}.kicad_pcb"))
    sol = solver.solve(prob, mesher_config=mesher_config)
    return sol, board


def evaluate(ts: TestSet, sol: solver.Solution, board: pcbnew.BOARD) -> list[ResultRow]:
    return [ResultRow(m, voltage_diff(sol, board, m)) for m in ts.all_measurements()]


def extract_calibration(ts: TestSet) -> CalibrationResult:
    """
    Fit effective sheet conductance and overetch from the ladder rungs.

    Model: R = length / (G_sheet * w_eff), w_eff = w_nominal - overetch. Linear
    in nominal width: length/R = G_sheet * w_nom - G_sheet * overetch, so a
    least-squares fit of (length/R) against w_nom gives slope = G_sheet and
    overetch = -intercept / slope.
    """
    if len(ts.cal_traces) < 2:
        raise ValueError("Need at least two cal traces to fit conductance + overetch")

    widths = np.array([ct.nominal_width_mm for ct in ts.cal_traces])
    ys = np.array([ct.length_mm / ct.measured_ohms for ct in ts.cal_traces])

    slope, intercept = np.polyfit(widths, ys, 1)
    overetch = -intercept / slope

    residuals = []
    for ct in ts.cal_traces:
        w_eff = ct.nominal_width_mm - overetch
        predicted_r = ct.length_mm / (slope * w_eff)
        residuals.append((ct, ct.measured_ohms - predicted_r))

    return CalibrationResult(
        sheet_conductance=slope,
        overetch_delta_mm=overetch,
        implied_thickness_mm=slope / kicad.COPPER_CONDUCTIVITY,
        rung_residuals=residuals,
    )


@functools.lru_cache(maxsize=None)
def _solved(ts_name: str) -> tuple[solver.Solution, pcbnew.BOARD]:
    return solve_test_set(TEST_SETS[ts_name])


def _measurement_cases() -> list[tuple[str, Measurement]]:
    return [(name, m) for name, ts in TEST_SETS.items() for m in ts.all_measurements()]


def _fmt_tol(tol: Optional[float]) -> str:
    return "n/a" if tol is None else f"{tol:.3g}"


@pytest.mark.parametrize(
    "ts_name,measurement",
    _measurement_cases(),
    ids=[f"{n}:{m.p_ref}-{m.n_ref}" for n, m in _measurement_cases()],
)
def test_measurement(ts_name, measurement):
    sol, board = _solved(ts_name)
    row = ResultRow(measurement, voltage_diff(sol, board, measurement))
    assert row.ok, (
        f"{ts_name} {measurement.p_ref}-{measurement.n_ref}: "
        f"measured {measurement.measured_v:.6g}V, predicted {row.predicted_v:.6g}V "
        f"(abs {row.abs_err:.3g} > {_fmt_tol(measurement.abs_tol)}, "
        f"rel {row.rel_err:.3g} > {_fmt_tol(measurement.rel_tol)})")


def _cmd_calibrate(ts: TestSet) -> None:
    cal = extract_calibration(ts)
    thickness_dev = cal.implied_thickness_mm / ts.copper_thickness_mm - 1
    print(f"sheet conductance : {cal.sheet_conductance:.6g} S")
    print(f"thickness vs ref  : {thickness_dev:+.1%} "
          f"(implied {cal.implied_thickness_mm * 1000:.2f} um, "
          f"ref {ts.copper_thickness_mm * 1000:.1f} um)")
    print(f"overetch delta    : {cal.overetch_delta_mm * 1000:.2f} um")
    print()
    print(f"{'rung':<14}{'w_nom/mm':>10}{'R_meas/ohm':>14}{'resid/ohm':>14}")
    for ct, resid in cal.rung_residuals:
        ref = f"{ct.measurement.p_ref}-{ct.measurement.n_ref}"
        print(f"{ref:<14}{ct.nominal_width_mm:>10.3f}"
              f"{ct.measured_ohms:>14.6g}{resid:>14.3g}")


def _cmd_report(ts: TestSet) -> None:
    sol, board = solve_test_set(ts)
    rows = evaluate(ts, sol, board)
    if not rows:
        print("no measurements defined")
        return

    print(f"{'pair':<16}{'measured/V':>14}{'predicted/V':>14}"
          f"{'abs':>12}{'rel':>10}  ok")
    for r in rows:
        m = r.measurement
        print(f"{m.p_ref + '-' + m.n_ref:<16}{m.measured_v:>14.6g}"
              f"{r.predicted_v:>14.6g}{r.abs_err:>12.3g}{r.rel_err:>10.2%}"
              f"  {'Y' if r.ok else 'N'}")

    abs_errs = [r.abs_err for r in rows]
    rms = math.sqrt(sum(e * e for e in abs_errs) / len(abs_errs))
    print(f"\nmax abs error {max(abs_errs):.3g} V, rms {rms:.3g} V, "
          f"{sum(r.ok for r in rows)}/{len(rows)} within tolerance")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("calibrate", _cmd_calibrate),
                          ("report", _cmd_report)):
        p = sub.add_parser(name)
        p.add_argument("test_set", choices=sorted(TEST_SETS),
                       nargs="?")
        p.set_defaults(handler=handler)

    args = parser.parse_args()
    args.handler(TEST_SETS[args.test_set])


if __name__ == "__main__":
    main()
