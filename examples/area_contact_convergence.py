#!/usr/bin/env python3
"""
Reproducible convergence example for the distributed SMD area contact.

A 260 A current is forced through a 5x5 mm SMD pad on a 16x8 mm copper plane,
once with the legacy *point* coupling (the lumped element binds to the pad
centre vertex) and once with the distributed *area contact* (Robin over the pad
footprint). The script sweeps the mesh size and prints, for each model:

    h        pad in-plane power P [W]     peak J [A/mm^2]     cut current [A]

Expected: the point model's power/peak grow without bound as h -> 0 (a numerical
artifact, padne issue #77), while the distributed contact converges (pad power
is within 5% between the two finest meshes) and still conducts the full current.
`layer_cut_current` is sink-positive, so the sign says which terminal sources vs
sinks; here the pad is the source, so it reads -260 A and the magnitude is the
through-current.

Run (needs the built CGAL extension, no KiCad/pcbnew):

    .venv/bin/python examples/area_contact_convergence.py
"""

import shapely.geometry

from padne import mesh, problem, solver

PLANE = shapely.geometry.box(0, 0, 16, 8)
PAD = shapely.geometry.box(6, 1.5, 11, 6.5)
CONDUCTANCE = 2082.0          # S (1 oz Cu)
THICKNESS = 0.035             # mm
G = 9.0e4                     # S/mm^2, default SMD joint conductance
CURRENT = 260.0               # A, like the powerstage shunts
H_SWEEP = (0.30, 0.15, 0.075)


def _layer():
    return problem.Layer(
        shape=shapely.geometry.MultiPolygon([PLANE]),
        name="F.Cu", conductance=CONDUCTANCE, thickness=THICKNESS)


def solve_point(h):
    """Legacy coupling: the source extracts at a single pad-centre vertex.

    Refined *uniformly* over the whole plane, so the pad centre where the
    current is injected is actually resolved (graded refinement would leave it
    coarse and hide the artifact).
    """
    layer = _layer()
    source = problem.Connection(layer=layer, point=shapely.geometry.Point(1.0, 4.0))
    sink = problem.Connection(layer=layer,
                              point=shapely.geometry.Point(*PAD.centroid.coords[0]))
    network = problem.Network(connections=[source, sink], elements=[
        problem.CurrentSource(f=source.node_id, t=sink.node_id, current=CURRENT)])
    prob = problem.Problem(layers=[layer], networks=[network])
    return solver.solve(prob, mesh.Mesher.Config(maximum_size=h))


def solve_robin(h):
    """Distributed contact over the pad footprint, refined at the perimeter
    (the shipped default): fine at `h` across the pad rim, relaxing inward."""
    layer = _layer()
    source = problem.Connection(layer=layer, point=shapely.geometry.Point(1.0, 4.0))
    terminal = problem.NodeID()
    network = problem.Network(connections=[source], elements=[
        problem.CurrentSource(f=source.node_id, t=terminal, current=CURRENT),
        problem.AreaContact(layer=layer, shape=shapely.geometry.MultiPolygon([PAD]),
                            node=terminal, conductance_per_area=G)])
    prob = problem.Problem(
        layers=[layer], networks=[network],
        refinement_regions=[("F.Cu", shapely.geometry.MultiPolygon([PAD]))])
    config = mesh.Mesher.Config(maximum_size=1.0, pad_refine_size=h,
                                pad_refine_transition=0.5)
    return solver.solve(prob, config)


def pad_metrics(solution):
    """(in-plane pad power [W], peak volumetric |J| [A/mm^2])."""
    import numpy as np

    ls = solution.layer_solutions[0]
    power = 0.0
    j_peak = 0.0
    for msh, pd in zip(ls.meshes, ls.power_densities):
        tris = msh.triangles().astype(np.int64)
        if len(tris) == 0:
            continue
        pos = msh.positions()
        v0, v1, v2 = pos[tris[:, 0]], pos[tris[:, 1]], pos[tris[:, 2]]
        centroids = (v0 + v1 + v2) / 3.0
        cross = ((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                 - (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0]))
        area = 0.5 * np.abs(cross)
        density = np.asarray(pd.values, dtype=float)
        mask = shapely.contains_xy(PAD, centroids[:, 0], centroids[:, 1])
        power += float((density[mask] * area[mask]).sum())
        if mask.any():
            j = np.sqrt(np.maximum(CONDUCTANCE * density[mask], 0.0)) / THICKNESS
            j_peak = max(j_peak, float(j.max()))
    return power, j_peak


def cut_current(solution, h):
    return solver.layer_cut_current(solution, 0, PAD.buffer(h))


def main():
    rows = {}
    print(f"{'model':6} {'h (mm)':>7} {'pad P (W)':>12} {'peak J (A/mm2)':>15} "
          f"{'I_cut (A)':>10}")
    for mode, solve in (("point", solve_point), ("robin", solve_robin)):
        for h in H_SWEEP:
            solution = solve(h)
            power, j_peak = pad_metrics(solution)
            i_cut = cut_current(solution, h)
            rows[(mode, h)] = (power, j_peak, i_cut)
            print(f"{mode:6} {h:7.3f} {power:12.6f} {j_peak:15.1f} {i_cut:10.4f}")

    print()
    hs = sorted(H_SWEEP)
    for mode in ("point", "robin"):
        coarse, fine = rows[(mode, hs[-1])], rows[(mode, hs[0])]
        print(f"{mode:6}: h {hs[-1]} -> {hs[0]}  peak J {coarse[1]:8.1f} -> "
              f"{fine[1]:8.1f} A/mm2 (x{fine[1] / coarse[1]:.1f}); pad P "
              f"{coarse[0]:.6f} -> {fine[0]:.6f} W")
    a, b = hs[0], hs[1]           # the two finest meshes
    p_a = rows[("robin", a)][0]
    p_b = rows[("robin", b)][0]
    rel = abs(p_b - p_a) / p_a
    print(f"robin gate: pad power changes {rel * 100:.2f}% from h={b} to "
          f"h={a} -> {'PASS < 5%' if rel < 0.05 else 'FAIL'}")
    # The distributed contact must still conduct the full current; the tight
    # contour is a fixed multiple of the element size, so check the finest mesh.
    finest = rows[("robin", hs[0])][2]
    assert abs(abs(finest) - CURRENT) / CURRENT < 0.05, finest


if __name__ == "__main__":
    main()
