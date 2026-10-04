# Distributed SMD area contact: example and validation

`area_contact_convergence.py` is a self-contained, reproducible demonstration of
the distributed SMD pad contact (`AreaContact`, Robin boundary condition). It
forces 260 A through a 5x5 mm pad on a copper plane twice, once with the legacy
**point** coupling (the lumped element binds to the pad-centre vertex) and once
with the **distributed contact**, and sweeps the mesh size:

```sh
.venv/bin/python examples/area_contact_convergence.py
```

Representative output:

```
model   h (mm)    pad P (W)  peak J (A/mm2)  I_cut (A)
point    0.300    20.452456          5410.3  -260.0000
point    0.150    23.598088          7576.8  -260.0000
point    0.075    27.621230         18575.8  -260.0000
robin    0.300     0.264086          1066.8  -260.0000
robin    0.150     0.308240          1313.7  -260.0000
robin    0.075     0.319214          1323.1  -260.0000

robin gate: pad power changes 3.44% from h=0.15 to h=0.075 -> PASS < 5%
```

- **Point** coupling: in-plane pad power and peak J grow without bound as the
  pad is resolved; this is the numerical injection artifact of padne issue #77.
- **Distributed contact**: both converge (pad power meets the 5% gate by
  `h ~ lambda/2`, where `lambda = sqrt(s_sheet / g)`), and the pad still
  conducts exactly the full current (`|I_cut| = 260 A`). `layer_cut_current` is
  sink-positive, so the sign marks source vs sink; here the pad is the source.

## Board validation (Inverter-Powerstage shunts R10/R13)

The same comparison on a real board (260 A through the 100 µΩ four-terminal
shunts; board data lives in the separate `next-gen-ekart` repo):

| model | h (mm) | pad power (W) | peak J (A/mm²) | 11b I_cut |
|---|---|---|---|---|
| point (uniform) | 0.518 | 28.9 | 2862 | |
| point (uniform) | 0.260 | 33.5 | 5415 | |
| point (uniform) | 0.130 | 38.4 | 12295 | |
| robin | 0.518 (2 lambda) | 0.350 | 282 | +/-254...258 |
| robin | 0.260 (lambda) | 0.426 | 339 | +/-260 |
| robin | 0.130 (lambda/2) | **0.443** | 362 | +/-260 |

- Gate: `lambda -> lambda/2` total pad power changes **3.87% (< 5%)**; the
  default auto size `2 lambda` is 21.8% off (a screening default, as
  documented). Percentages use the finer mesh as the denominator,
  `|P(coarse) - P(fine)| / P(fine)`.
- The artifact is removed: at `h = 0.13 mm`, **38.4 W -> 0.44 W** (~87x) and
  peak J **12295 -> 362 A/mm²** (~34x).
- 11b: a contour tight around the pad carries **+/-260 A = dV/R**, confirming
  the contact conducts the right total; the sink pad is positive and the source
  pad negative. Contours at `>~0.5 mm` pull in neighbouring vias, and the bare
  pad polygon misses the one-element rim the contact spreads onto.
- Contact-conductance sweep (at fixed `h = lambda(g0)/2`): `P ~ 1/sqrt(g)`,
  with 1.333 / 0.443 / 0.122 W for `g = 9.3e3 / 9.3e4 / 9.3e5 S/mm²`. The
  ratios (3.01 / 0.276) differ slightly from `sqrt(10) = 3.16` because `h` is
  held while `lambda` moves with `g`, so the high-`g` case is under-resolved and
  biased low (the expected direction). Peak J is nearly g-independent. So the
  absolute pad power carries a `1/sqrt(g)` uncertainty; the localisation, the
  convergence, and the point-vs-distributed contrast are the robust results.
