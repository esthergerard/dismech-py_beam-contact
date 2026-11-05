# Reproducibility parameters for the paper's contact benchmarks

Every value below was read from the code/run scripts/archived outputs of this
repo (file:line given), not from memory. Items marked **FLAG** are
discrepancies or subtleties the paper text should state explicitly.

---

## 1. Common to all benchmarks

### 1.1 Contact potential and its parameters (c, delta)

Potential (both implementations are the same law):
- quadratic `E = (2r - Delta)^2` for `Delta <= 2r - delta`;
- squared log-sum-exp `E = ((1/K) log(1 + e^{K(2r - Delta)}))^2` in the band
  `(2r - delta, 2r + delta)`;
- identically zero beyond `2r + delta` (truncation cutoff).
- Edge-edge (2-node) form: `src/dismech/contact/contact_energy.py:11-36`;
  arc/Eberly/bead form (variable `delta_c`, width `d_imc`):
  `src/dismech/contact/circle_four_nodes_imc_weighted.py:23-48`.

Stiffness constant: **K = 15/delta in every production run.**
- 2-node default `k_1 = 15/delta` (`imc_energy.py:12-13`), never overridden by
  the time stepper (`time_stepper.py:79`).
- 3-node/bead: `K_imc = 15/d_imc` at every call site
  (`eberly_contact_3d.py:1715,1954`, `circle_node_bead.py:82`,
  `self_contact.py:359,518`).
- In the normalized ("barred") variables used in Appendix A the constant is
  `c = 15 r/delta` (normalization `contact_energy.py:45-57`, scale = 1/r).

delta per benchmark (state each in its section):
| benchmark | model | delta (= d_imc) | delta/r |
|---|---|---|---|
| §3 bending (r = 1 mm) | all three models | 8.0e-4 m | 0.8 |
| §4 orthogonal beams (r = 0.2 m) | 3-node | 0.02 m | 0.1 |
| §4 orthogonal beams | 2-node | delta = 0.02, h = r = 0.2 m | 0.1 |
| §5 net (r = 1.5 mm) | 2-node | 2.0e-4 m | 0.133 |
| §5 net | 3-node | 5.0e-5 m | 0.033 |
| App. A sweep (a = 1 mm) | 2-node | 1.5e-4 m (R3: 5e-5) | 0.15 (R3: 0.05) |

**FLAG (§3):** in the 2D circle model (`circleFourNodesIMCWeightedXY`)
`d_imc = 0.0008` is HARD-CODED (`circle_four_nodes_imc_weighted.py:487-488`
and `:712-714`), not a benchmark input.
**FLAG (App. A):** the sliding sweep overrides the default and uses
`c = 50` dimensionless (`harness_slide_2node.py:75,145`), while the static
apex probe keeps the library default `K1 = 15/delta_bar = 100`
(`apex_multiplicity_probe.py:52`, `probe_run.log:4`). Two different
smoothing sharpnesses — say so in the appendix.

### 1.2 Contact stiffness k_c

- Initialised at `kc = 1.0` in every benchmark.
- **Adaptive update is ON by default and used in §3, §4, §5.** Rule (first
  Newton iteration of each step):
  `k_c <- 0.9 k_c + 0.1 (max nodal elastic force) * kc_user`
  (`contact_energy.py:156-166` for 2-node — unconditional;
  `time_stepper.py:563-600` for Eberly beam-to-beam, gated by
  `gap_min < d_imc` and decaying back to `kc_user` when contact is lost).
- **FLAG:** the driver is the max NODAL FORCE over the structure (an
  exponential smoothing of it), with a global min-gap activation gate in the
  Eberly/circle/bead paths — not "a global minimum distance". Word the paper
  accordingly.
- Exceptions: Appendix A fixes `k_c = 1` (adaptive disabled by a subclass,
  `harness_slide_2node.py:95-100`); §4 has one archived run with
  `fixed_kc = 1.0e4` (the paper's K_N) — `zavarise_summary_steps6_trace_eberly_mu0_kc1e4.txt`;
  §5 supports `--fixed-kc` but the campaign left it adaptive.

### 1.3 Hybrid Jacobian scheme

**Not implemented in dismech-py.** The contact Hessian is assembled at every
Newton iteration (`contact_energy.py:137-138`, `eberly_contact_3d.py:2046`);
the only things frozen on the first iteration are the broad-phase candidate
list, the k_c update, and the friction pair list. No iteration-count gating
of the contact Hessian exists anywhere in `src/`. The paper can state: "the
hybrid Jacobian scheme of the original IMC implementation is not used; the
contact Hessian is assembled at every iteration."

### 1.4 Broad-phase cutoff g_cut

- 2-node edge-edge: candidate pairs kept when
  `Delta < 2h + 10 delta` (`contact_energy.py:103-104`), i.e.
  **g_cut = 2r + 10 delta**, evaluated once per time step (first Newton
  iteration) and frozen for the rest of the step.
- 3-node Eberly: bounding-sphere culling with interaction range
  `2r + d_imc` (no extra margin), `eberly_contact_3d.py:1955-1956,269-291`.
- Stencil pre-filter: neighbouring edges within `min_contact_edge_gap = 3`
  edges are excluded from self-contact pairs (`params.py:45`).
- §4 additionally sets `cross_beam_only_contact=True`
  (`contact_bending_soft_soft.py:534`) — required for meshes finer than
  ~N_x = 40.

### 1.5 Straight-edge fallback (3-node model)

- Hard guard: arc fit raises when `|e_a x e_b|^2 < 1e-24`
  (`eberly_contact_3d.py:311-314`); such a pair is skipped unless the line
  fallback is enabled via `straight_R_threshold` (default **None = off**,
  `eberly_contact_3d.py:1910,2009-2024`).
- §3/§4 avoid collinearity by seeding a small initial curvature
  (kappa0 = -1e-3 resp. 1e-4 1/m).
- §5 wraps the arc fit in `_safe_circle_from_three_nodes`
  (`contact_net_pullout.py:622-656`): near-collinear triplets get a synthetic
  arc with `R = 1e6 x chord` instead of being dropped; counters
  `n_arc_fit_fail` / `n_fallback_used` and one-shot warnings
  (`contact_net_pullout.py:681-684,790-808`).
- **Count: zero.** No fallback/arc-fit warning appears in any archived
  cluster log (11 logs, incl. two complete Phase A-E runs). The paper can
  state the fallback never triggered in the production runs.

### 1.6 Time integrator and Newton tolerance

- Integrator: **backward (implicit) Euler** in all contact benchmarks
  (`ImplicitEulerTimeStepper`; inertia `F = (M/dt)((q-q_n)/dt - u_n)`,
  `implicit_euler.py:11-13`).
- Convergence: any of `err < tol`, `err < err_0 * ftol`,
  `||dq||/dt < dtol`, `iter >= max_iter` (`time_stepper.py:695-709`), where
  `err = ||residual on free DOFs||_2`.
- Per benchmark:
  | benchmark | tol = ftol = dtol | max_iter | line search |
  |---|---|---|---|
  | §3 2D (N-sweep script) | 1e-5 | 50 | off |
  | §3 3D + bead | 1e-4 | 100 | off |
  | §4 | 1e-4 | 100 (mu=0) / 200 (mu>0) | **off** (`contact_bending_soft_soft.py:505`) |
  | §5 | 1e-4 | 200 | **on**, active from iteration 0 (`--use-line-search --line-search-iters 0`) |
- Linear solver: dense NumPy in §3/§4 (default), Pardiso in §5 (`SOLVER=pardiso`).

### 1.7 One-node (bead) model radius convention

`bead_radius = L / (2 (N-1))` — half the inter-node spacing, so adjacent
beads are tangent in the undeformed rod; **independent of the rod
cross-section radius** (`circle_node_bead.py:10-15,58-60`;
`contact_bending_test_3D_bead.py:154`). For §3 with N = 50, L = 1 m:
bead_radius = 10.204 mm vs rod radius 1 mm. The gap is
`||X_i - C|| - R_obs - bead_radius`, and the potential is evaluated with the
bead radius playing the role of r.

---

## 2. §3 — three-point bending

| quantity | value | source |
|---|---|---|
| E | 2.1e10 Pa (`210e8`) — **FLAG: code comment says "210 GPa" but the number is 21 GPa**; postprocessing normalization consistently uses 210e8 | `contact_bending_test.py:55`, `fig2_force_displacement.py:52` |
| nu | 0.3 | `contact_bending_test.py:56` |
| rho | 7850 kg/m^3 nominal, **effective 0.0785 kg/m^3**: quasi-static rescaling `k_inext = 1e3` (A_eff = 1000A, EI unchanged) and `m_quasi = 0.01` -> `rho_eff = rho/k_inext*m_quasi` | `contact_bending_test.py:64-65,271-276` |
| dt | 1.0e-3 s (2D script) / 1.0e-2 s (3D Eberly + bead) | `contact_bending_test.py:76`, `_3D.py:75`, `_3D_bead.py:67` |
| steps | total_time = 1000 s -> 1e6 (2D) / 1e5 (3D, bead) steps; N=50 bead archive stops at t = 963.31 s (wall-time) | scripts + `reg_beam_coords_N50_bead.txt` |
| loading | prescribed displacement: clamp (nodes 0,1) moved at **v = 1.0e-3 m/s** -> 1e-6 m/step (2D) / 1e-5 m/step (3D, bead); nominal cap 1.2 m, reached ~0.96-1.0 m | `contact_bending_test.py:72-73,407-422` |
| damping | `eta = 1.0`, but the rod-node implementation applies `0.005*eta` -> **effective 5e-3 N s/m per node** (`F = -0.005 eta u`) | `contact_bending_test.py:330`, `src/dismech/external_forces/damping.py:47,65` |
| dynamic vs static | **dynamic implicit-Euler run with slow loading** (`static_sim=False`); quasi-staticity from m_quasi = 0.01 + damping + slow rate; final max velocity printed as check | `contact_bending_test.py:305,345,556-559` |
| N (main curves) | 50 for Figs 1/2/4 (2-node curve from external npz; 3-node N=50 from the MATLAB-postprocessed dataset); mesh sweep N = 5-300 | `fig2_force_displacement.py:70-87`, `fig3_mesh_convergence.py:42-61` |
| contact | kc = 1.0 adaptive (all models); d_imc = 8e-4 m; K = 15/d_imc = 18750 | see §1.1-1.2 |
| geometry | L = 1.0 m, r = 1 mm, obstacle center (0.8, 0), R_obs = 0.2 m; beam seeded with kappa0 = -1e-3 | `contact_bending_test.py:27-51,87-122` |
| **F in delta/L = F L^2/(48 E I)** | plotted F = **total vertical contact force summed over all beam nodes** = the single obstacle reaction of the half-model. The analytical full-model central load is divided by 2 to compare (`ANALYTICAL_FACTOR = 0.5`). Plot axes use `F L^2/(E I)` (no 48); the 48 only appears in the fig6 slope conversion. **Normalization length L = 1.6 m (not the coded 1.0 m)** so that R_obs/L = 0.125. Curves re-zeroed at contact onset. | `fig2_force_displacement.py:52,94,107-142,156`, `fig6_jitter_metrics.py:68-70` |

## 3. §4 — two orthogonal beams (Zavarise & Wriggers Ex. 2)

| quantity | value | source |
|---|---|---|
| dt | 1.0e-2 s in all main runs; **stable-dt ladder 0.01/0.02/0.04/0.08/0.15/0.30 -> all (N, model) complete at 0.08 s and fail at 0.15 s** | `contact_bending_soft_soft.py:145`, `faster_implementation/bench_stable_dt.py:49`, `bench_stable_dt_results.jsonl` |
| damping | `eta = 10.0` -> effective **0.05 N s/m per rod node** (x0.005 factor, §2 note); plus m_quasi = 1e-3 (rho_eff = 7.85) | `contact_bending_soft_soft.py:132,592`, `damping.py:47` |
| N_X, Fig 7(c) trace | **21** (n_z = 29); same for the jitter table; mesh-convergence panels use 11/21/51/101 (mu=0) and 21/51/101 (mu=0.1) | `postprocessing/fig_soft_soft.py:18-19,70-89` |
| load steps | 6 "load steps" are **one continuous constant-velocity ramp** of clamp A: v_y = +0.3 m/s (approach), v_z = -0.03 m/s (slide), each step = 1.0 s = 100 dt-increments; equilibrium snapshot at the end of each second. The t ~ 2 s transient in Fig 7(c) is **contact engagement** (0.6 m initial surface gap closes during step 2), not a step discontinuity. | `contact_bending_soft_soft.py:135-146,313-314,655-684` |
| units | the reference case is dimensionless; the script relabels the same numbers as SI (m, Pa, N, s). rho = 7850 and eta = 10 are additional, genuinely dimensional choices. **FLAG:** the paper should either quote v_tol in the same consistent units (0.001 length/time) or say the SI relabelling is used throughout; also A = 4e-2 is not consistent with r = 0.2 (flagged in code, `:121`) — it is the paper's own data. | `contact_bending_soft_soft.py:10-22,96-128` |
| v_tol | 1.0e-3 (m/s), smooth-Coulomb stick-slip gate; sweep runs at 1e-3/1e-4/1e-5 archived. Sliding speed 0.03 m/s -> v_tol/v_slide = 1/30 | `contact_bending_soft_soft.py:221-225`, summaries `vt3/vt4/vt5` |
| line search | **disabled** (`use_line_search=False`) — state in text, not only Table 3 | `contact_bending_soft_soft.py:505` |
| Newton / solver | tol = ftol = dtol = 1e-4; max_iter 100 (mu=0) / 200 (mu>0); dense NumPy solver; implicit Euler; `cross_beam_only_contact=True` | `contact_bending_soft_soft.py:501-535,615` |
| contact | Eberly: kc = 1.0 adaptive, d_imc = 0.02 m (= 0.1 r); 2-node: delta = 0.02, h = 0.2; one fixed-kc = 1e4 control run archived; mu run set {0, 0.1, 0.8} (0.6, 1.5 exploratory) | `contact_bending_soft_soft.py:544-570` |
| geometry | Beam X: L = 10 m, both ends clamped; Beam Z: L = 14 m, clamped at driven end A, free at B; crossing 10 m from A / 4 m from B; initial axis gap 1.0 m (surface gap 0.6 m); r = 0.2 m both; kappa0 = 1e-4 seed | `contact_bending_soft_soft.py:96-120,602-613` |

## 4. §5 — woven net pull-out

| quantity | value | source |
|---|---|---|
| Phase A clamp displacement | **3.75 mm per side** at constant rate over t in [0, 1] s (3.75 mm/s per side) — belongs in §5.1, not only App. C | campaign scripts `END_DISP_ENGAGE=3.75e-3`, `T_ENGAGE=1.0` |
| phase schedule | A engage 0-1 s; B relax 1-2 s; C release 2-3 s (central x-beam left clamp moved back 3.75 mm); D free settle 3-3.5 s; E pull-out 3.5-15.5 s | `contact_net_pullout.py:506-511`, defaults `:189-196` |
| pull-out speed (E) | 30 mm over 12 s = **2.5 mm/s** | `END_DISP_PULLOUT=30e-3`, `T_PULLOUT=12` |
| dt | 1.0e-2 s (phases A, B); 5.0e-3 s (C, D via `--dt-release`); 5.0e-3 s (E) | campaign scripts `DT`, `DT_RELEASE`, `DT_PULLOUT` |
| damping | `eta = 500` -> effective **2.5 N s/m per rod node** (x0.005, `damping.py:47`) | `DAMPING_ETA=500`, `contact_net_pullout.py:992` |
| Newton | tol = ftol = dtol = 1e-4, max_iter = 200; line search ON from iteration 0; Pardiso solver | `contact_net_pullout.py:211-212,930-932`, scripts `LINE_SEARCH_ITERS=0`, `SOLVER=pardiso` |
| contact | kc = 1.0 adaptive (campaign `FIXED_KC` unset); d_imc = 2e-4 m (2-node) / 5e-5 m (3-node); 3-node force scaled by 1/r (`--eberly-h-scale`); mu = 0; vel_tol 1e-3 m/s (inactive at mu = 0) | mesh-campaign script + `contact_net_pullout.py` |
| N range | 21-131 (both models), (N-1) multiple of 5 | campaign script |

## 5. Appendix A — sliding sweep + apex probe

Sweep (`harness_slide_2node.py`):
- Potential: squared-log IMC with **c = 50** (dimensionless; k_1 = 5e4 1/m),
  **k_c = 1 fixed** (adaptive disabled), delta_bar = 0.15 (R3: 0.05).
- Geometry: rod radius **a = 1 mm**, cylinder = same radius (single-radius
  2-node model), centerline clearance **1.95 a** (0.05 a penetration),
  L_rod = 0.5 m, cylinder edge 0.2 m along z.
- Runs: R1 baseline kappa = 5 /m, N = 50 (h = 10.204 mm); R2 kappa = 0;
  R3 delta_bar = 0.05; R4 refresh every sample; R5 N = 100 (h = 5.051 mm);
  R6 N = 200 (h = 2.513 mm).
- Sampling: 12 h sweep at 1000 samples/h (12001 samples); metric-B
  decimations **h/10, h/100, h/1000**; candidate set frozen for 10 samples
  (= 100 emulated steps per h) except R4 (every sample). Broad-phase margin
  10 delta as in production.
- Metric A: residual after 3h-boxcar moving average (3001 samples,
  reflected ends), peak-to-peak; FFT with Hann window, frequency in
  cycles/h.

Static apex probe (`apex_multiplicity_probe.py`, `probe_run.log`):
- a = 1 mm, edge length H = 10 mm, 13 nodes, apex at node 6.
- Gap: 0.05 a penetration -> centerline approach Delta_bar = 1.95 (same as
  sweep).
- delta = 0.15 a with default **K1 = 100** (differs from the sweep's c = 50
  — FLAG, state both).
- Apex wedge: parabola kappa = 5 /m -> interior angle 177.14 deg
  (half-slope 1.432 deg); "straight" control kappa = 0.02 /m (179.99 deg).
- 3-node comparison uses d_imc = 0.8 a, K = 15/d_imc; sphere obstacle
  radius a. 601 samples over 3 edge lengths.
- Headline numbers: 2-node multiplicity M = 2 at a shared node, M = 4 at a
  facing apex pair; 3-node M = 1 within 1.2e-2.

## 6. §5.2 — Abaqus solid finite-element reference (net pull-out)

**Updated for this repo** (the original version of this section described
an earlier C3D4/frictionless/no-damping/Phase-A-only validation run that
has since been superseded by the full Phase A-E quadratic-element study
below; that earlier run is not part of this repo).

dismech side (alternated soft recipe, three-node model, N=21..131):
- **dt = 1.0e-2 s** (phases A, B), **5.0e-3 s** (C, D, E);
- **eta = 500** (effective 2.5 N s/m per rod node, x0.005 factor);
- **Newton tol = ftol = dtol = 1e-4**, max_iter 200, line search on
  (iteration 0), Pardiso.

Abaqus side (`net_pullout/abaqus/quadratic_C3D10M_alpha1000/`):
- **C3D10M quadratic tetrahedra, 298,586 elements / 480,770 nodes**
  (mesh shared with every alpha variant, generated by `net_pullout_gmsh.py`
  with `Mesh.MeshSizeFromCurvature=24`), SI units;
- Abaqus/Explicit, general contact over all exterior surfaces, default
  hard normal behaviour, no contact stabilization;
- **mass-proportional (Rayleigh) damping, alpha = 1000 s^-1, beta = 0**
  — the value used in the paper; see
  `quadratic_C3D10M_alpha1000/README.md` for the energy-based argument
  for this choice (an alpha in {1, 10, 50, 100} sweep showed no
  measurable effect above Abaqus/Explicit's own bulk-viscosity
  dissipation; alpha=1000 is the smallest value producing a measurable
  one);
- the same five loading phases (A-E) as the rod model, `*Fixed Mass
  Scaling, dt=1.5e-6`;
- field output at each phase boundary plus intermediate snapshots
  (`*_final_shape.csv`, `*_shape_t*.csv`); history output for the
  central-strand clamp force and energies
  (`*_central_clamp.csv`, `*_energies.csv`).
